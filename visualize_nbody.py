#!/usr/bin/env python3
"""
Play back and export synthetic n-body trajectories from the HEGNN dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a synthetic n-body trajectory and export it as an MP4 animation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/nbody/data_small"),
        help="Directory that stores the *.npy/*.pkl files for the dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="5_0_0",
        help="Dataset suffix that follows 'charged' in the file names (see datasets/nbody/data_*).",
    )
    parser.add_argument(
        "--partition",
        choices=("train", "valid", "test"),
        default="valid",
        help="Which partition to visualise.",
    )
    parser.add_argument(
        "--simulation-index",
        type=int,
        default=0,
        help="Select which simulation inside the tensor to play (0-indexed).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to include in the playback (0-indexed, inclusive).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Frame index to stop at (exclusive). Leave empty to use the full rollout.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=10,
        help="How many previous frames to keep for trajectory trails (0 disables, -1 keeps all past frames).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Animation frames-per-second.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(6.0, 6.0),
        help="Matplotlib figure size in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution when exporting frames.",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=1800,
        help="Target bitrate (kbps) for the MP4 writer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the MP4 file to create. Defaults to ./nbody_<partition>_<dataset>_sim<idx>.mp4.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window after (or without) saving the animation.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Label bodies with their index in each frame.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="Elevation angle for the 3D camera.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=45.0,
        help="Azimuth angle for the 3D camera.",
    )
    return parser.parse_args()


def build_prefix(args: argparse.Namespace) -> str:
    return f"{args.partition}_charged{args.dataset_name}"


def load_arrays(root: Path, prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    loc_path = root / f"loc_{prefix}.npy"
    charges_path = root / f"charges_{prefix}.npy"

    if not loc_path.exists():
        raise FileNotFoundError(
            f"Could not locate trajectory tensor: {loc_path}. "
            "Double-check --dataset-root, --partition, and --dataset-name."
        )

    loc = np.load(loc_path)
    if loc.ndim != 4 or loc.shape[-1] != 3:
        raise ValueError(f"Unexpected loc tensor shape {loc.shape}; expected (num_sims, T, num_bodies, 3).")

    if not charges_path.exists():
        raise FileNotFoundError(
            f"Charges file missing: {charges_path}. Required for colouring the bodies."
        )

    charges = np.load(charges_path)
    if charges.ndim not in (2, 3):
        raise ValueError(f"Unexpected charges tensor shape {charges.shape}; expected (num_sims, num_bodies, [1]).")

    if charges.ndim == 3 and charges.shape[-1] == 1:
        charges = charges[..., 0]

    if charges.shape[0] != loc.shape[0] or charges.shape[1] != loc.shape[2]:
        raise ValueError(
            f"Charges tensor shape {charges.shape} incompatible with trajectories {loc.shape}."
        )

    return loc, charges


def validate_indices(loc: np.ndarray, args: argparse.Namespace) -> Tuple[int, int, int]:
    num_systems, num_frames, _, _ = loc.shape

    if args.simulation_index < 0 or args.simulation_index >= num_systems:
        raise IndexError(
            f"--simulation-index {args.simulation_index} out of range for dataset containing {num_systems} simulations."
        )

    start = int(args.start_frame)
    end = num_frames if args.end_frame is None else int(args.end_frame)

    if start < 0 or start >= num_frames:
        raise IndexError(f"--start-frame {start} must fall within [0, {num_frames}).")
    if end <= start or end > num_frames:
        raise IndexError(f"--end-frame {end} must satisfy {start + 1} <= end <= {num_frames}.")

    return num_systems, start, end


def format_output_path(args: argparse.Namespace, prefix: str) -> Path:
    if args.output is not None:
        return args.output
    filename = f"nbody_{prefix}_sim{args.simulation_index}.mp4"
    return Path(filename)


def compute_axis_limits(points: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    max_span = float(np.max(spans))
    if max_span == 0.0:
        max_span = 1.0
    margin = max_span * 0.1

    limits = []
    for axis in range(3):
        center = 0.5 * (maxs[axis] + mins[axis])
        half_range = 0.5 * max_span + margin
        limits.append((center - half_range, center + half_range))
    return tuple(limits)  # type: ignore[return-value]


def make_trail_lines(
    ax, num_bodies: int, colours: Iterable[str]
) -> List:
    lines = []
    for body_idx, colour in zip(range(num_bodies), colours):
        (line,) = ax.plot([], [], [], lw=1.2, alpha=0.7, color=colour)
        lines.append(line)
    return lines


def main() -> None:
    args = parse_args()
    prefix = build_prefix(args)

    try:
        loc, charges = load_arrays(args.dataset_root, prefix)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        _, start, end = validate_indices(loc, args)
    except (IndexError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    positions = loc[args.simulation_index, start:end]  # (T, num_bodies, 3)
    charges_sim = charges[args.simulation_index]  # (num_bodies,)
    num_frames = positions.shape[0]
    num_bodies = positions.shape[1]

    colours = np.where(charges_sim >= 0.0, "tab:red", "tab:blue")

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    fig = plt.figure(figsize=tuple(args.figsize))
    ax = fig.add_subplot(111, projection="3d")
    fig.tight_layout()

    flat_positions = positions.reshape(-1, 3)
    (x_lim, y_lim, z_lim) = compute_axis_limits(flat_positions)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"N-body trajectory · sim {args.simulation_index} · frames {start}–{end - 1}"
    )

    scatter = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        positions[0, :, 2],
        s=80,
        c=colours,
        depthshade=False,
        edgecolors="k",
        linewidths=0.5,
    )

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    annotations: List = []
    if args.annotate:
        for body_idx in range(num_bodies):
            annotations.append(
                ax.text(
                    positions[0, body_idx, 0],
                    positions[0, body_idx, 1],
                    positions[0, body_idx, 2],
                    f"{body_idx}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=colours[body_idx],
                )
            )

    trail_lines: List = []
    if args.trail_length != 0:
        trail_lines = make_trail_lines(ax, num_bodies, colours)

    def update(frame_index: int):
        frame_pos = positions[frame_index]
        scatter._offsets3d = (
            frame_pos[:, 0],
            frame_pos[:, 1],
            frame_pos[:, 2],
        )

        if annotations:
            for body_idx, text in enumerate(annotations):
                text.set_position((frame_pos[body_idx, 0], frame_pos[body_idx, 1]))
                text.set_3d_properties(frame_pos[body_idx, 2], zdir="z")

        if trail_lines:
            if args.trail_length < 0:
                start_idx = 0
            else:
                start_idx = max(0, frame_index - args.trail_length)
            segment = positions[start_idx : frame_index + 1]
            for body_idx, line in enumerate(trail_lines):
                line.set_data(segment[:, body_idx, 0], segment[:, body_idx, 1])
                line.set_3d_properties(segment[:, body_idx, 2])

        time_text.set_text(f"frame {start + frame_index:03d} / {start + num_frames - 1:03d}")
        artists = [scatter, time_text]
        artists.extend(trail_lines)
        artists.extend(annotations)
        return artists

    interval = int(round(1000 / max(1, args.fps)))
    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        blit=False,
        repeat=True,
    )

    exit_code = 0
    output_path = format_output_path(args, prefix)
    if args.fps <= 0:
        print("[warning] --fps must be positive to export MP4; skipping export.", file=sys.stderr)
    else:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = FFMpegWriter(fps=args.fps, bitrate=args.bitrate)
            animation.save(str(output_path), writer=writer, dpi=args.dpi)
            print(f"[info] Saved animation to {output_path}")
        except (RuntimeError, FileNotFoundError) as exc:
            print(
                "[error] Failed to export MP4 via ffmpeg. "
                "Install ffmpeg or pass --show to preview without saving.",
                file=sys.stderr,
            )
            print(f"        {exc}", file=sys.stderr)
            exit_code = 1

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
