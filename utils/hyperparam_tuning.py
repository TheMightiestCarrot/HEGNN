import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _load_nbody_traces(
    data_dir: Path,
    dataset_name: str,
    partition: str,
    frame_0: int,
    frame_T: int,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load position/velocity trajectories for the requested split and frames."""
    suffix = f"{partition}_charged{dataset_name}"
    loc = np.load(data_dir / f"loc_{suffix}.npy")
    vel = np.load(data_dir / f"vel_{suffix}.npy")

    if max_samples is not None:
        loc = loc[:max_samples]
        vel = vel[:max_samples]

    loc_0 = loc[:, frame_0]
    loc_t = loc[:, frame_T]
    vel_0 = vel[:, frame_0]
    vel_t = vel[:, frame_T]

    return (loc_0, loc_t), (vel_0, vel_t)


def _safe_trim_mean(samples: np.ndarray, lower: float, upper: float) -> float:
    """Compute trimmed mean, guarding against degenerate sample sizes."""
    lower_idx = int(np.floor(lower * samples.size))
    upper_idx = int(np.ceil(upper * samples.size))
    if upper_idx <= lower_idx:
        return float(np.mean(samples))
    trimmed = np.sort(samples)[lower_idx:upper_idx]
    if trimmed.size == 0:
        return float(np.mean(samples))
    return float(np.mean(trimmed))


def estimate_coarse_dt_from_dataset(
    data_dir: str,
    dataset_name: str,
    partition: str = "train",
    frame_0: int = 30,
    frame_T: int = 40,
    max_samples: Optional[int] = None,
    clamp_positive: bool = True,
    denom_eps: float = 1e-9,
) -> Dict[str, float]:
    """Estimate the symplectic step Δt that best matches the dataset kinematics.

    The estimator solves, for each sample i,
        argmin_{Δt} ||loc_T - loc_0 - Δt * vel_T||_2^2
     which yields Δt_i = <disp_i, vel_T_i> / ||vel_T_i||^2.
    We aggregate the per-sample values with robust statistics.
    """
    loc_pair, vel_pair = _load_nbody_traces(
        Path(data_dir), dataset_name, partition, frame_0, frame_T, max_samples
    )
    loc_0, loc_t = loc_pair
    _, vel_t = vel_pair

    disp = loc_t - loc_0
    numerator = np.sum(disp * vel_t, axis=(1, 2))
    denom = np.sum(vel_t * vel_t, axis=(1, 2))

    valid_mask = denom > denom_eps
    dt_samples = numerator[valid_mask] / denom[valid_mask]

    if clamp_positive:
        dt_samples = dt_samples[dt_samples > 0]

    if dt_samples.size == 0:
        raise RuntimeError("No valid Δt estimates found; check dataset configuration.")

    dt_mean = float(np.mean(dt_samples))
    dt_median = float(np.median(dt_samples))
    dt_trim = _safe_trim_mean(dt_samples, 0.1, 0.9)
    dt_std = float(np.std(dt_samples))
    dt_low, dt_high = np.percentile(dt_samples, [5.0, 95.0])

    return {
        "mean": dt_mean,
        "median": dt_median,
        "trimmed_mean": dt_trim,
        "std": dt_std,
        "p05": float(dt_low),
        "p95": float(dt_high),
        "num_samples": int(dt_samples.size),
    }


def suggest_velocity_loss_weight(dt_estimate: float, min_weight: float = 1e-4, max_weight: float = 10.0) -> float:
    """Suggest λ_vel so that λ_vel * ||Δv||² is comparable to ||Δx||² at scale Δx ≈ Δt · Δv."""
    weight = max(dt_estimate * dt_estimate, min_weight)
    return float(min(weight, max_weight))


def auto_tune_coarse_dt_and_weight(
    data_dir: str,
    dataset_name: str,
    frame_0: int = 30,
    frame_T: int = 40,
    partitions: Iterable[str] = ("train",),
    max_samples: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Estimate Δt statistics across partitions and propose a velocity-loss weight."""
    stats = {}
    for partition in partitions:
        stats[partition] = estimate_coarse_dt_from_dataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            partition=partition,
            frame_0=frame_0,
            frame_T=frame_T,
            max_samples=max_samples,
        )

    # Use the median across partitions as our canonical Δt.
    canonical_dt = float(np.median([item["median"] for item in stats.values()]))
    suggested_weight = suggest_velocity_loss_weight(canonical_dt)

    return {
        "dt_stats": stats,
        "suggested_coarse_dt": canonical_dt,
        "suggested_loss_vel_weight": suggested_weight,
    }


def save_tuning_report(report: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Persist tuning results as JSON for quick reference."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as handle:
        json.dump(report, handle, indent=2)
