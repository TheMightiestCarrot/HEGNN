#!/usr/bin/env python3
"""
Command-line helper to estimate coarse Δt and velocity-loss weight for N-body runs.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hyperparam_tuning import (  # noqa: E402
    auto_tune_coarse_dt_and_weight,
    save_tuning_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate optimal coarse Δt and velocity-loss weight from dataset statistics.",
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        required=True,
        help="Directory containing loc_*/vel_* tensors.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset suffix (e.g. 5_0_0).",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default="train",
        help="Comma-separated list of partitions to analyse (default: train).",
    )
    parser.add_argument(
        "--frame_0",
        type=int,
        default=30,
        help="Start frame index (default: 30).",
    )
    parser.add_argument(
        "--frame_T",
        type=int,
        default=40,
        help="Target frame index (default: 40).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of systems to sample.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON report. If omitted, only stdout is used.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
    report = auto_tune_coarse_dt_and_weight(
        data_dir=args.data_directory,
        dataset_name=args.dataset_name,
        frame_0=args.frame_0,
        frame_T=args.frame_T,
        partitions=partitions,
        max_samples=args.max_samples,
    )

    if args.output is not None:
        save_tuning_report(report, args.output)

    dump_kwargs = {"indent": 2} if args.pretty else {}
    print(json.dumps(report, **dump_kwargs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
