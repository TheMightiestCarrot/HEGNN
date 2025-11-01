import pickle
import sys
from pathlib import Path

import os
import random

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from utils.seed import fix_seed
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        def fix_seed(seed: int = 0) -> None:
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
    else:
        raise


@pytest.fixture(autouse=True)
def _set_deterministic_seed():
    fix_seed(0)


def _write_partition_files(root: Path, dataset_name: str, partition: str, num_systems: int, num_frames: int, num_nodes: int) -> None:
    suffix = f"{partition}_charged{dataset_name}"
    loc = np.linspace(
        0,
        1,
        num_systems * num_frames * num_nodes * 3,
        dtype=np.float32,
    ).reshape(num_systems, num_frames, num_nodes, 3)
    vel = np.linspace(
        1,
        2,
        num_systems * num_frames * num_nodes * 3,
        dtype=np.float32,
    ).reshape(num_systems, num_frames, num_nodes, 3)
    charges = np.full((num_systems, num_nodes, 1), 1.0, dtype=np.float32)

    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    edges_array = np.asarray(edges, dtype=np.int64)

    np.save(root / f"loc_{suffix}.npy", loc)
    np.save(root / f"vel_{suffix}.npy", vel)
    np.save(root / f"charges_{suffix}.npy", charges)
    np.save(root / f"edges_{suffix}.npy", edges_array)

    cfg = {"num_nodes": num_nodes, "num_frames": num_frames}
    with open(root / f"cfg_{suffix}.pkl", "wb") as handle:
        pickle.dump(cfg, handle)


def _prepare_dataset(root: Path, dataset_name: str, partitions) -> None:
    for partition in partitions:
        _write_partition_files(root, dataset_name, partition, num_systems=2, num_frames=2, num_nodes=3)


@pytest.fixture
def synthetic_nbody_dir(tmp_path):
    dataset_name = "mini"
    root = tmp_path / "nbody"
    root.mkdir()
    _prepare_dataset(root, dataset_name, ("train", "valid", "test"))
    return str(root), dataset_name
