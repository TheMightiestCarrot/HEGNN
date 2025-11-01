import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _require_torch():
    if torch is None:
        pytest.skip("PyTorch is required for rotation sanity checks.")


def test_random_rotate_returns_orthonormal_matrix():
    _require_torch()
    from utils.rotate import random_rotate

    rotation = random_rotate().to(torch.float32)
    assert rotation.shape == (3, 3)

    identity = torch.eye(3, dtype=torch.float32)
    product = rotation @ rotation.T
    assert torch.allclose(product, identity, atol=1e-5)

    det = torch.linalg.det(rotation)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)
