import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _require_torch():
    if torch is None:
        pytest.skip("PyTorch is required for dataset shape checks.")


def test_nbody_dataset_shapes(synthetic_nbody_dir):
    _require_torch()
    from datasets.nbody.dataset import NBodySystemDataset

    data_dir, dataset_name = synthetic_nbody_dir
    dataset = NBodySystemDataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        virtual_channels=1,
        partition="train",
        max_samples=2,
        frame_0=0,
        frame_T=1,
        cutoff_rate=0.0,
        device="cpu",
    )

    assert len(dataset) == 2

    sample = dataset[0]

    assert sample.loc_0.shape == (3, 3)
    assert sample.loc_t.shape == (3, 3)
    assert sample.vel_0.shape == (3, 3)
    assert sample.node_feat.shape == (3, 2)
    assert sample.edge_index.shape[0] == 2
    assert sample.edge_attr.shape[1] == 1

    loc_mean = sample.loc_mean.squeeze(-1).squeeze(0)
    expected_mean = sample.loc_0.mean(dim=0)
    assert torch.allclose(loc_mean, expected_mean, atol=1e-6)

    velocity_norm = torch.sqrt((sample.vel_0 ** 2).sum(dim=1, keepdim=True))
    assert torch.allclose(sample.node_feat[:, :1], velocity_norm, atol=1e-6)

    charge_feature = sample.node_feat[:, 1]
    assert torch.allclose(charge_feature, torch.ones_like(charge_feature))
