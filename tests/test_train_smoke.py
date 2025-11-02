import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError:
    DataLoader = None


def _require_dependencies():
    if torch is None:
        pytest.skip("PyTorch is required for the training smoke test.")
    if DataLoader is None:
        pytest.skip("torch_geometric is required for the training smoke test.")


def test_train_single_epoch_smoke(synthetic_nbody_dir):
    _require_dependencies()
    from torch import nn

    from datasets.nbody.dataset import NBodySystemDataset
    from models.HEGNN import HEGNN
    from utils.train import train_single_epoch

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
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = HEGNN(
        num_layer=1,
        node_input_dim=2,
        edge_attr_dim=2,
        hidden_dim=8,
        max_ell=1,
        device="cpu",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    avg_loss, pos_err, pos_mae, vel_pct_err, vel_mae, vel_rmse, vel_loss = train_single_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss=loss_fn,
        sigma=1.0,
        weight=0.01,
        epoch_index=1,
        backprop=True,
        tag="train",
        sample=1,
        device="cpu",
    )

    metrics = torch.tensor(
        [avg_loss, pos_err, pos_mae, vel_pct_err, vel_mae, vel_rmse, vel_loss],
        dtype=torch.float32,
    )
    assert torch.isfinite(metrics).all()
    assert avg_loss >= 0.0
