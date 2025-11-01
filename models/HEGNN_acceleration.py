from __future__ import annotations

from functools import partial
from typing import Optional

import torch
from torch import nn
from torch_scatter import scatter

import e3nn


class BaseMLP(nn.Module):
    """Simple 2-layer MLP with optional residual connection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: nn.Module = nn.SiLU(),
        residual: bool = False,
        last_act: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation if last_act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x) if self.residual else self.mlp(x)


class HEGNN_Layer(nn.Module):
    """Single HEGNN layer operating on latent node features and steerable tensors."""

    def __init__(
        self,
        edge_attr_dim: int,
        hidden_dim: int,
        sh_irreps: e3nn.o3.Irreps,
        activation: nn.Module = nn.SiLU(),
    ) -> None:
        super().__init__()
        self.sh_irreps = sh_irreps
        mlp_factory = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)

        # Messages use invariant scalars including SH inner products.
        self.mlp_msg = mlp_factory(
            input_dim=2 * hidden_dim + edge_attr_dim + 1 + sh_irreps.lmax + 1,
            output_dim=hidden_dim,
            last_act=True,
        )
        self.mlp_node_feat = mlp_factory(
            input_dim=hidden_dim + hidden_dim,
            output_dim=hidden_dim,
        )

        self.sh_msg = SH_Msg(sh_irreps)
        self.sh_coff = e3nn.o3.FullyConnectedTensorProduct(
            self.sh_irreps, "1x0e", self.sh_irreps, shared_weights=False
        )
        self.mlp_sh = mlp_factory(
            input_dim=hidden_dim,
            output_dim=self.sh_coff.weight_numel,
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        node_sh: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        msg, diff_sh = self._message(edge_index, edge_attr, node_feat, node_pos, node_sh)
        msg_agg, sh_agg = self._aggregate(edge_index, node_feat.size(0), msg, diff_sh)
        node_feat, node_sh = self._update(node_feat, node_sh, msg_agg, sh_agg)
        return node_feat, node_sh

    def _message(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        node_sh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]
        dist_sq = torch.norm(diff_pos, p=2, dim=-1, keepdim=True) ** 2
        sh_ip = self.sh_msg(edge_index, node_sh)

        msg_input = torch.cat(
            [node_feat[row], node_feat[col], edge_attr, dist_sq, sh_ip],
            dim=-1,
        )
        msg = self.mlp_msg(msg_input)

        diff_sh = node_sh[row] - node_sh[col]
        one = torch.ones([diff_sh.size(0), 1], device=diff_sh.device)
        diff_sh = self.sh_coff(diff_sh, one, self.mlp_sh(msg))
        return msg, diff_sh

    def _aggregate(
        self,
        edge_index: torch.Tensor,
        dim_size: int,
        msg: torch.Tensor,
        diff_sh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row, _ = edge_index
        msg_agg = scatter(src=msg, index=row, dim=0, dim_size=dim_size, reduce="mean")
        sh_agg = scatter(src=diff_sh, index=row, dim=0, dim_size=dim_size, reduce="mean")
        return msg_agg, sh_agg

    def _update(
        self,
        node_feat: torch.Tensor,
        node_sh: torch.Tensor,
        msg_agg: torch.Tensor,
        sh_agg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_sh = node_sh + sh_agg
        node_feat = torch.cat([node_feat, msg_agg], dim=-1)
        node_feat = self.mlp_node_feat(node_feat)
        return node_feat, node_sh


class SH_Msg(nn.Module):
    """Computes symmetric inner products between steerable features."""

    def __init__(self, sh_irreps: e3nn.o3.Irreps) -> None:
        super().__init__()
        self.sh_irreps = sh_irreps

    def forward(self, edge_index: torch.Tensor, node_sh: torch.Tensor) -> torch.Tensor:
        assert node_sh.size(1) == self.sh_irreps.dim
        row, col = edge_index
        temp = node_sh[row] * node_sh[col]

        idx = 0
        ip = torch.zeros(
            [node_sh[row].size(0), self.sh_irreps.lmax + 1],
            device=node_sh.device,
        )
        for mul, ir in self.sh_irreps:
            ip[:, ir.l] = torch.sum(temp[:, idx : idx + ir.dim], dim=-1)
            idx = idx + ir.dim

        return ip


class SH_INIT(nn.Module):
    """Initializes steerable node features from invariant inputs."""

    def __init__(
        self,
        edge_attr_dim: int,
        hidden_dim: int,
        max_ell: int,
        activation: nn.Module = nn.SiLU(),
    ) -> None:
        super().__init__()
        self.sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            self.sh_irreps,
            normalize=True,
            normalization="norm",
        )

        self.sh_coff = e3nn.o3.FullyConnectedTensorProduct(
            self.sh_irreps, "1x0e", self.sh_irreps, shared_weights=False
        )

        mlp_factory = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_sh = mlp_factory(
            input_dim=2 * hidden_dim + edge_attr_dim + 1,
            output_dim=self.sh_coff.weight_numel,
            last_act=True,
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]
        dist = torch.norm(diff_pos, dim=-1, keepdim=True)

        msg = torch.cat([dist, node_feat[row], node_feat[col], edge_attr], dim=-1)
        msg = self.mlp_sh(msg)
        diff_sh = self.spherical_harmonics(diff_pos).detach()
        one = torch.ones([diff_sh.size(0), 1], device=diff_sh.device).detach()
        diff_sh = self.sh_coff(diff_sh, one, msg)

        node_sh = scatter(diff_sh, index=row, dim=0, dim_size=node_feat.size(0), reduce="mean")
        return node_sh


class HEGNN(nn.Module):
    """HEGNN that predicts accelerations and integrates with symplectic schemes."""

    def __init__(
        self,
        num_layer: int,
        node_input_dim: int,
        edge_attr_dim: int,
        hidden_dim: int,
        max_ell: int,
        activation: nn.Module = nn.SiLU(),
        device: str = "cpu",
        force_activation: Optional[nn.Module] = None,
        distance_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.distance_eps = distance_eps

        self.embedding = nn.Linear(node_input_dim, hidden_dim)
        self.sh_init = SH_INIT(edge_attr_dim, hidden_dim, max_ell, activation)

        self.layers = nn.ModuleList(
            HEGNN_Layer(edge_attr_dim, hidden_dim, self.sh_init.sh_irreps, activation=activation)
            for _ in range(self.num_layer)
        )

        self.force_sh_msg = SH_Msg(self.sh_init.sh_irreps)
        self.force_mlp = BaseMLP(
            input_dim=2 * hidden_dim + edge_attr_dim + 1 + self.sh_init.sh_irreps.lmax + 1,
            hidden_dim=hidden_dim,
            output_dim=1,
            activation=activation,
        )
        self.force_activation = force_activation if force_activation is not None else nn.Identity()

        self.to(device)

    def forward(
        self,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        node_vel: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for predict_acceleration to keep old call-sites working."""
        del node_vel  # velocities are currently not consumed inside the encoder
        return self.predict_acceleration(node_feat, node_pos, edge_index, edge_attr, masses)

    def predict_acceleration(
        self,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_feat_latent, node_sh = self._encode(node_feat, node_pos, edge_index, edge_attr)
        return self._compute_acceleration(
            node_feat_latent,
            node_pos,
            node_sh,
            edge_index,
            edge_attr,
            masses=masses,
        )

    def symplectic_euler_step(
        self,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        node_vel: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single symplectic Euler step using the predicted accelerations."""
        accel = self.predict_acceleration(node_feat, node_pos, edge_index, edge_attr, masses)
        vel_next = node_vel + dt * accel
        pos_next = node_pos + dt * vel_next
        return pos_next, vel_next, accel

    def velocity_verlet_step(
        self,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        node_vel: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single Velocity Verlet step with two acceleration evaluations."""
        accel_start = self.predict_acceleration(node_feat, node_pos, edge_index, edge_attr, masses)
        vel_half = node_vel + 0.5 * dt * accel_start
        pos_next = node_pos + dt * vel_half

        accel_end = self.predict_acceleration(node_feat, pos_next, edge_index, edge_attr, masses)
        vel_next = vel_half + 0.5 * dt * accel_end
        return pos_next, vel_next, accel_end

    def _encode(
        self,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_feat_latent = self.embedding(node_feat)
        node_sh = self.sh_init(node_feat_latent, node_pos, edge_index, edge_attr)
        for layer in self.layers:
            node_feat_latent, node_sh = layer(
                node_feat_latent,
                node_pos,
                node_sh,
                edge_index,
                edge_attr,
            )
        return node_feat_latent, node_sh

    def _compute_acceleration(
        self,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        node_sh: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        masses: Optional[torch.Tensor],
    ) -> torch.Tensor:
        sym_feat, r_hat, row = self._build_symmetric_edge_features(
            node_feat, node_pos, node_sh, edge_index, edge_attr
        )
        force_mag = self.force_mlp(sym_feat)
        force_mag = self.force_activation(force_mag)

        force_vec = force_mag * r_hat
        accel = scatter(force_vec, row, dim=0, dim_size=node_pos.size(0), reduce="sum")

        if masses is not None:
            accel = accel / masses.view(-1, 1)

        return accel

    def _build_symmetric_edge_features(
        self,
        node_feat: torch.Tensor,
        node_pos: torch.Tensor,
        node_sh: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, col = edge_index
        rel_vec = node_pos[row] - node_pos[col]
        dist = torch.norm(rel_vec, dim=-1, keepdim=True)
        denom = dist.clamp_min(self.distance_eps)
        r_hat = rel_vec / denom
        sh_ip = self.force_sh_msg(edge_index, node_sh)

        h_sum = node_feat[row] + node_feat[col]
        h_mul = node_feat[row] * node_feat[col]

        inputs = [h_sum, h_mul]
        if edge_attr is not None:
            inputs.append(edge_attr)
        inputs.extend([dist, sh_ip])
        sym_feat = torch.cat(inputs, dim=-1)

        return sym_feat, r_hat, row
