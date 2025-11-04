import numpy as np
import torch
import torch.nn as nn
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from typing import Optional, Union

from models.segnn.balanced_irreps import WeightBalancedIrreps
from models.segnn.instance_norm import InstanceNorm

from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate, O3Transform


class SEGNNFull(nn.Module):
    """Steerable E(3) equivariant message passing network with rich O(3) features."""

    def __init__(
        self,
        input_irreps: Optional[Union[Irreps, str]] = None,
        hidden_features: int = 64,
        lmax_h: int = 1,
        lmax_attr: int = 1,
        num_layers: int = 4,
        output_irreps: Optional[Union[Irreps, str]] = None,
        norm: Optional[str] = "batch",
        pool: str = "avg",
        task: str = "node",
        additional_message_irreps: Optional[Union[Irreps, str]] = None,
        use_force_input: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if input_irreps is None:
            input_irreps = Irreps("2x1o + 1x0e")
        else:
            input_irreps = Irreps(input_irreps)
        if output_irreps is None:
            output_irreps = Irreps("2x1o")
        else:
            output_irreps = Irreps(output_irreps)
        if additional_message_irreps is None:
            additional_message_irreps = Irreps("2x0e")
        else:
            additional_message_irreps = Irreps(additional_message_irreps)

        self.hidden_features = hidden_features
        self.lmax_h = lmax_h
        self.lmax_attr = lmax_attr
        self.num_layers = num_layers
        self.task = task
        self.norm = norm
        self.pool = pool
        self.additional_message_irreps = additional_message_irreps
        self.transform = O3Transform(lmax_attr, use_force_input=use_force_input)

        self.node_attr_irreps = Irreps.spherical_harmonics(lmax_attr)

        hidden_irreps = WeightBalancedIrreps(
            Irreps(f"{hidden_features}x0e"),
            self.node_attr_irreps,
            sh=True,
            lmax=lmax_h,
        )
        edge_attr_irreps = Irreps.spherical_harmonics(lmax_attr)

        self.hidden_irreps = hidden_irreps
        self.edge_attr_irreps = edge_attr_irreps
        self.output_irreps = output_irreps

        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, self.node_attr_irreps
        )

        layers = []
        for _ in range(num_layers):
            layers.append(
                SEGNNFullLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    self.node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                )
            )
        self.layers = nn.ModuleList(layers)

        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, self.node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, self.node_attr_irreps
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
            self.init_pooler(pool)
        elif task == "node":
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, self.node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, self.node_attr_irreps
            )
        else:
            raise ValueError(f"Unsupported task '{task}' for SEGNNFull.")

        if device is not None:
            self.to(device)

    def init_pooler(self, pool):
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool
        else:
            raise ValueError(f"Unknown pool type '{pool}'.")

    def catch_isolated_nodes(self, graph):
        has_fn = getattr(graph, "has_isolated_nodes", None)
        if callable(has_fn):
            if graph.has_isolated_nodes() and graph.edge_index.max().item() + 1 != graph.num_nodes:
                nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
                add_attr = graph.node_attr.new_tensor(
                    np.zeros((nr_add_attr, graph.node_attr.shape[-1]))
                )
                graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        if graph.node_attr.shape[1] > 0:
            graph.node_attr[:, 0] = 1.0

    def build_graph(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        force: Optional[torch.Tensor] = None,
    ) -> Data:
        if node_attr.dim() == 1:
            node_attr = node_attr.unsqueeze(-1)
        if force is None:
            force = torch.zeros_like(pos)
        graph = Data(
            pos=pos,
            vel=vel,
            node_attr=node_attr,
            force=force,
            edge_index=edge_index,
            batch=batch,
        )
        graph = self.transform(graph)
        return graph

    def forward(
        self,
        graph: Optional[Data] = None,
        *,
        pos: Optional[torch.Tensor] = None,
        vel: Optional[torch.Tensor] = None,
        node_attr: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        force: Optional[torch.Tensor] = None,
    ):
        if graph is None:
            required = (pos, vel, node_attr, edge_index, batch)
            if any(t is None for t in required):
                raise ValueError(
                    "Either provide a graph or all of pos, vel, node_attr, edge_index, batch."
                )
            graph = self.build_graph(pos, vel, node_attr, edge_index, batch, force=force)
        else:
            if not hasattr(graph, "x"):
                graph = self.transform(graph)

        x, edge_index, edge_attr, node_attr, batch_vec = (
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
        )
        additional_message_features = getattr(
            graph, "additional_message_features", None
        )

        self.catch_isolated_nodes(graph)

        x = self.embedding_layer(x, node_attr)

        for layer in self.layers:
            x = layer(
                x,
                edge_index,
                edge_attr,
                node_attr,
                batch_vec,
                additional_message_features,
            )

        x = self.pre_pool1(x, node_attr)
        x = self.pre_pool2(x, node_attr)

        if self.task == "graph":
            x = self.pooler(x, batch_vec)
            x = self.post_pool1(x)
            x = self.post_pool2(x)
        return x


class SEGNNFullLayer(MessagePassing):
    """E(3) equivariant message passing layer with gated tensor products."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = Irreps(hidden_irreps)
        additional_message_irreps = (
            Irreps(additional_message_irreps)
            if additional_message_irreps is not None
            else Irreps()
        )

        input_irreps = Irreps(input_irreps)
        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + self.hidden_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )

        self.setup_normalisation(norm)

    def setup_normalisation(self, norm):
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        if additional_message_features is None:
            input_features = torch.cat((x_i, x_j), dim=-1)
        else:
            input_features = torch.cat(
                (x_i, x_j, additional_message_features), dim=-1
            )
        message = self.message_layer_1(input_features, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        update_input = torch.cat((x, message), dim=-1)
        update = self.update_layer_1(update_input, node_attr)
        update = self.update_layer_2(update, node_attr)
        x = x + update
        return x
