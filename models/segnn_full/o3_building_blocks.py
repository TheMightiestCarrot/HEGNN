from math import sqrt

import torch
import torch.nn as nn
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, spherical_harmonics
from torch_scatter import scatter


class O3TensorProduct(nn.Module):
    """A bilinear layer, computing CG tensor product and normalising them."""

    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, tp_rescale=True) -> None:
        super().__init__()

        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_out = Irreps(irreps_out)
        if irreps_in2 is None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = Irreps(irreps_in2)
        self.tp_rescale = tp_rescale

        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            shared_weights=True,
            normalization="component",
        )

        self.irreps_out_orders = [
            int(irrep_str[-2]) for irrep_str in str(self.irreps_out).split("+")
        ]
        self.irreps_out_dims = [
            int(irrep_str.split("x")[0]) for irrep_str in str(self.irreps_out).split("+")
        ]
        self.irreps_out_slices = self.irreps_out.slices()

        self.biases = []
        self.biases_slices = []
        self.biases_slice_idx = []
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_slice = self.irreps_out.slices()[slice_idx]
                out_bias = torch.zeros(
                    self.irreps_out_dims[slice_idx], dtype=self.tp.weight.dtype
                )
                self.biases += [out_bias]
                self.biases_slices += [out_slice]
                self.biases_slice_idx += [slice_idx]

        self.slices_sqrt_k = {}

        self.tensor_product_init()
        self.vectorise()

    def tensor_product_init(self) -> None:
        with torch.no_grad():
            slices_fan_in = {}
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                slice_idx = instr[2]
                mul_1, mul_2, _ = weight.shape
                fan_in = mul_1 * mul_2
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in
                    else fan_in
                )

            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                slice_idx = instr[2]
                if self.tp_rescale:
                    sqrt_k = 1 / sqrt(slices_fan_in[slice_idx])
                else:
                    sqrt_k = 1.0
                weight.data.uniform_(-sqrt_k, sqrt_k)
                self.slices_sqrt_k[slice_idx] = (
                    self.irreps_out_slices[slice_idx],
                    sqrt_k,
                )

            for out_slice_idx, out_slice, out_bias in zip(
                self.biases_slice_idx, self.biases_slices, self.biases
            ):
                sqrt_k = 1 / sqrt(slices_fan_in[out_slice_idx])
                out_bias.uniform_(-sqrt_k, sqrt_k)

    def vectorise(self):
        if len(self.biases) > 0:
            with torch.no_grad():
                self.biases = torch.cat(self.biases, dim=0)
            self.biases = nn.Parameter(self.biases)

            bias_idx = torch.LongTensor()
            for slice_idx in range(len(self.irreps_out_orders)):
                if self.irreps_out_orders[slice_idx] == 0:
                    out_slice = self.irreps_out.slices()[slice_idx]
                    bias_idx = torch.cat(
                        (bias_idx, torch.arange(out_slice.start, out_slice.stop)), dim=0
                    )

            self.register_buffer("bias_idx", bias_idx, persistent=False)
        else:
            self.biases = None

        sqrt_k_correction = torch.zeros(self.irreps_out.dim)
        for instr in self.tp.instructions:
            slice_idx = instr[2]
            slice_, sqrt_k = self.slices_sqrt_k[slice_idx]
            sqrt_k_correction[slice_] = sqrt_k

        self.register_buffer("sqrt_k_correction", sqrt_k_correction, persistent=False)

    def forward_tp_rescale_bias(self, data_in1, data_in2=None) -> torch.Tensor:
        if data_in2 is None:
            data_in2 = torch.ones_like(data_in1[:, 0:1])

        data_out = self.tp(data_in1, data_in2)

        if self.tp_rescale:
            data_out /= self.sqrt_k_correction

        if self.biases is not None:
            data_out[:, self.bias_idx] += self.biases
        return data_out

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        return self.forward_tp_rescale_bias(data_in1, data_in2)


class O3TensorProductSwishGate(O3TensorProduct):
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None) -> None:
        irreps_out = Irreps(irreps_out)
        irreps_g_scalars = Irreps(str(irreps_out[0]))
        irreps_g_gate = Irreps(
            "{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps)
        )
        irreps_g_gated = Irreps(str(irreps_out[1:]))
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()

        super().__init__(irreps_in1, irreps_g, irreps_in2)
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(
                irreps_g_scalars,
                [nn.SiLU()],
                irreps_g_gate,
                [torch.sigmoid],
                irreps_g_gated,
            )
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        data_out = self.gate(data_out)
        return data_out


class O3Transform:
    """Construct SEGNN inputs from raw N-body state (pos, vel, mass, force)."""

    def __init__(self, lmax_attr, use_force_input=False):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        self.use_force_input = use_force_input

    def __call__(self, graph):
        pos = graph.pos
        vel = graph.vel
        node_attr = getattr(graph, "node_attr", None)
        if node_attr is None:
            node_attr = getattr(graph, "mass", None)
        if node_attr is None:
            node_attr = getattr(graph, "charge", None)
        if node_attr is None:
            raise ValueError("SEGNNFull requires node_attr (charges) on the graph.")
        force = getattr(graph, "force", None)
        if force is None:
            force = torch.zeros_like(pos)
        num_nodes = pos.shape[0]

        prod_attr = node_attr[graph.edge_index[0]] * node_attr[graph.edge_index[1]]
        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.linalg.norm(rel_pos, dim=1, keepdims=True)

        graph.edge_attr = spherical_harmonics(
            self.attr_irreps, rel_pos, normalize=True, normalization="integral"
        )

        vel_embedding = spherical_harmonics(
            self.attr_irreps, vel, normalize=True, normalization="integral"
        )
        graph.node_attr = (
            scatter(
                graph.edge_attr,
                graph.edge_index[1],
                dim=0,
                reduce="mean",
                dim_size=num_nodes,
            )
            + vel_embedding
        )

        if self.use_force_input:
            force_embedding = spherical_harmonics(
                self.attr_irreps, force, normalize=True, normalization="integral"
            )
            graph.node_attr += force_embedding

        vel_abs = torch.linalg.norm(vel, dim=1, keepdims=True)
        mean_pos = scatter(pos, graph.batch, dim=0, reduce="mean")[graph.batch]

        graph.x = torch.cat((pos - mean_pos, vel, vel_abs), dim=1)
        graph.additional_message_features = torch.cat((edge_dist, prod_attr), dim=-1)
        return graph
