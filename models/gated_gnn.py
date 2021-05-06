"""This is the implementation of a simple Gated Graph Neural Network with typed edges and initial node features."""
import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch import Tensor


@dataclass
class GGNNInput:
    """
        adj_tensor: (edge_type, node, node) sparse tensor
        node_embedding: (node, embedding_dim) tensor
    """
    adj_tensor: Tensor
    node_embedding: Tensor
    n_prop_steps: int
    n_nodes: int = field(init=False)
    n_edge_types: int = field(init=False)
    device: torch.device = field(init=False)
    embedding_dim: int = field(init=False)

    def __post_init__(self):
        assert self.adj_tensor.ndim == 3
        assert self.node_embedding.ndim == 2
        assert self.adj_tensor.shape[1] == self.adj_tensor.shape[2] == self.node_embedding.shape[0]
        assert self.adj_tensor.device == self.node_embedding.device
        assert self.n_prop_steps >= 0
        self.n_nodes = self.adj_tensor.shape[1]
        self.n_edge_types = self.adj_tensor.shape[0]
        self.device = self.adj_tensor.device
        self.embedding_dim = self.node_embedding.shape[1]

    def to(self, device: torch.device):
        return GGNNInput(self.adj_tensor.to(device), self.node_embedding.to(device), self.n_prop_steps)


@dataclass
class GGNNOutput:
    """
        node_embedding (prop_step, node, embedding_dim) Tensor
    """
    node_embedding: Tensor
    n_nodes: int = field(init=False)
    device: torch.device = field(init=False)
    embedding_dim: int = field(init=False)

    def __post_init__(self):
        assert self.node_embedding.ndim == 2
        self.n_nodes = self.node_embedding.shape[0]
        self.embedding_dim = self.node_embedding.shape[1]
        self.device = self.node_embedding.device

    def to(self, device: torch.device):
        return GGNNOutput(self.node_embedding.to(device))


class GGNN(nn.Module):
    def __init__(self, hidden_dim: int, n_edge_types: int, bias: bool = True):
        assert hidden_dim > 0
        assert n_edge_types > 0

        super(GGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_edge_types = n_edge_types
        self.bias = bias

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        std = 1 / math.sqrt(hidden_dim)
        self.edge_tensor = nn.Parameter(
            torch.normal(mean=0, std=std, size=(n_edge_types, hidden_dim + bias, hidden_dim))
        )

    def _check_inputs(self, inputs: GGNNInput) -> None:
        assert inputs.n_edge_types == self.n_edge_types
        assert self.hidden_dim == inputs.embedding_dim

    def _check_outputs(self, inputs: GGNNInput, outputs: GGNNOutput) -> None:
        assert outputs.n_nodes == inputs.n_nodes
        assert outputs.embedding_dim == inputs.embedding_dim == self.hidden_dim

    def forward(self, inputs: GGNNInput) -> GGNNOutput:
        self._check_inputs(inputs)

        adj_transpose = inputs.adj_tensor.transpose(1, 2)
        node_embedding = inputs.node_embedding

        for i in range(inputs.n_prop_steps):
            # pad with ones to add bias with matmul
            if self.bias:
                one_pad = torch.ones(inputs.n_nodes, 1, device=inputs.device)
                activation = torch.cat([node_embedding, one_pad], dim=1)
            else:
                activation = node_embedding
            assert activation.shape == (inputs.n_nodes, self.hidden_dim + self.bias)

            activation = torch.matmul(adj_transpose, activation)
            assert activation.shape == (self.n_edge_types, inputs.n_nodes, self.hidden_dim + self.bias)

            activation = torch.matmul(activation, self.edge_tensor)
            assert activation.shape == (self.n_edge_types, inputs.n_nodes, self.hidden_dim)

            activation = torch.sum(activation, dim=0)
            assert activation.shape == (inputs.n_nodes, self.hidden_dim)

            node_embedding = self.gru(activation, node_embedding)

        outputs = GGNNOutput(node_embedding)
        self._check_outputs(inputs, outputs)
        return outputs
