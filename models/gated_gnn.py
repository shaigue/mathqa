"""This is the implementation of a simple Gated Graph Neural Network with typed edges and initial node features."""
import math

import torch
from torch import nn
from torch import Tensor
from models.common import one_pad


class GGNN(nn.Module):
    def __init__(self, dim: int, n_edge_types: int, bias: bool = True,
                 backward_edges: bool = True):
        assert dim > 0
        assert n_edge_types > 0

        super(GGNN, self).__init__()
        self.dim = dim
        self.n_edge_types = n_edge_types
        self.n_virtual_edge_types = 2 * n_edge_types if backward_edges else n_edge_types
        self.bias = bias
        self.backward_edges = backward_edges

        self.gru = nn.GRUCell(dim, dim)
        std = 1 / math.sqrt(dim)
        edge_tensor_size = (self.n_virtual_edge_types, dim + bias, dim)
        self.edge_tensor = nn.Parameter(
            torch.normal(mean=0, std=std, size=edge_tensor_size)
        )

    def forward(self, adj_tensor: Tensor, node_embedding: Tensor,
                n_prop_steps: int) -> Tensor:
        assert adj_tensor.ndim == 3
        assert adj_tensor.shape[0] == self.n_edge_types
        assert adj_tensor.shape[1] == adj_tensor.shape[2] == node_embedding.shape[0]
        assert node_embedding.ndim == 2
        assert node_embedding.shape[1] == self.dim
        assert n_prop_steps >= 0

        n_nodes = node_embedding.shape[0]

        if self.backward_edges:
            adj_tensor = torch.cat([adj_tensor, adj_tensor.transpose(1, 2)])
        else:
            adj_tensor = adj_tensor.transpose(1, 2)

        for i in range(n_prop_steps):
            activation = node_embedding
            # pad with ones to add bias with matmul
            if self.bias:
                activation = one_pad(activation, 1)

            assert activation.shape == (n_nodes, self.dim + self.bias)

            activation = torch.matmul(adj_tensor, activation)
            assert activation.shape == (self.n_virtual_edge_types, n_nodes,
                                        self.dim + self.bias)

            activation = torch.matmul(activation, self.edge_tensor)
            assert activation.shape == (self.n_virtual_edge_types, n_nodes,
                                        self.dim)

            activation = torch.sum(activation, dim=0)
            assert activation.shape == (n_nodes, self.dim)

            node_embedding = self.gru(activation, node_embedding)

        return node_embedding
