"""This is the implementation of a simple Gated Graph Neural Network with typed edges and initial node features."""
import itertools
import math

import torch
from torch import nn
from torch import Tensor


class GatedGraphNN(nn.Module):
    def __init__(self, hidden_dim: int, n_edge_types: int, bias: bool = False,
                 aggregation_type: str = 'avg', n_iter: int = 1):
        assert aggregation_type in ['avg', 'sum']
        assert hidden_dim > 0
        assert n_edge_types > 0

        super(GatedGraphNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_edge_types = n_edge_types
        self.bias = bias
        self.aggregation_type = aggregation_type
        self.n_iter = n_iter

        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)
        std = 1 / math.sqrt(hidden_dim)
        self.edge_tensor = nn.Parameter(
            torch.normal(mean=0, std=std, size=(n_edge_types * 2, hidden_dim + bias, hidden_dim))
        )

    def forward(self, adj_tensor: Tensor, node_embedding: Tensor) -> Tensor:
        # initialize the hidden vector
        n_edge_types, n_nodes, n_nodes0 = adj_tensor.shape
        assert n_edge_types == self.n_edge_types
        assert n_nodes == n_nodes0
        assert node_embedding.shape == (n_nodes, self.hidden_dim)

        # calculate the double adj matrix, with reversed edges
        adj_tensor = torch.cat([adj_tensor, adj_tensor.transpose(1, 2)])

        for i in range(self.n_iter):
            if self.bias:
                one_pad = torch.ones(n_nodes, 1, device=node_embedding.device)
                activation = torch.cat([node_embedding, one_pad], dim=1)       # N, H + b
            else:
                activation = node_embedding         # N, H + b

            activation = torch.matmul(adj_tensor, activation)           # (E, N, N) x (N, H + b) -> (E, N, H + b)
            activation = torch.matmul(activation, self.edge_tensor)     # (E, N, H + b) x (E, H + b, H) -> (E, N, H)
            activation = torch.sum(activation, dim=0)                   # (N, H)
            assert activation.shape == node_embedding.shape

            if self.aggregation_type == 'avg':
                n_edges_per_node = torch.sum(adj_tensor, dim=(0, 1))    # (E, N, N) -> (N, )
                n_edges_per_node[n_edges_per_node == 0] = 1             # avoid dividing by 0
                activation = activation / n_edges_per_node.view(n_nodes, 1)

            node_embedding, _ = self.gru(activation.view(1, n_nodes, self.hidden_dim),
                                         node_embedding.view(1, n_nodes, self.hidden_dim))

            node_embedding = node_embedding.view(n_nodes, self.hidden_dim)

        return node_embedding


def test_gnn_basic():
    hidden_dim = 64
    n_nodes = 27
    edge_types_list = [1, 5]
    bias_list = [True, False]
    agg_types_list = ['sum', 'avg']
    iter_list = [1, 5]
    dev_list = ['cpu', 'cuda']
    i = 0
    for et, b, ag, it, d in itertools.product(edge_types_list, bias_list, agg_types_list, iter_list, dev_list):
        print(i)
        i += 1
        gnn = GatedGraphNN(
            hidden_dim=hidden_dim,
            n_edge_types=et,
            bias=b,
            aggregation_type=ag,
            n_iter=it,
        ).to(d)
        adj_tensor = torch.ones(et, n_nodes, n_nodes, device=d)
        node_embedding = torch.zeros(n_nodes, hidden_dim, device=d)
        out = gnn(adj_tensor, node_embedding)
        assert out.shape == (n_nodes, hidden_dim)


if __name__ == "__main__":
    test_gnn_basic()


