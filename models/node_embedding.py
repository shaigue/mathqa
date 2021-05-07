from torch import nn, Tensor

from models.gated_gnn import GGNN

# TODO: test


class NodeEmbedding(nn.Module):
    def __init__(self, n_node_labels: int, dim: int, n_edge_types: int,
                 bias: bool = True, backward_edges: bool = True):
        super(NodeEmbedding, self).__init__()
        assert n_node_labels > 0
        self.n_node_labels = n_node_labels

        self.label_embedding = nn.Embedding(n_node_labels, dim)
        self.ggnn = GGNN(dim, n_edge_types, bias, backward_edges)

    def forward(self, adj_tensor: Tensor, node_labels: Tensor, n_prop_steps: int) -> Tensor:
        node_embedding = self.label_embedding(node_labels)
        return self.ggnn(adj_tensor, node_embedding, n_prop_steps)
