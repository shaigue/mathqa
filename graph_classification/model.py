import torch
from torch import nn, Tensor

from models.simple_ggnn import GatedGraphNN


class GraphNodeEmbedding(nn.Module):
    def __init__(self, n_node_labels: int, hidden_dim: int, with_gnn: bool,
                 n_edge_types: int = None, bias: bool = False, aggregation_type: str = 'avg',
                 n_iter: int = 1):
        super(GraphNodeEmbedding, self).__init__()
        self.node_label_embedding = nn.Embedding(num_embeddings=n_node_labels,
                                                 embedding_dim=hidden_dim,)
        self.with_gnn = with_gnn
        if self.with_gnn:
            self.gnn = GatedGraphNN(hidden_dim=hidden_dim,
                                    n_edge_types=n_edge_types,
                                    bias=bias,
                                    aggregation_type=aggregation_type,
                                    n_iter=n_iter)

    def forward(self, adj_tensor: Tensor, node_labels: Tensor) -> Tensor:
        """Receives a labeled graph, and outputs node embeddings"""
        n_edge_types, n_nodes, n_nodes0 = adj_tensor.shape
        assert n_nodes == n_nodes0
        assert node_labels.shape == (n_nodes,)

        node_embedding = self.node_label_embedding(node_labels)
        if self.with_gnn:
            return self.gnn(adj_tensor, node_embedding)
        else:
            return node_embedding


class GraphClassifier(nn.Module):
    def __init__(self, n_node_labels: int, n_edge_types: int, hidden_dim: int, n_classes: int,
                 gnn: bool, gnn_bias: bool, gnn_n_iter: int):
        super().__init__()
        self.n_node_labels = n_node_labels
        self.n_edge_types = n_edge_types
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.node_embedding_layer = GraphNodeEmbedding(n_node_labels, hidden_dim, gnn, n_edge_types, gnn_bias,
                                                       n_iter=gnn_n_iter)

        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward(self, adj_tensor: Tensor, node_labels: Tensor):
        # N - n_nodes, E - n_edge_types, H - hidden_dim, C - n_classes
        node_embedding = self.node_embedding_layer(adj_tensor, node_labels)  # (N, H)

        mean_embedding = torch.mean(node_embedding, dim=0)              # (H,)
        mean_embedding = torch.unsqueeze(mean_embedding, 0)             # (1, H)
        class_scores = self.linear(mean_embedding)                      # (1, C)
        return class_scores.squeeze()                                   # (C,)


