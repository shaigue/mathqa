import torch
from torch import nn, Tensor
from models.simple_ggnn import GatedGraphNN


class GraphClassifier(nn.Module):
    def __init__(self, n_node_labels: int, n_edge_types: int, hidden_dim: int, n_classes: int,
                 gnn: bool, gnn_bias: bool, gnn_n_iter: int):
        super().__init__()
        self.n_node_labels = n_node_labels
        self.n_edge_types = n_edge_types
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.node_label_embedding = nn.Embedding(num_embeddings=n_node_labels, embedding_dim=hidden_dim)
        self.with_gnn = gnn
        if gnn:
            self.gnn = GatedGraphNN(hidden_dim=hidden_dim, n_edge_types=n_edge_types, bias=gnn_bias,
                                    n_iter=gnn_n_iter)

        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward(self, adj_tensor: Tensor, node_labels: Tensor):
        # N - n_nodes, E - n_edge_types, H - hidden_dim, C - n_classes
        n_edge_types, n_nodes, n_nodes0 = adj_tensor.shape
        assert n_nodes == n_nodes0
        assert n_edge_types == self.n_edge_types
        assert node_labels.shape == (n_nodes, )

        node_embedding = self.node_label_embedding(node_labels)         # (N, H)
        if self.with_gnn:
            node_embedding = self.gnn(adj_tensor, node_embedding)       # (N, H)

        mean_embedding = torch.mean(node_embedding, dim=0)              # (H,)
        mean_embedding = torch.unsqueeze(mean_embedding, 0)             # (1, H)
        class_scores = self.linear(mean_embedding)                      # (1, C)
        return class_scores.squeeze()                                   # (C,)
