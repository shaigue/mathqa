"""This is the model for selecting the edges, given:
* node embeddings
* initial context
* embedding of node to connect
"""
import math

import torch
from torch import nn, Tensor


def dot_attention_score(keys: Tensor, queries: Tensor, scaled: bool = False) -> Tensor:
    assert keys.ndim == queries.ndim == 2
    assert keys.shape[1] == queries.shape[1]

    attention_scores = torch.matmul(queries, keys.T)

    if scaled:
        d = keys.shape[1]
        return attention_scores / math.sqrt(d)
    else:
        return attention_scores


class AdditiveAttention(nn.Module):
    def __init__(self, key_dim: int, query_dim: int, attention_dim: int):
        super().__init__()
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.attention_dim = attention_dim

        self.key_linear = nn.Linear(key_dim, attention_dim)
        self.query_linear = nn.Linear(query_dim, attention_dim)
        self.attention_linear = nn.Linear(attention_dim, 1)

    def forward(self, keys: Tensor, query: Tensor) -> Tensor:
        assert keys.ndim == 2
        assert query.ndim == 1
        assert keys.shape[1] == self.key_dim
        assert query.shape[0] == self.query_dim

        keys_transform = self.key_linear(keys)
        query_transform = self.query_linear(query.unsqueeze(0))
        additive = keys_transform + query_transform
        additive = torch.tanh(additive)
        return self.attention_linear(additive)


class EdgeSelectionModel(nn.Module):
    def __init__(self, n_node_labels: int, hidden_dim: int, n_edge_type: int = None,
                 with_gnn: bool = True, bias: bool = True, aggregation_type: str = 'avg',
                 n_iter: int = 1, attention_dim: int = None):
        super().__init__()
        if attention_dim is None:
            attention_dim = hidden_dim

        self.node_embedding_layer = GraphNodeEmbedding(
            n_node_labels=n_node_labels,
            hidden_dim=hidden_dim, with_gnn=with_gnn,
            n_edge_types=n_edge_type,
            bias=bias, aggregation_type=aggregation_type,
            n_iter=n_iter)
        self.attention_layer = AdditiveAttention(key_dim=hidden_dim,
                                                 query_dim=hidden_dim,
                                                 attention_dim=attention_dim)

    def forward(self, adj_tensor: Tensor, node_labels: Tensor, dst_node: int) -> Tensor:
        """
        :return: logits for the probability for attaching each node to the target node
        """
        # run NodeEmbedding on the graph to get node embedding
        node_embedding = self.node_embedding_layer(adj_tensor, node_labels)
        # separate the node to be connected and all the other nodes
        n_nodes = node_embedding.shape[0]
        query = node_embedding[dst_node]
        other_indices = torch.tensor([i for i in range(node_embedding.shape[0]) if i != dst_node])
        keys = node_embedding[other_indices]
        # calculate and return the attention logits
        logits = self.attention_layer(keys, query)
        assert logits.shape == (n_nodes - 1, 1)
        return logits.squeeze()

