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
    # TODO: implement using the GraphNodeEmbedding module
    def __init__(self):
        super().__init__()

    def forward(self):
        # run GraphNodeEmbedding on the graph to get node embedding
        # separate the node to be connected and all the other nodes
        # calculate and return the attention scores
        pass







