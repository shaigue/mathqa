# TODO: test
from math import sqrt

import torch
from torch import nn, Tensor


class GeneralAttention(nn.Module):
    def __init__(self, key_dim: int, query_dim: int):
        super(GeneralAttention, self).__init__()
        self.key_dim = key_dim
        self.query_dim = query_dim

        max_dim = max(query_dim, key_dim)
        self.weights = nn.Parameter(
            torch.normal(0, 1 / sqrt(max_dim), size=(key_dim, query_dim))
        )

    def forward(self, keys: Tensor, queries: Tensor) -> Tensor:
        scores = torch.matmul(self.weights, queries.T)
        return torch.matmul(keys, scores)
