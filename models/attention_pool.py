from math import sqrt

import torch
from torch import nn, Tensor


class AttentionPool(nn.Module):
    def __init__(self, key_dim: int):
        assert key_dim > 0

        super(AttentionPool, self).__init__()
        self.key_dim = key_dim
        self.query = nn.Parameter(
            torch.normal(0.0, 1.0 / sqrt(key_dim), size=(key_dim,))
        )

    def forward(self, keys: Tensor, values: Tensor) -> Tensor:
        assert keys.ndim == 2
        assert values.ndim == 2
        assert keys.shape[0] == values.shape[0]
        assert keys.shape[1] == self.key_dim

        scores = torch.matmul(keys, self.query)         # (N, D) @ D -> N
        weights = torch.softmax(scores, dim=0)          # N -> N
        return torch.matmul(weights, values)            # N @ (N, D) -> D
