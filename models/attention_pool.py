# TODO: test
from math import sqrt

import torch
from torch import nn, Tensor


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super(AttentionPool, self).__init__()
        self.dim = dim
        self.query = nn.Parameter(
            torch.normal(0.0, 1.0 / sqrt(dim), size=(dim,))
        )

    def forward(self, keys: Tensor, values: Tensor) -> Tensor:
        scores = torch.matmul(keys, self.query)         # (N, D) @ D -> N
        weights = torch.softmax(scores, dim=0)          # N -> N
        weighted = weights.view(-1, 1) * values         # (N, 1) * (N, D) -> (N, D)
        return torch.sum(weighted, dim=0)               # (N, D) -> D
