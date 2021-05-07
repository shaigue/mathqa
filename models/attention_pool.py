# TODO: implement
# TODO: test
# TODO: create a shared dataclass base class for all inputs and outputs

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn, Tensor


@dataclass()
class AttentionPoolInput:
    keys: Tensor
    values: Tensor


@dataclass()
class AttentionPoolOutput:
    aggregate: Tensor


class AttentionPool(nn.Module):
    def __init__(self, key_dim: int):
        super(AttentionPool, self).__init__()
        self.key_dim = key_dim
        self.query = nn.Parameter(
            torch.normal(0.0, 1.0 / sqrt(key_dim), size=(key_dim, ))
        )

    def forward(self, inputs: AttentionPoolInput) -> AttentionPoolOutput:
        scores = torch.matmul(inputs.keys, self.query)  # (N, D) @ D -> N
        weights = torch.softmax(scores, dim=0)          # N -> N
        weighted = weights.view(-1, 1) * inputs.values  # (N, 1) * (N, D) -> (N, D)
        aggregate = torch.sum(weighted, dim=0)          # (N, D) -> D
        return AttentionPoolOutput(aggregate)
