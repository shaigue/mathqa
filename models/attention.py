from abc import abstractmethod
from dataclasses import dataclass

from torch import nn

@dataclass
class AttentionInput:
    pass

@dataclass
class AttentionOutput:
    pass


class AttentionInterface(nn.Module):
    def __init__(self):
        super(AttentionInterface, self).__init__()

    @abstractmethod
    def forward(self, inputs: AttentionInput) -> AttentionOutput:
        pass

    def _check_inputs(self, inputs: AttentionInput):
        pass

    def _check_outputs(self, inputs: AttentionInput, outputs: AttentionOutput):
        pass


class DotAttention(AttentionInterface):
    """k^T q"""
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, inputs: AttentionInput) -> AttentionOutput:
        pass


class ScaledDotAttention(AttentionInterface):
    """k^T q / sqrt(d)"""
    def __init__(self):
        super(ScaledDotAttention, self).__init__()

    def forward(self, inputs: AttentionInput) -> AttentionOutput:
        pass


class AdditiveAttention(AttentionInterface):
    """v^T tanh(Aq + a + Bk + b)"""
    def __init__(self):
        super(AdditiveAttention, self).__init__()

    def forward(self, inputs: AttentionInput) -> AttentionOutput:
        pass


class GeneralAttention(AttentionInterface):
    """k^T W q"""
    def __init__(self):
        super(GeneralAttention, self).__init__()

    def forward(self, inputs: AttentionInput) -> AttentionOutput:
        pass
