from torch import LongTensor, FloatTensor
from torch import nn

from abc import abstractmethod

def get_device(module: nn.Module):
    parameter_iterator = module.parameters()
    first_parameter = next(parameter_iterator)
    return first_parameter.device


def get_gru(input_size: int, hidden_dim: int, num_layers=1, bias=True,
            batch_first=True, dropout=0.0, bidirectional=False) -> nn.GRU:
    return nn.GRU(
        input_size=input_size,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional
    )


# TODO - make a unified interface
class Seq2SeqInterface(nn.Module):
    def __init__(self):
        super(Seq2SeqInterface, self).__init__()

    @abstractmethod
    def forward(self, inputs: LongTensor, outputs: LongTensor,
                inputs_len: list[int] = None, output_lens: list[int] = None) -> FloatTensor:
        pass

    @abstractmethod
    def generate(self, inputs: LongTensor, inputs_len: list[int] = None,):
        pass