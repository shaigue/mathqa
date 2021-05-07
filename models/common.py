import torch
from torch import nn, Tensor


def get_module_device(module: nn.Module):
    parameter_iterator = module.parameters()
    first_parameter = next(parameter_iterator)
    return first_parameter.device


def one_pad(t: Tensor, dim: int) -> Tensor:
    """Adds an extra one line to a given dimension"""
    size = list(t.shape)
    size[dim] = 1
    pad = torch.ones(*size, device=t.device, dtype=t.dtype)
    return torch.cat([t, pad], dim)
