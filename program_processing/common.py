from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from torch import Tensor

import config
from math_qa import math_qa


class ArgType(Enum):
    const = auto()
    op = auto()
    input = auto()


def get_node_labels_by_type(max_inputs: int = config.MAX_INPUTS) -> \
        dict[ArgType, list[str]]:
    return {
        ArgType.const: [const_desc.name for const_desc in math_qa.get_constants_descriptors()],
        ArgType.op: [op_desc.name for op_desc in math_qa.get_operators_descriptors()],
        ArgType.input: [f'n{i}' for i in range(max_inputs)]
    }


def get_node_label_list(max_inputs: int = config.MAX_INPUTS) -> list[str]:
    node_label_dict = get_node_labels_by_type(max_inputs)
    concatenation = []
    for sublist in node_label_dict.values():
        concatenation += sublist
    return concatenation


def create_item_to_index_dict(l: list[Any]) -> dict[Any, int]:
    return {item: idx for idx, item in enumerate(l)}


def get_n_node_labels() -> int:
    return len(get_node_label_list())


def get_max_n_args() -> int:
    return config.MAX_ARGS


def get_n_categories() -> int:
    return len(math_qa.get_categories())


def get_node_label_to_idx() -> dict[str, int]:
    return create_item_to_index_dict(get_node_label_list())


def get_node_label_to_edge_types() -> dict[int, list[int]]:
    """
    :returns: a mapping from node label index to the edge types that need to connect to it.
    """
    node_label_list = get_node_label_list()
    op_name_to_n_args = {
        op_desc.name: op_desc.n_args
        for op_desc in math_qa.get_operators_descriptors()
    }
    node_label_to_edge_types = {}
    for i, node_label in enumerate(node_label_list):
        if node_label in op_name_to_n_args:
            node_label_to_edge_types[i] = list(range(op_name_to_n_args[node_label]))
        else:
            node_label_to_edge_types[i] = []

    return node_label_to_edge_types


def get_op_label_indices() -> list[int]:
    node_labels_by_type = get_node_labels_by_type()
    node_label_to_idx = get_node_label_to_idx()
    return [node_label_to_idx[op_label] for op_label in node_labels_by_type[ArgType.op]]


def get_partial_graph_tensor(adj_tensor: Tensor, node_labels: Tensor, start_node_i: int) -> tuple[Tensor, Tensor]:
    return adj_tensor[:, :start_node_i, :start_node_i].clone(), node_labels[:start_node_i].clone()


@dataclass
class GraphTensor:
    adj_tensor: Tensor
    node_labels: Tensor
