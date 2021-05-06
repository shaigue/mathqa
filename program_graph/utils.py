import config
from math_qa import math_qa


def get_node_labels(max_inputs: int = config.MAX_INPUTS) -> list[str]:
    constants = [const_desc.name for const_desc in math_qa.get_constants_descriptors()]
    ops = [op_desc.name for op_desc in math_qa.get_operators_descriptors()]
    inputs = [f'n{i}' for i in range(max_inputs)]
    return constants + ops + inputs


def get_n_node_labels() -> int:
    return len(get_node_labels())


def get_max_n_args() -> int:
    return config.MAX_ARGS


def get_n_categories() -> int:
    return len(math_qa.get_categories())
