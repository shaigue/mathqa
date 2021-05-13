from program_processing.parsed_to_nx import parsed_to_nx
from program_processing.parse_linear_formula import parse_linear_formula
from program_processing.nx_to_tensor import nx_to_tensor
from program_processing.common import GraphTensor


def linear_formula_to_tensor(linear_formula: str, n_inputs: int, n_edge_types: int,
                             node_labels_to_idx: dict[str, int]) -> GraphTensor:
    parsed = parse_linear_formula(linear_formula)
    graph = parsed_to_nx(parsed, n_inputs)
    return nx_to_tensor(graph, node_labels_to_idx, n_edge_types)

