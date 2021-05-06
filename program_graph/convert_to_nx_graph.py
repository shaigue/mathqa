import networkx as nx

from program_graph.parse_linear_formula import ParsedLinearFormula, ArgType, Arg
from math_qa.math_qa import get_constants_descriptors


def parsed_linear_formula_to_nx_graph(parsed_lf: ParsedLinearFormula, n_inputs: int) -> nx.MultiDiGraph:
    """Creates a Graph where the Nodes are Arg, and have the 'label' property
    label is const_name, n{i} for inputs, and op type in operation node."""
    # create the graph
    graph = nx.MultiDiGraph()

    # add the constant nodes
    for const_desc in get_constants_descriptors():
        node = Arg(ArgType.const, key=const_desc.name)
        graph.add_node(node, label=const_desc.name)

    # add the input nodes
    for i in range(n_inputs):
        node = Arg(ArgType.input, key=i)
        graph.add_node(node, label=f"n{i}")

    # add the op nodes
    for op_i in range(len(parsed_lf)):
        node = Arg(ArgType.temp, key=op_i)
        op_name = parsed_lf.op_list[op_i]
        graph.add_node(node, label=op_name)

        # add the edges
        arg_list = parsed_lf.arg_list_list[op_i]
        for arg_idx, arg in enumerate(arg_list):
            graph.add_edge(arg, node, key=arg_idx)

    return graph
