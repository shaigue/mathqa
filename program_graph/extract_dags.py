"""Script to extract the program graphs from the linear formulas in MathQA"""
# TODO: remove this, it is code duplication for making it work for the macros

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from itertools import count, permutations
from typing import Union, Optional, Iterator

import networkx as nx
from networkx import isomorphism

from math_qa.math_qa import is_commutative
from math_qa import constants
from math_qa import operations


class NodeType(Enum):
    operation = 0
    constant = 1
    input = 2


class Node(ABC):
    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()


class ConstantNode(Node):
    def __init__(self, const_name: str):
        assert const_name in constants.const_dict, f"{const_name} is not a recognized constant."
        self.const_name = const_name

    def get_value(self):
        return constants.const_dict[self.const_name]

    def __eq__(self, other):
        if not isinstance(other, ConstantNode):
            return False
        return self.const_name == other.const_name

    def __hash__(self):
        return hash(self.const_name)

    def __str__(self):
        return self.const_name


class InputNode(Node):
    def __init__(self, arg_num: int):
        assert arg_num >= 0, f"arg_num={arg_num}. should be non-negative."
        self.arg_num = arg_num

    def get_value(self, inputs: list[Union[int, float]]) -> Union[int, float]:
        assert self.arg_num < len(inputs), f"arg_num={self.arg_num} is out-of-range for input list len={len(inputs)}"
        return inputs[self.arg_num]

    def __eq__(self, other):
        if not isinstance(other, InputNode):
            return False
        return self.arg_num == other.arg_num

    def __hash__(self):
        return hash(self.arg_num)

    def __str__(self):
        return f"n{self.arg_num}"


class OperationNode(Node):
    def __init__(self, identifier: int, operation_name: str):
        self.identifier = identifier
        self.operation_name = operation_name

    def get_value(self, inputs: list[Union[int, float]], macro_dict: Optional[dict] = None) -> Union[int, float]:
        if hasattr(operations, self.operation_name):
            func = getattr(operations, self.operation_name)
            return func(*inputs)
        elif macro_dict is not None and self.operation_name in macro_dict:
            macro = macro_dict[self.operation_name]
            return macro.eval(inputs, macro_dict)
        else:
            assert False, f"Could not find operation_name={self.operation_name}."

    def is_commutative(self):
        return is_commutative(self.operation_name)

    def __eq__(self, other):
        if not isinstance(other, OperationNode):
            return False
        return self.identifier == other.identifier

    def __hash__(self):
        return hash(self.identifier)

    def __str__(self):
        return f"<OP_NODE op_name={self.operation_name}, id={self.identifier}>"


class Program:
    """A class that represents a arithmetic formula as a graph, DAG."""
    # ==================== Public Methods ==============================================
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        # this is a cache in order to save time
        self._eq_graph = None

    @classmethod
    def from_linear_formula(cls, linear_formula: str):
        """Creating a program graph from a linear formula string."""
        program = cls()
        # remove access spaces
        linear_formula = linear_formula.replace(' ', '')
        # split with |
        linear_formula_split = linear_formula.split('|')
        # keep only non-empty strings
        linear_formula_split = list(filter(lambda x: x != '', linear_formula_split))
        # for each split, create it's arguments and operations
        for op_index, op_and_args in enumerate(linear_formula_split):
            # remove the trailing ')'
            op_and_args = op_and_args.replace(')', '')
            # split the operation name and the arguments
            op, args = op_and_args.split('(')
            # add the operation node
            op_node = OperationNode(op_index, op)
            program.graph.add_node(op_node)
            args = args.split(',')
            for arg_index, arg in enumerate(args):
                arg_node = None
                # output from previous operation node
                if arg.startswith('#'):
                    index = int(arg[1:])
                    arg_node = program._get_op_node(index)
                # input
                elif arg.startswith('n'):
                    arg_num = int(arg[1:])
                    arg_node = InputNode(arg_num)
                # constant
                elif arg.startswith('const_'):
                    arg_node = ConstantNode(arg)
                else:
                    assert False, f"cannot resolve the argument={arg}"
                assert arg_node is not None, "got None arg_node."
                # add all the edges between the arguments and the operations
                # the key represents the order of the arguments
                program.graph.add_edge(arg_node, op_node, key=arg_index)

        return program

    def eval(self, inputs: list[Union[int, float]], macro_dict: Optional[dict] = None) -> Union[int, float]:
        """Evaluate the program with the given inputs and macros(optional)."""
        macro_dict: dict[str, Program]

        node_to_value = {}
        last_node = None
        # start by taking a topological ordering of the nodes
        for node in nx.topological_sort(self.graph):
            last_node = node
            if isinstance(node, ConstantNode):
                value = node.get_value()
            elif isinstance(node, InputNode):
                value = node.get_value(inputs)
            elif isinstance(node, OperationNode):
                # need to order the arguments in the correct order
                op_inputs = self._get_op_inputs(node)
                op_inputs = [node_to_value[input_node] for input_node in op_inputs]
                value = node.get_value(op_inputs, macro_dict)
            else:
                assert False, "Should not get here."
            node_to_value[node] = value

        assert last_node is not None, "Last node is None"
        # the last node is the return value
        return node_to_value[last_node]

    def get_n_operations(self):
        return len(list(self._iter_operators_nodes()))

    def get_n_inputs(self):
        return len(list(self._iter_input_nodes()))

    def function_cut_iterator(self, min_size: int = None, max_size: int = None, max_inputs: int = None,
                              return_subsets=False):
        @dataclass(frozen=True)
        class FunctionCutSearchNode:
            contained_nodes: frozenset[Node]
            reachable_nodes: frozenset[Node]

            def __eq__(self, other):
                return self.contained_nodes == other.contained_nodes

        # get all the inputs and constants nodes
        input_nodes = frozenset(self._iter_input_nodes())
        constant_nodes = frozenset(self._iter_constants())
        inputs_and_constants = input_nodes.union(constant_nodes)

        for operator_node in self._iter_operators_nodes():
            head_ancestors: set = nx.ancestors(self.graph, operator_node)
            graph_without_head = nx.subgraph(self.graph, set(self.graph) - {operator_node})
            descendants_dict = {node: nx.descendants(graph_without_head, node) for node in head_ancestors}
            bad_nodes = set()
            # nodes that are connected to nodes that are not ancestors of `head` are bad, can cause multiple outputs
            for node in head_ancestors:
                if not head_ancestors.issuperset(descendants_dict[node]):
                    bad_nodes.add(node)

            bad_nodes = bad_nodes.union(inputs_and_constants)

            def good_predecessors(node: OperationNode) -> frozenset[OperationNode]:
                return frozenset(self.graph.predecessors(node)) - bad_nodes

            closed_set = set()

            contained_nodes = frozenset({operator_node})
            reachable_nodes = good_predecessors(operator_node)
            search_node = FunctionCutSearchNode(contained_nodes, reachable_nodes)
            open_set = {search_node}

            while len(open_set) > 0:
                search_node = open_set.pop()
                closed_set.add(search_node)
                if max_size is not None and len(search_node.contained_nodes) > max_size:
                    continue
                sub_program = self.sub_program(search_node.contained_nodes)
                if sub_program._satisfies_constraints(min_size, max_size, max_inputs):
                    if return_subsets:
                        yield sub_program, search_node.contained_nodes
                    else:
                        yield sub_program

                for node_to_add in search_node.reachable_nodes:
                    contained_nodes = search_node.contained_nodes.union({node_to_add})
                    contained_nodes = contained_nodes.union(descendants_dict[node_to_add])
                    reachable_nodes = search_node.reachable_nodes.union(good_predecessors(node_to_add))
                    reachable_nodes = reachable_nodes - contained_nodes
                    search_node_to_add = FunctionCutSearchNode(contained_nodes, reachable_nodes)

                    if search_node_to_add not in closed_set:
                        open_set.add(search_node_to_add)

    def refactor_macro(self, node_subset: frozenset[OperationNode], target_macro, symbol: str):
        """Returns the refactored program by replacing a set of nodes with a single symbol."""
        # copy the graph
        refactored_program = Program()
        refactored_program.graph = nx.MultiDiGraph(self.graph)
        # to correctly align the inputs, we first need to map the original nodes to the nodes in the local macro
        # then match those to the inputs of the target macro(is isomorphic to the local one, but might be in a different
        # order)
        local_macro, original_to_input = self.sub_program(node_subset, return_original_to_input_map=True)
        local_to_target_match = self._match_programs(local_macro, target_macro)
        assert local_to_target_match is not None, "The target and local macros are not isomorphic."
        # add the new symbol
        macro_node = OperationNode(self._get_fresh_op_identifier(), symbol)
        refactored_program.graph.add_node(macro_node)
        # add outgoing and incoming edges
        other_nodes = set(self.graph.nodes) - node_subset
        for source_node, target_node, arg_index in self.graph.edges:
            # incoming
            if source_node in other_nodes and target_node in node_subset:
                local_input = original_to_input[source_node]
                target_input = local_to_target_match[local_input]
                assert isinstance(target_input, InputNode), "Should be an input node"
                refactored_program.graph.add_edge(source_node, macro_node, target_input.arg_num)
            # outgoing
            elif source_node in node_subset and target_node in other_nodes:
                refactored_program.graph.add_edge(macro_node, target_node, arg_index)
        # remove the nodes
        refactored_program.graph.remove_nodes_from(node_subset)
        return refactored_program

    def sub_program(self, node_subset: frozenset[OperationNode], return_original_to_input_map=False):
        """Assumes that the node subset is only operation nodes.

        :param node_subset: the subset of nodes to take.
        :param return_original_to_input_map: if to return the mapping from the input nodes in the olf graph to
        the input nodes nodes in the new graph.
        """
        assert self._is_operations_only(node_subset), "not only operations"

        sub_program = Program()
        # copy the subgraph
        sub_program.graph = nx.MultiDiGraph(self.graph.subgraph(node_subset))
        # first find the input nodes
        inputs = set()
        for node in node_subset:
            inputs.update(self._get_op_inputs(node))
        # remove the inputs that are already inside
        inputs.difference_update(node_subset)
        original_to_input_map = {}
        for i, input_node in enumerate(inputs):
            new_input_node = InputNode(i)
            original_to_input_map[input_node] = new_input_node

        # now add the edges from the inputs to the new program
        for source, target, arg_index in self.graph.edges:
            if source in inputs and target in node_subset:
                sub_program.graph.add_edge(original_to_input_map[source], target, arg_index)

        if return_original_to_input_map:
            return sub_program, original_to_input_map

        return sub_program

    # ======================= Private methods ========================================

    def _get_op_node(self, index: int) -> OperationNode:
        for node in self.graph.nodes:
            if isinstance(node, OperationNode) and node.identifier == index:
                return node
        assert False, f"Could not find operation node {index}"

    def _iter_operators_nodes(self) -> Iterator[OperationNode]:
        for node in self.graph.nodes:
            if isinstance(node, OperationNode):
                yield node

    def _get_eq_graph(self):
        """Returns the graph for isomorphic matching"""
        if self._eq_graph is not None:
            return self._eq_graph

        self._eq_graph = nx.MultiDiGraph()
        # copy all the nodes, add the operation names as attributes
        for node in self.graph.nodes:
            attr = {}
            if isinstance(node, OperationNode):
                attr['op_name'] = node.operation_name
            self._eq_graph.add_node(node, **attr)
        for input_node, output_node, arg_index in self.graph.edges:
            assert isinstance(output_node, OperationNode), "only operation nodes should have inputs"
            attr = {}
            # if it is not commutative then give meaning to the ordering
            if not output_node.is_commutative():
                attr['arg_index'] = arg_index
            self._eq_graph.add_edge(input_node, output_node, **attr)

        return self._eq_graph

    @staticmethod
    def _edge_match(e1, e2):
        # check if there exists a permutation of them so they match on their values
        if len(e1) != len(e2):
            return False
        n = len(e1)
        for perm in permutations(range(n)):
            if all([e1[i] == e2[perm[i]] for i in range(n)]):
                return True
        return False

    @staticmethod
    def _match_programs(program1, program2):
        """try to match program1 and program2, by finding an isomorphism operation preserving, and edge order preserving on
        non-commutative operations.
        :returns a matching (mapping from program1 nodes to program2 nodes) if exists one, else returns None.
        """
        program1: Program
        program2: Program
        g1 = program1._get_eq_graph()
        g2 = program2._get_eq_graph()
        matcher = isomorphism.MultiDiGraphMatcher(g1, g2, node_match=dict.__eq__, edge_match=Program._edge_match)
        if matcher.is_isomorphic():
            return matcher.mapping
        else:
            return None

    def _get_op_inputs(self, node: OperationNode) -> list[Node]:
        """return an ordered list of the nodes that the operation depends on"""
        op_inputs = []
        for pred_node, edges in self.graph.pred[node].items():
            for arg_num in edges:
                op_inputs.append((pred_node, arg_num))
        op_inputs.sort(key=lambda x: x[1])
        op_inputs = [x[0] for x in op_inputs]
        return op_inputs

    def _iter_constants(self):
        for node in self.graph.nodes:
            if isinstance(node, ConstantNode):
                yield node

    def _iter_input_nodes(self):
        for node in self.graph.nodes:
            if isinstance(node, InputNode):
                yield node

    def _is_operations_only(self, node_subset: frozenset[OperationNode]) -> bool:
        all_ops = set(self._iter_operators_nodes())
        return node_subset.issubset(all_ops)

    def _get_fresh_op_identifier(self):
        """Returns an unused operation id"""
        used_operation_indices = {op_node.identifier for op_node in self._iter_operators_nodes()}
        for i in count():
            if i not in used_operation_indices:
                return i

    def _satisfies_constraints(self, min_size: int = None, max_size: int = None, max_inputs: int = None) -> bool:
        n_operations = self.get_n_operations()
        if min_size is not None and n_operations < min_size:
            return False
        if max_size is not None and n_operations > max_size:
            return False
        n_inputs = self.get_n_inputs()
        if max_inputs is not None and n_inputs > max_inputs:
            return False
        return True

    # ===================== Operator overloading =======================================

    def __hash__(self):
        """Hashing by the number of operations in each type"""
        ops = [op_node.operation_name for op_node in self._iter_operators_nodes()]
        ops = sorted(ops)
        ops = ''.join(ops)
        return hash(ops)

    def __eq__(self, other):
        match = self._match_programs(self, other)
        return match is not None

    def __str__(self):
        """Converts the graph into a linear formula"""
        linear_formula = ""
        node_ordering = {}
        node_index = 0
        for node in nx.topological_sort(self.graph):
            if isinstance(node, OperationNode):
                # assign the node its number
                node_ordering[node] = node_index
                node_index += 1
                linear_formula += node.operation_name + "("
                input_nodes = self._get_op_inputs(node)
                for input_node in input_nodes:
                    if isinstance(input_node, OperationNode):
                        linear_formula += f"#{node_ordering[input_node]}"
                    else:
                        linear_formula += str(input_node)
                    linear_formula += ','
                # drop the last ','
                linear_formula = linear_formula[:-1]
                linear_formula += ")|"
        # drop the last |
        linear_formula = linear_formula[:-1]
        return linear_formula

    def __repr__(self):
        return self.__str__()
