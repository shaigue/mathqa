"""Script to extract the program graphs from the linear formulas in MathQA"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import logging
import pickle
import re
import time
from typing import FrozenSet

import networkx as nx
import pandas as pd

from math_qa.dataset import load_json
from math_qa.dataset import is_commutative

import config

logging.basicConfig(level=logging.INFO, filemode='w', filename='log.txt')


class NodeType(Enum):
    operation = 0
    constant = 1
    input = 2


class Program:
    """
    constant nodes are saved with their names,
    operation nodes are saved with non-negative index (index, {operation_name: <operation_name>})
    variable nodes are saved with negative number (-1 - input_num) 
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    @staticmethod
    def input_node_identifier(input_num: int) -> int:
        assert input_num >= 0, "input_num should be non-negative"
        return -1 - input_num

    @staticmethod
    def input_node_to_num(node_identifier: int) -> int:
        assert node_identifier < 0, "input node identifier should be negative"
        return -1 - node_identifier

    @staticmethod
    def get_node_type(node) -> NodeType:
        if isinstance(node, str):
            return NodeType.constant
        if node < 0:
            return NodeType.input
        return NodeType.operation

    @staticmethod
    def arg_to_str(arg_id, operation_map=None):
        node_type = Program.get_node_type(arg_id)
        if node_type == NodeType.input:
            return f"n{Program.input_node_to_num(arg_id)}"
        if node_type == NodeType.operation:
            if operation_map is not None:
                return f"#{operation_map[arg_id]}"
            return f"#{arg_id}"
        return arg_id

    @classmethod
    def from_linear_formula(cls, linear_formula: str):
        program = cls()
        operation_name_regexp = re.compile(r'(\w+)\((.*?)\)')

        linear_formula = linear_formula.replace(' ', '')
        linear_formula = linear_formula.split('|')

        for op_index, operation in enumerate(linear_formula):
            if len(operation) == 0:
                continue
            m = operation_name_regexp.match(operation)
            assert m is not None, "got no match"
            operation_name = m.group(1)
            arguments = m.group(2)
            arguments = arguments.split(',')
            program.graph.add_node(op_index, operation_name=operation_name)
            for arg_index, arg in enumerate(arguments):
                src_node = None
                if arg.startswith('const_'):
                    src_node = arg
                    program.graph.add_node(src_node)
                if arg.startswith('n'):
                    src_node = int(arg[1:])
                    src_node = cls.input_node_identifier(src_node)
                    program.graph.add_node(src_node)
                if arg.startswith('#'):
                    src_node = int(arg[1:])

                assert src_node is not None, "got bad argument"
                if is_commutative(operation_name):
                    program.graph.add_edge(src_node, op_index)
                else:
                    program.graph.add_edge(src_node, op_index, arg_index=arg_index)

        return program

    def __hash__(self):
        """Hashing by the number of operations in each type"""
        concatenated_operations = [operation_name for _, operation_name in self.operation_index_name_iterator()]
        concatenated_operations = sorted(concatenated_operations)
        concatenated_operations = ''.join(concatenated_operations)
        return hash(concatenated_operations)

    def __eq__(self, other):
        return nx.is_isomorphic(self.graph, other.graph, node_match=dict.__eq__, edge_match=dict.__eq__)

    def __str__(self):
        # normalize the nodes to be from 0,1,...
        operation_index_sorted = sorted(index for index, _ in self.operation_index_name_iterator())
        operation_index_normalization_map = {index: i for i, index in enumerate(operation_index_sorted)}
        linear_formula = ""
        for operation_index in operation_index_sorted:
            operation_name = self.graph.nodes[operation_index]["operation_name"]
            linear_formula += f"{operation_name}("

            # add the arguments in a sorted manner
            arguments_ids = self.graph.predecessors(operation_index)
            args_strings = []
            if is_commutative(operation_name):
                for arg_id in arguments_ids:
                    args_strings.append(self.arg_to_str(arg_id, operation_index_normalization_map))
            else:
                edges = [(arg_id, self.graph.edges[(arg_id, operation_index)]['arg_index'])
                         for arg_id in arguments_ids]
                edges.sort(key=lambda x: x[1])
                for arg_id, _ in edges:
                    args_strings.append(self.arg_to_str(arg_id, operation_index_normalization_map))

            linear_formula += ','.join(args_strings)
            linear_formula += ')|'

        return linear_formula

    def __repr__(self):
        return self.__str__()

    def operation_index_name_iterator(self):
        for node, data in self.graph.nodes.data():
            if self.get_node_type(node) == NodeType.operation:
                yield node, data['operation_name']

    def constant_iterator(self):
        for node in self.graph.nodes:
            if self.get_node_type(node) == NodeType.constant:
                yield node

    def input_iterator(self):
        for node in self.graph.nodes:
            if self.get_node_type(node) == NodeType.input:
                yield node

    def all_operations(self, node_subset: FrozenSet[int]) -> bool:
        operations_indices = {index for index, _ in self.operation_index_name_iterator()}
        return node_subset.issubset(operations_indices)

    def sub_program(self, node_subset: FrozenSet[int]):
        """Assumes that the node subset is only operation nodes"""
        assert self.all_operations(node_subset), "not only operations"
        sub_program = Program()
        sub_program.graph = nx.DiGraph(nx.subgraph(self.graph, node_subset))
        # input nodes are all of the nodes outside of the cut that have incoming edge to the program
        input_node_idx = 0
        outside_nodes = set(self.graph.nodes) - node_subset
        for outside_node in outside_nodes:
            used = False
            for successor in self.graph.successors(outside_node):
                if successor in node_subset:
                    edge_attr = self.graph.edges[(outside_node, successor)]
                    input_node_id = self.input_node_identifier(input_node_idx)
                    used = True
                    sub_program.graph.add_node(input_node_id)
                    sub_program.graph.add_edge(input_node_id, successor, **edge_attr)
            if used:
                input_node_idx += 1

        return sub_program

    def get_n_operations(self):
        return len(list(self.operation_index_name_iterator()))

    def get_n_inputs(self):
        return len(list(self.input_iterator()))

    def satisfies_constraints(self, min_size: int = None, max_size: int = None, max_inputs: int = None) -> bool:
        n_operations = self.get_n_operations()
        if min_size is not None and n_operations < min_size:
            return False
        if max_size is not None and n_operations > max_size:
            return False
        n_inputs = self.get_n_inputs()
        if max_inputs is not None and n_inputs > max_inputs:
            return False
        return True

    def function_cut_iterator(self, min_size: int = None, max_size: int = None, max_inputs: int = None):
        @dataclass(frozen=True)
        class FunctionCutSearchNode:
            contained_nodes: FrozenSet[int]
            reachable_nodes: FrozenSet[int]

            def __eq__(self, other):
                return self.contained_nodes == other.contained_nodes

        # get all the inputs and constants nodes
        inputs = frozenset(self.input_iterator())
        constants = frozenset(self.constant_iterator())
        inputs_and_constants = inputs.union(constants)

        for head, _ in self.operation_index_name_iterator():
            head_ancestors: set = nx.ancestors(self.graph, head)
            graph_without_head = nx.subgraph(self.graph, set(self.graph) - {head})
            descendants_dict = {node: nx.descendants(graph_without_head, node) for node in head_ancestors}
            bad_nodes = set()
            # nodes that are connected to nodes that are not ancestors of `head` are bad, can cause multiple outputs
            for node in head_ancestors:
                if not head_ancestors.issuperset(descendants_dict[node]):
                    bad_nodes.add(node)

            bad_nodes = bad_nodes.union(inputs_and_constants)

            def good_predecessors(node: int) -> FrozenSet[int]:
                return frozenset(self.graph.predecessors(node)) - bad_nodes

            closed_set = set()

            contained_nodes = frozenset({head})
            reachable_nodes = good_predecessors(head)
            search_node = FunctionCutSearchNode(contained_nodes, reachable_nodes)
            open_set = {search_node}

            while len(open_set) > 0:
                search_node = open_set.pop()
                closed_set.add(search_node)
                if len(search_node.contained_nodes) > max_size:
                    continue
                sub_program = self.sub_program(search_node.contained_nodes)
                if sub_program.satisfies_constraints(min_size, max_size, max_inputs):
                    yield sub_program

                for node_to_add in search_node.reachable_nodes:
                    contained_nodes = search_node.contained_nodes.union({node_to_add})
                    contained_nodes = contained_nodes.union(descendants_dict[node_to_add])
                    reachable_nodes = search_node.reachable_nodes.union(good_predecessors(node_to_add))
                    reachable_nodes = reachable_nodes - contained_nodes
                    search_node_to_add = FunctionCutSearchNode(contained_nodes, reachable_nodes)

                    if search_node_to_add not in closed_set:
                        open_set.add(search_node_to_add)


def get_data(part='test'):
    return load_json(config.MATH_QA_PATH / f'{part}.json')


def count_repeating_functions(part, min_size, max_size, max_inputs) -> Counter:
    pickle_file = f"{part}_function_counter.pkl"
    function_counter = Counter()
    data = get_data(part)
    for i, dp in enumerate(data):
        logging.info(f"{i+1} out of {len(data)} in {part}...")
        program = Program.from_linear_formula(dp.linear_formula)
        function_counter.update(program.function_cut_iterator(min_size, max_size, max_inputs))

    with open(pickle_file, "wb") as fd:
        pickle.dump(function_counter, fd)

    return function_counter


def count_for_each_partition(min_size, max_size, max_inputs):
    partitions = ['test', 'train', 'dev']
    rows_list = []
    counters = {part: count_repeating_functions(part, min_size, max_size, max_inputs) for part in partitions}
    for part in partitions:
        counter = counters[part]
        part_programs = list(counter.keys())
        for program in part_programs:
            row = {part0: counters[part0].pop(program, 0) for part0 in partitions}
            row['linear_formula'] = str(program)
            row['size'] = program.get_n_operations()
            row['n_inputs'] = program.get_n_inputs()
            rows_list.append(row)
    data_frame = pd.DataFrame(rows_list, columns=['linear_formula', 'size', 'n_inputs'] + partitions)
    data_frame.sort_values(by=partitions, ascending=False, inplace=True, ignore_index=True)
    data_frame = data_frame[:10_000]
    data_frame.to_csv('count_for_each_partition.csv', index=False)


if __name__ == "__main__":
    start_time = time.time()
    count_for_each_partition(min_size=2, max_size=10, max_inputs=5)
    elapsed_time = time.time() - start_time
    logging.info(f"***finished***")
    logging.info(f"time={elapsed_time} seconds")

