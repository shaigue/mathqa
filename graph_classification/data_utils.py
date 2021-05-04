import re
from collections import namedtuple
from typing import Iterator

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

import config
from graph_classification.common import get_node_labels
from math_qa import math_qa


def get_node_label_encoder() -> LabelEncoder:
    le = LabelEncoder()
    labels = get_node_labels()
    le.fit(labels)
    return le


def get_category_label_encoder() -> LabelEncoder:
    le = LabelEncoder()
    categories = math_qa.get_categories()
    le.fit(categories)
    return le


ArgSelectionDatapoint = namedtuple('ArgSelectionDatapoint', ['adj_tensor', 'node_labels', 'dst_node', 'src_node'])


class Node:
    def __init__(self, label: str, incoming: list[int]):
        self.label = label
        self.incoming = incoming


class FormulaGraph:
    def __init__(self, linear_formula: str, max_allowed_n_args: int = config.MAX_ARGS):
        self.in_adj_list = []
        # first add all the constants
        for const_desc in math_qa.get_constants_descriptors():
            self.in_adj_list.append(Node(const_desc.name, []))

        self.n_const = len(self.in_adj_list)

        # now add the inputs
        input_regexp = re.compile(r'n\d+')
        self.n_inputs = 0
        for match in input_regexp.findall(linear_formula):
            input_num = int(match[1:])
            if (input_num + 1) > self.n_inputs:
                self.n_inputs = input_num + 1

        for input_num in range(self.n_inputs):
            self.in_adj_list.append(Node(f'n{input_num}', []))

        # now add the ops
        op_list = linear_formula.split('|')
        op_list = list(filter(lambda x: x != '', op_list))
        self.n_ops = len(op_list)

        for op_desc in op_list:
            op_desc = op_desc.replace(')', '')
            op_name, args = op_desc.split('(')
            args = args.split(',')
            incoming = []
            for arg in args:
                if arg.startswith('#'):
                    index = int(arg[1:])
                    incoming.append(index + self.n_const + self.n_inputs)
                else:
                    for i in range(self.n_const + self.n_inputs):
                        if self.in_adj_list[i].label == arg:
                            incoming.append(i)
                            break

            self.in_adj_list.append(Node(op_name, incoming))
            self.n_nodes = len(self.in_adj_list)
            max_n_args = max(len(node.incoming) for node in self.in_adj_list)
            assert max_n_args <= max_allowed_n_args
            self.n_edge_types = max_allowed_n_args

    def get_adj_tensor(self) -> ndarray:
        """

        :return: a bool array stating if there is an edge of type t, from node i to node j in the entry
            t, i, j. dimensions are (n_edge_types, n_nodes, n_nodes).
        """
        adj_tensor = np.zeros((self.n_edge_types, self.n_nodes, self.n_nodes), dtype=np.bool8)
        for i, node in enumerate(self.in_adj_list):
            for t, j in enumerate(node.incoming):
                adj_tensor[t, j, i] = True

        return adj_tensor

    def get_node_labels(self) -> ndarray:
        """

        :return: an array of string node_labels
        """
        return np.array([node.label for node in self.in_adj_list])

    def iter_partial(self) -> Iterator[ArgSelectionDatapoint]:
        """

        :yields: a ArgSelectionDatapoint
        """
        node_labels = self.get_node_labels()
        adj_tensor = np.zeros((self.n_edge_types, self.n_nodes, self.n_nodes), dtype=np.bool8)

        # first place all the const nodes and the input nodes
        for dst_node_i in range(self.n_inputs + self.n_const, self.n_nodes):
            node = self.in_adj_list[dst_node_i]
            for edge_type, src_node_i in enumerate(node.incoming):
                datapoint = ArgSelectionDatapoint(
                    adj_tensor=adj_tensor[:, :dst_node_i+1, :dst_node_i+1].copy(),
                    node_labels=node_labels[:dst_node_i+1].copy(),
                    dst_node=dst_node_i,
                    src_node=src_node_i,
                )
                yield datapoint
                adj_tensor[edge_type, src_node_i, dst_node_i] = True


def get_formula_graphs(partition: str) -> list[FormulaGraph]:
    assert partition in ['train', 'dev', 'test']
    raw_entries = math_qa.load_dataset(partition, config.MATHQA_DIR)
    return [FormulaGraph(e.linear_formula) for e in raw_entries]


class GraphClassificationDataset(Dataset):
    def __init__(self, partition: str):
        assert partition in ['train', 'dev', 'test']
        category_encoder = get_category_label_encoder()
        raw_entries = math_qa.load_dataset(partition, config.MATHQA_DIR)
        node_label_encoder = get_node_label_encoder()
        self.adj_tensors, self.node_labels, self.categories = [], [], []

        for entry in raw_entries:
            formula_graph = FormulaGraph(entry.linear_formula)

            adj_tensor = torch.tensor(formula_graph.get_adj_tensor(), dtype=torch.float32)
            self.adj_tensors.append(adj_tensor)

            label = node_label_encoder.transform(formula_graph.get_node_labels())
            label = torch.tensor(label, dtype=torch.long)
            self.node_labels.append(label)

            self.categories.append(entry.category)

        self.categories = category_encoder.transform(self.categories)

    def __getitem__(self, index) -> tuple[Tensor, Tensor, int]:
        return self.adj_tensors[index], self.node_labels[index], self.categories[index]

    def __len__(self):
        return len(self.adj_tensors)


def test_dataset():
    partitions = ['train', 'test', 'dev']
    for part in partitions:
        dataset = GraphClassificationDataset(part)
        n_edge_type = config.MAX_ARGS
        for i in range(len(dataset)):
            adj_tensor, node_labels, category = dataset[i]
            n_edge_type0, n_nodes, n_nodes0 = adj_tensor.shape
            assert n_edge_type0 == n_edge_type
            assert n_nodes == n_nodes0
            assert node_labels.shape == (n_nodes,)

    print('finished')


if __name__ == "__main__":
    test_dataset()
