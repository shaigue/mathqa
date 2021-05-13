"""Implement the graph generation dataset and dataloaders"""
# TODO: write tests
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from extract_pretrained_problem_vectors import get_vector_path
from math_qa import math_qa
from math_qa.math_qa import get_n_inputs
from models.graph_generator import has_incoming_edge_of_type
from program_processing.linear_formula_to_tensor import linear_formula_to_tensor
from program_processing.common import create_item_to_index_dict, get_max_n_args, get_op_label_indices, \
    get_node_label_to_idx, get_node_label_to_edge_types


@dataclass
class GraphGenerationDatapoint:
    adj_tensor: Tensor
    node_labels: Tensor
    start_node_i: int
    condition_vector: Tensor
    node_targets: Tensor
    edge_targets: Tensor

    def to(self, device):
        d = self.__dict__.copy()
        for k in d.keys():
            if isinstance(d[k], Tensor):
                d[k] = d[k].to(device)
        return GraphGenerationDatapoint(**d)


class MathQAGraphGenerationDataset(Dataset):
    def __init__(self, partition: str, dummy: bool = False):
        if dummy:
            raw_data = math_qa.load_dataset('train')
            raw_data = raw_data[:100]
        else:
            raw_data = math_qa.load_dataset(partition)

        self.entries = []

        n_edge_types = get_max_n_args()
        node_label_to_idx = get_node_label_to_idx()
        op_label_indices = get_op_label_indices()
        stop_token = len(op_label_indices)
        # this is because we do not generate labels that are not operations,
        # so the logits dimensions must match
        real_label_to_generated_label = create_item_to_index_dict(op_label_indices)

        for i, entry in enumerate(raw_data):
            n_inputs = get_n_inputs(entry)
            graph_tensor = linear_formula_to_tensor(entry.linear_formula,
                                                    n_inputs,
                                                    n_edge_types,
                                                    node_label_to_idx)
            start_node_i = 0
            while graph_tensor.node_labels[start_node_i] not in op_label_indices:
                start_node_i += 1

            datapoint = GraphGenerationDatapoint(
                adj_tensor=graph_tensor.adj_tensor,
                node_labels=graph_tensor.node_labels,
                start_node_i=start_node_i,
                condition_vector=torch.load(get_vector_path(partition, i)).squeeze(),
                node_targets=create_node_cross_entropy_target(graph_tensor.node_labels, stop_token, start_node_i,
                                                              real_label_to_generated_label),
                edge_targets=create_edge_cross_entropy_target(graph_tensor.adj_tensor, start_node_i),
            )
            self.entries.append(datapoint)

    def __getitem__(self, i: int) -> GraphGenerationDatapoint:
        return self.entries[i]

    def __len__(self):
        return len(self.entries)


def get_dataloader(partition: str, dummy=False) -> DataLoader:
    shuffle = True if partition == 'train' else False
    dataset = MathQAGraphGenerationDataset(partition, dummy)
    # batch size is None to disable automatic batching, and collate_fn is the identity function
    return DataLoader(dataset=dataset, batch_size=None, shuffle=shuffle, collate_fn=lambda x: x)


def create_edge_cross_entropy_target(adj_tensor: Tensor, start_node_i: int) -> Tensor:
    n_edge_types, n_nodes, _ = adj_tensor.shape
    target = []
    for node_i in range(start_node_i, n_nodes):
        for edge_type in range(n_edge_types):
            if has_incoming_edge_of_type(adj_tensor, edge_type, node_i):
                src_node_i = torch.argmax(adj_tensor[edge_type, :, node_i])
                target.append(src_node_i)

    return torch.tensor(target, dtype=torch.long)


def create_node_cross_entropy_target(node_labels: Tensor, stop_token: int, start_node_i: int,
                                     real_label_to_generated_label: dict[int, int]) -> Tensor:
    n_nodes = node_labels.shape[0]
    target = torch.empty(n_nodes - start_node_i + 1, dtype=torch.long)
    for i in range(len(target) - 1):
        # need to take item() otherwise cannot index the dictionary with it
        node_label = node_labels[start_node_i + i].item()
        target[i] = real_label_to_generated_label[node_label]
    target[-1] = stop_token
    return target
