from collections import namedtuple

import torch
from torch.utils.data import Dataset

from graph_classification.data_utils import ArgSelectionDatapoint, get_formula_graphs, get_node_label_encoder


class ArgSelectionDataset(Dataset):
    def __init__(self, partition: str):
        formula_graphs = get_formula_graphs(partition)
        node_label_encoder = get_node_label_encoder()
        self.datapoints = []
        for fg in formula_graphs:
            for dp in fg.iter_partial():
                dp = ArgSelectionDatapoint(
                    adj_tensor=torch.tensor(dp.adj_tensor, dtype=torch.float32),
                    node_labels=torch.tensor(node_label_encoder.transform(dp.node_labels), dtype=torch.long),
                    dst_node=dp.dst_node,
                    src_node=dp.src_node,
                )
                self.datapoints.append(dp)

    def __getitem__(self, index) -> ArgSelectionDatapoint:
        return self.datapoints[index]

    def __len__(self):
        return len(self.datapoints)


