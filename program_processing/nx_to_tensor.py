import networkx as nx
import torch

# TODO: test
from program_processing.common import GraphTensor


def nx_to_tensor(nx_graph: nx.MultiDiGraph, node_labels_to_idx: dict[str, int],
                 n_edge_types: int) -> GraphTensor:
    """Converts an nx graph data structure to adjacency matrix and integer
    node labels
    """
    n_nodes = nx_graph.number_of_nodes()
    adj_tensor = torch.zeros(n_edge_types, n_nodes, n_nodes, dtype=torch.float32)
    node_labels = torch.empty(n_nodes, dtype=torch.long)

    node_to_idx = {}
    for i, (node, attr) in enumerate(nx_graph.nodes(data=True)):
        label_idx = node_labels_to_idx[attr['label']]
        node_to_idx[node] = i
        node_labels[i] = label_idx

    for src_node, dst_node, edge_type in nx_graph.edges(keys=True):
        src_idx = node_to_idx[src_node]
        dst_idx = node_to_idx[dst_node]
        adj_tensor[edge_type, src_idx, dst_idx] += 1

    return GraphTensor(adj_tensor, node_labels)
