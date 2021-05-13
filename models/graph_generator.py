from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from models.attention_pool import AttentionPool
from models.general_attention import GeneralAttention
from models.node_embedding import NodeEmbedding
# TODO: use an abstract graph representation instead of
#   directly using the adj tensor, node_labels
# TODO: currently only works with teacher forcing
# TODO: use multiple possible actions with KLD / CE
# TODO: try more complex attention mechanism


@dataclass
class GraphGeneratorTrainOutput:
    edge_logits: Tensor
    node_logits: Tensor


def has_incoming_edge_of_type(adj_tensor: Tensor, edge_type: int,
                              node_i: int) -> bool:
    return torch.any(adj_tensor[edge_type, :, node_i] != 0).item()


def graph_generator_loss_fn(output: GraphGeneratorTrainOutput,
                            edge_targets: Tensor, node_targets: Tensor) -> Tensor:
    edge_loss = F.cross_entropy(output.edge_logits, edge_targets)
    node_loss = F.cross_entropy(output.node_logits, node_targets)
    return edge_loss + node_loss


class GraphGenerator(nn.Module):
    def __init__(self, n_node_labels: int, n_edge_types: int, node_embedding_dim: int, condition_dim: int,
                 node_label_to_edge_types: dict[int, list[int]], node_labels_to_generate: list[int],
                 bias: bool = True, backward_edges: bool = True):
        """Generates a conditional graph that is DAG

        :param n_node_labels: number of all the possible node labels
        :param n_edge_types: number of all possible edge types
        :param node_embedding_dim: embedding dimension for nodes
        :param condition_dim: dimension for condition vector
        :param node_label_to_edge_types: mapping from node label, to the edge types that needs to be defined for it
        :param node_labels_to_generate: a subset of the nodes that can be generated
        :param bias: if to have bias in the graph embedding layer
        :param backward_edges: if to use backward edges in the graph embedding layer
        """
        self._check_init_inputs(n_node_labels, n_edge_types, node_embedding_dim, condition_dim,
                                node_label_to_edge_types, node_labels_to_generate)
        super().__init__()
        self.n_edge_types = n_edge_types
        self.n_node_labels = n_node_labels
        self.node_label_to_edge_types = node_label_to_edge_types
        self.node_labels_to_generate = node_labels_to_generate
        self.n_node_labels_to_generate = len(node_labels_to_generate)

        self.node_embedding_layer = NodeEmbedding(n_node_labels, node_embedding_dim,
                                                  n_edge_types, bias, backward_edges)
        self.node_embedding_pool_layer = AttentionPool(node_embedding_dim)
        self.node_label_classifier = nn.Linear(condition_dim + node_embedding_dim,
                                               self.n_node_labels_to_generate + 1)
        self.select_dst_node_layer = GeneralAttention(node_embedding_dim, node_embedding_dim + condition_dim)

    def forward(self, adj_tensor: Tensor, node_labels: Tensor,
                start_node_i: int, n_prop_steps: int,
                condition_vector: Tensor) -> GraphGeneratorTrainOutput:

        n_nodes = adj_tensor.shape[1]
        edge_logits_list = []
        node_logits_list = []

        for curr_node_i in range(start_node_i, n_nodes + 1):
            partial_adj_tensor, partial_node_labels = self._prepare_predict_node_partial_graph(
                adj_tensor, node_labels, curr_node_i
            )
            node_logits = self._predict_node(partial_adj_tensor, partial_node_labels, condition_vector, n_prop_steps)
            node_logits_list.append(node_logits)

            if curr_node_i == n_nodes:
                break

            # need to take item(), otherwise cannot index the dictionary
            true_node_label = node_labels[curr_node_i].item()
            for edge_type in self.node_label_to_edge_types[true_node_label]:
                partial_adj_tensor, partial_node_labels = self._prepare_predict_edge_partial_graph(
                    adj_tensor, node_labels, curr_node_i, edge_type
                )
                edge_logits = self._predict_edge(partial_adj_tensor, partial_node_labels,
                                                 condition_vector, n_prop_steps)
                # pad it with -inf so that the tensors will have the same size, and will small effect on softmax
                edge_logits = F.pad(edge_logits, (0, n_nodes - curr_node_i),
                                    value=-1e-9)
                edge_logits_list.append(edge_logits)

        edge_logits = torch.stack(edge_logits_list)
        node_logits = torch.stack(node_logits_list)
        return GraphGeneratorTrainOutput(edge_logits, node_logits)

    def generate(self, partial_adj_tensor: Tensor, partial_node_labels: Tensor,
                 condition_vector: Tensor, n_prop_steps: int, max_nodes: int = 200) -> tuple[Tensor, Tensor]:

        n_start_nodes = partial_node_labels.shape[0]
        for dst_node in range(n_start_nodes, max_nodes):
            node_logits = self._predict_node(partial_adj_tensor, partial_node_labels, condition_vector, n_prop_steps)
            pred_node_label = torch.argmax(node_logits).item()

            if pred_node_label == self.n_node_labels_to_generate:   # The stop symbol
                break

            pred_node_label = self._generated_label_to_real_label(pred_node_label)
            partial_adj_tensor, partial_node_labels = self._add_node(partial_adj_tensor, partial_node_labels,
                                                                     pred_node_label)

            for edge_type in self.node_label_to_edge_types[pred_node_label]:
                src_node_logits = self._predict_edge(partial_adj_tensor, partial_node_labels,
                                                     condition_vector, n_prop_steps)
                pred_src_node = torch.argmax(src_node_logits).item()
                partial_adj_tensor = self._add_edge(partial_adj_tensor, pred_src_node, dst_node,
                                                    edge_type)

        return partial_adj_tensor, partial_node_labels

    def _predict_node(self, partial_adj_tensor: Tensor, partial_node_labels: Tensor,
                      condition_vector: Tensor, n_prop_steps: int) -> Tensor:

        node_embedding = self.node_embedding_layer(partial_adj_tensor,
                                                   partial_node_labels,
                                                   n_prop_steps)
        node_embedding_agg = self.node_embedding_pool_layer(node_embedding, node_embedding)
        label_classifier_in = torch.cat([condition_vector, node_embedding_agg])
        label_classifier_in = label_classifier_in.unsqueeze(0)
        return self.node_label_classifier(label_classifier_in).squeeze()

    def _predict_edge(self, partial_adj_tensor: Tensor,
                      partial_node_labels: Tensor, condition_vector: Tensor,
                      n_prop_steps: int) -> Tensor:

        node_embedding = self.node_embedding_layer(partial_adj_tensor,
                                                   partial_node_labels,
                                                   n_prop_steps)
        keys = node_embedding[:-1]
        query = torch.cat([condition_vector, node_embedding[-1]])
        select_logits = self.select_dst_node_layer(keys, query)
        return select_logits

    def _generated_label_to_real_label(self, node_label: int) -> int:
        return self.node_labels_to_generate[node_label]

    @staticmethod
    def _prepare_predict_node_partial_graph(adj_tensor: Tensor, node_labels: Tensor,
                                            curr_node_i: int) -> tuple[Tensor, Tensor]:
        partial_adj_tensor = adj_tensor[:, :curr_node_i, :curr_node_i]
        partial_node_labels = node_labels[:curr_node_i]
        return partial_adj_tensor, partial_node_labels

    @staticmethod
    def _prepare_predict_edge_partial_graph(adj_tensor: Tensor, node_labels: Tensor,
                                            curr_node_i: int, edge_type: int) -> tuple[Tensor, Tensor]:
        partial_node_labels = node_labels[:curr_node_i + 1]
        partial_adj_tensor = adj_tensor[:, :curr_node_i + 1, :curr_node_i + 1].clone()
        # hide the edges that are not yet generated
        partial_adj_tensor[edge_type:, :, curr_node_i] = 0
        return partial_adj_tensor, partial_node_labels

    @staticmethod
    def _add_node(adj_tensor: Tensor, node_labels: Tensor, node_label: int) -> tuple[Tensor, Tensor]:
        node_labels = F.pad(node_labels, (0, 1), mode='constant', value=node_label)
        adj_tensor = F.pad(adj_tensor, (0, 1, 0, 1), mode='constant', value=0)
        return adj_tensor, node_labels

    @staticmethod
    def _add_edge(adj_tensor: Tensor, src_node: int, dst_node: int, edge_type: int) -> Tensor:
        adj_tensor[edge_type, src_node, dst_node] += 1
        return adj_tensor

    @staticmethod
    def _check_init_inputs(n_node_labels: int, n_edge_types: int, node_embedding_dim: int, condition_dim: int,
                           node_label_to_edge_types: dict[int, list[int]], node_labels_to_generate: list[int]):
        assert n_node_labels > 0
        assert n_edge_types > 0
        assert node_embedding_dim > 0
        assert condition_dim > 0
        assert list(node_label_to_edge_types.keys()) == list(range(n_node_labels))
        assert all(0 <= node_label < n_node_labels for node_label in node_labels_to_generate)
        # check that all the elements are unique
        assert len(set(node_labels_to_generate)) == len(node_labels_to_generate)
