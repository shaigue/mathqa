from dataclasses import dataclass

import networkx as nx
import torch
from torch import nn
from torch import Tensor, LongTensor, FloatTensor

# TODO: currently only works with teacher forcing
# TODO: use multiple possible actions with KLD / CE
# TODO: try more complex attention mechanism
from models.attention_pool import AttentionPool
from models.node_embedding import NodeEmbedding


@dataclass
class GraphGeneratorInput:
    adj_tensor: Tensor
    node_labels: Tensor
    generate_from: int
    n_prop_steps: int


@dataclass
class GraphGeneratorOutput:
    node_labels_logits: Tensor
    node_select_logits: Tensor


class GraphGenerator(nn.Module):
    def __init__(self, n_node_labels: int, n_edge_types: int, node_embedding_dim: int,
                 text_vector_dim: int,
                 bias: bool = True, backward_edges: bool = True,
                 ):
        super().__init__()
        # embeds the nodes to vectors
        self.node_embedding_layer = NodeEmbedding(n_node_labels, node_embedding_dim,
                                                  n_edge_types, bias, backward_edges)
        # aggregates the nodes embedding to a single vector
        self.node_embedding_agg = AttentionPool(node_embedding_dim)
        # takes the text and node aggregate vectors and predict the next node or stop
        self.node_label_classifier = nn.Linear(text_vector_dim + node_embedding_dim,
                                               n_node_labels + 1)

    def predict_next_node_label(self):
        pass

    def predict_node_to_connect(self):
        pass

    def forward_teacher_forcing_with_text_vector(self, text_vector: FloatTensor,
                                                 graph: nx.MultiDiGraph) -> list[Tensor]:
        """Creates logits of the actions in each step, given the text vector and graph to generate

        :param text_vector:
        :param graph:
        :return:
        """
        # TODO: verify input constraints
        # TODO: verify output constraints
        pass

    def forward_teacher_forcing_with_text_tokens(self, text_tokens: LongTensor,
                                                 graph: nx.MultiDiGraph) -> list[Tensor]:
        """Creates logits of the actions in each step, given the text tokens and graph to generate

        :param text_tokens:
        :param graph:
        :return:
        """
        # TODO: verify input constraints
        # TODO: verify output constraints
        pass
