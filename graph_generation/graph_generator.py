import networkx as nx
import torch
from torch import nn
from torch import Tensor, LongTensor, FloatTensor


class GraphGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        pass

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
