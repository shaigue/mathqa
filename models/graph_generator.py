import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

# TODO: currently only works with teacher forcing
# TODO: use multiple possible actions with KLD / CE
# TODO: try more complex attention mechanism
from models.attention_pool import AttentionPool
from models.general_attention import GeneralAttention
from models.node_embedding import NodeEmbedding


class GraphGenerator(nn.Module):
    def __init__(self, n_node_labels: int, n_edge_types: int, node_embedding_dim: int,
                 text_vector_dim: int,
                 bias: bool = True, backward_edges: bool = True,
                 ):
        super().__init__()
        self.n_edge_types = n_edge_types
        # embeds the nodes to vectors
        self.embedding_layer = NodeEmbedding(n_node_labels, node_embedding_dim,
                                             n_edge_types, bias, backward_edges)
        # aggregates the nodes embedding to a single vector
        self.embedding_agg_layer = AttentionPool(node_embedding_dim)
        # takes the text and node aggregate vectors and predict the next node or stop
        self.label_classifier = nn.Linear(text_vector_dim + node_embedding_dim,
                                          n_node_labels + 1)
        # used for selecting the nodes with attention
        self.select_attention_layer = GeneralAttention(node_embedding_dim, node_embedding_dim)

    def forward(self, text_vector: Tensor, adj_tensor: Tensor, node_labels: Tensor,
                start_node_i: int, n_prop_steps: int) -> dict[str, Tensor]:
        # TODO: refactor, too long!
        n_nodes = adj_tensor.shape[1]

        select_logits_list = []
        label_logits_list = []

        for curr_node_i in range(start_node_i, n_nodes + 1):
            # create the inputs for the add node step
            partial_adj_tensor = adj_tensor[:, :curr_node_i - 1, :curr_node_i - 1]
            partial_node_labels = node_labels[:curr_node_i - 1]

            node_embedding = self.embedding_layer(partial_adj_tensor,
                                                  partial_node_labels,
                                                  n_prop_steps)

            node_embedding_agg = self.embedding_agg_layer(node_embedding, node_embedding)

            # use the classifier to predict the next node label
            label_classifier_in = torch.cat([text_vector, node_embedding_agg]).unsqueeze(0)
            label_logits = self.label_classifier(label_classifier_in).squeeze()
            label_logits_list.append(label_logits)

            # create the input for the node selection step, "hide" the true edges
            partial_node_labels = node_labels[:curr_node_i]
            partial_adj_tensor = adj_tensor[:, :curr_node_i, :curr_node_i].clone()
            partial_adj_tensor[:, :, curr_node_i] = 0

            for edge_type in range(self.n_edge_types):
                # assumes only a single incoming edge of each type,
                # and if there aren't an edge of type i+1 if none of type i
                # and that the graph is DAG
                # TODO: assert to verify assumption

                # if there are no edges of this type then break and go to the next node
                if torch.all(adj_tensor[edge_type, :, curr_node_i] == 0):
                    break

                node_embedding = self.embedding_layer(partial_adj_tensor,
                                                      partial_node_labels,
                                                      n_prop_steps)
                # calculate the attention scores between the current node, and other nodes
                keys = node_embedding[:curr_node_i - 1]
                query = node_embedding[curr_node_i]
                select_logits = self.select_attention_layer(keys, query)
                # pad it with -inf so that the tensors will have the same size,
                # and that it will not affect the softmax score
                select_logits = F.pad(select_logits, (0, n_nodes - curr_node_i + 1),
                                      value=-1e9)
                select_logits_list.append(select_logits)

                # update the partial adj_tensor to have the correct value of this edge_type
                partial_adj_tensor[edge_type, :, curr_node_i] = adj_tensor[edge_type, :, curr_node_i]

        select_logits = torch.stack(select_logits_list)
        label_logits = torch.stack(label_logits_list)
        return {
            'select_logits': select_logits,
            'label_logits': label_logits,
        }



