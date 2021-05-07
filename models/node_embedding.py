import torch
from torch import nn, Tensor
from dataclasses import dataclass, field

from models.gated_gnn import GGNNOutput, GGNN, GGNNInput

# TODO: test


@dataclass()
class NodeEmbeddingInput:
    adj_tensor: Tensor
    node_labels: Tensor
    n_prop_steps: int
    n_nodes: int = field(init=False)
    n_edge_types: int = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self):
        assert self.adj_tensor.ndim == 3
        assert self.node_labels.ndim == 1
        assert self.adj_tensor.shape[1] == self.adj_tensor.shape[2] == self.node_labels.shape[0]
        assert self.adj_tensor.device == self.node_labels.device
        assert self.n_prop_steps >= 0
        assert self.adj_tensor.dtype == torch.float32
        assert self.node_labels.dtype == torch.long
        self.n_nodes = self.adj_tensor.shape[1]
        self.n_edge_types = self.adj_tensor.shape[0]
        self.device = self.adj_tensor.device

    def to(self, device: torch.device):
        return NodeEmbeddingInput(self.adj_tensor.to(device), self.node_labels.to(device), self.n_prop_steps)


NodeEmbeddingOutput = GGNNOutput


class NodeEmbedding(nn.Module):
    def __init__(self, n_node_labels: int, hidden_dim: int, n_edge_types: int,
                 bias: bool = True, backward_edges: bool = True):
        super(NodeEmbedding, self).__init__()
        assert n_node_labels > 0
        self.n_node_labels = n_node_labels

        self.label_embedding = nn.Embedding(n_node_labels, hidden_dim)
        self.gnn = GGNN(hidden_dim, n_edge_types, bias, backward_edges)

    def _check_inputs(self, inputs: NodeEmbeddingInput):
        assert torch.all(inputs.node_labels < self.n_node_labels)

    def _check_outputs(self, inputs: NodeEmbeddingInput, outputs: NodeEmbeddingOutput):
        pass

    def forward(self, inputs: NodeEmbeddingInput) -> NodeEmbeddingOutput:
        self._check_inputs(inputs
                           )
        node_embedding = self.label_embedding(inputs.node_labels)
        gnn_inputs = GGNNInput(inputs.adj_tensor, node_embedding, inputs.n_prop_steps)
        outputs = self.gnn(gnn_inputs)

        self._check_outputs(inputs, outputs)
        return outputs
