import unittest
from random import randint
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.common import one_pad
from models.attention_pool import AttentionPool
from models.gated_gnn import GGNN
from models.general_attention import GeneralAttention
from models.graph_generator import GraphGenerator, GraphGeneratorTrainOutput, graph_generator_loss_fn
from data_utils.mathqa_graph_generation_data import create_edge_cross_entropy_target, create_node_cross_entropy_target
from models.node_embedding import NodeEmbedding
from program_processing.common import get_node_label_to_edge_types, get_op_label_indices
from data_utils.mathqa_graph_generation_data import create_node_cross_entropy_target, create_edge_cross_entropy_target
from train.mathqa_graph_generation_train import is_exact_match


class MyTestCase(unittest.TestCase):
    def test_one_pad(self):
        x = torch.randn(10, 11, 12)
        x_pad = one_pad(x, 1)
        self.assertEqual(x_pad.shape, (10, 12, 12))
        self.assertTrue(torch.all(x_pad[:, -1, :] == 1))

    def test_attention_pool(self):
        kd = 100
        vd = 111
        n = 100

        ap = AttentionPool(kd)
        k = torch.randn(n, kd)
        v = torch.randn(n, vd)
        s = ap(k, v)
        self.assertEqual(s.shape, (vd, ))

    def test_attention_pool_overfit(self):
        dim = 100
        n = 32
        keys = torch.randn(n, dim)
        values = torch.randn(n, dim)
        target_w = torch.randn(dim)
        target_agg = torch.matmul(keys, target_w)
        target_agg = torch.softmax(target_agg, dim=0)
        target_agg = torch.matmul(target_agg, values)

        model = AttentionPool(dim)
        optimizer = torch.optim.Adam(model.parameters())

        def closure():
            optimizer.zero_grad()
            result = model(keys, values)
            loss = F.mse_loss(result, target_agg)
            loss.backward()
            return loss.item()

        for i in range(1000):
            optimizer.step(closure)

        self.assertLess(closure(), 1e-3)

    def test_general_attention(self):
        key_dim = 128
        query_dim = 57
        n_keys = 20
        n_queries = 12

        keys = torch.randn(n_keys, key_dim)
        queries = torch.randn(n_queries, query_dim)
        model = GeneralAttention(key_dim, query_dim)
        out = model(keys, queries)
        self.assertEqual(out.shape, (n_keys, n_queries))

        queries = torch.randn(query_dim)
        out = model(keys, queries)
        self.assertEqual(out.shape, (n_keys, ))

    def test_ggnn(self):
        dim = 128
        n_edge_types = 5
        model = GGNN(dim, n_edge_types)

        n_nodes = 30
        n_prop_steps = 10
        adj_tensor = torch.randint(2, size=(n_edge_types, n_nodes, n_nodes),
                                   dtype=torch.float32)
        node_embedding = torch.randn(n_nodes, dim)
        out = model(adj_tensor, node_embedding, n_prop_steps)
        self.assertEqual(out.shape, node_embedding.shape)

    def test_node_embedding(self):
        n_labels = 10
        dim = 128
        n_edge_type = 10
        model = NodeEmbedding(n_labels, dim, n_edge_type)

        n_nodes = 66
        adj_tensor = torch.randn(n_edge_type, n_nodes, n_nodes)
        labels = torch.randint(n_labels, (n_nodes, ), dtype=torch.long)
        n_prop_steps = 7
        out = model(adj_tensor, labels, n_prop_steps)
        self.assertEqual(out.shape, (n_nodes, dim))


class TestGraphGenerator(unittest.TestCase):
    def init_node_labels_to_edge_types(self) -> dict[int, list[int]]:
        node_labels_to_edge_types = {}
        for node_label in range(self.n_node_labels):
            if node_label in self.node_labels_not_to_generate:
                node_labels_to_edge_types[node_label] = []
            else:
                node_labels_to_edge_types[node_label] = list(range(self.n_edge_types))

        return node_labels_to_edge_types

    def init_inputs(self) -> tuple[Tensor, Tensor]:
        adj_tensor = torch.zeros(self.n_edge_types, self.n_nodes, self.n_nodes)
        node_labels = torch.empty(self.n_nodes, dtype=torch.long)

        for node_i in range(self.n_nodes):
            if node_i < self.start_node_i:
                node_label = self.node_labels_not_to_generate[node_i % len(self.node_labels_not_to_generate)]
            else:
                node_label = self.node_labels_to_generate[node_i % len(self.node_labels_to_generate)]
            node_labels[node_i] = node_label
            for edge_type in self.node_labels_to_edge_types[node_label]:
                src_node_i = max(0, node_i - edge_type - 1)
                adj_tensor[edge_type, src_node_i, node_i] += 1

        return adj_tensor, node_labels

    def init_targets(self) -> tuple[Tensor, Tensor]:
        edge_target = create_edge_cross_entropy_target(
            adj_tensor=self.adj_tensor,
            start_node_i=self.start_node_i
        )
        node_target = create_node_cross_entropy_target(
            node_labels=self.node_labels,
            stop_token=len(self.node_labels_to_generate),
            start_node_i=self.start_node_i,
            real_label_to_generated_label=self.real_label_to_generated_label,
        )
        return edge_target, node_target

    def to_cuda(self):
        device = torch.device('cuda')
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, (Tensor, nn.Module)):
                setattr(self, attr_name, attr.to(device))

    def setUp(self) -> None:
        self.n_node_labels = 50
        self.n_edge_types = 5
        self.node_embedding_dim = 64
        self.condition_dim = 64
        self.node_labels_to_generate = list(range(self.n_node_labels // 2, self.n_node_labels))
        self.node_labels_not_to_generate = list(set(range(self.n_node_labels)).difference(self.node_labels_to_generate))
        self.node_labels_to_edge_types = self.init_node_labels_to_edge_types()
        self.graph_generator = GraphGenerator(
            n_node_labels=self.n_node_labels,
            n_edge_types=self.n_edge_types,
            node_embedding_dim=self.node_embedding_dim,
            condition_dim=self.condition_dim,
            node_label_to_edge_types=self.node_labels_to_edge_types,
            node_labels_to_generate=self.node_labels_to_generate,
        )
        self.n_nodes = 50
        self.start_node_i = self.n_nodes // 2
        self.adj_tensor, self.node_labels = self.init_inputs()
        self.n_prop_steps = 5
        self.condition_vector = torch.full((self.condition_dim,), 0.1)
        self.real_label_to_generated_label = {label: i for i, label in enumerate(self.node_labels_to_generate)}
        self.edge_target, self.node_target = self.init_targets()

    def run_model(self) -> GraphGeneratorTrainOutput:
        return self.graph_generator(
            adj_tensor=self.adj_tensor,
            node_labels=self.node_labels,
            start_node_i=self.start_node_i,
            n_prop_steps=self.n_prop_steps,
            condition_vector=self.condition_vector,
        )

    def test_output_shapes(self):
        output = self.run_model()
        self.assertEqual(output.edge_logits.shape, self.edge_target.shape + (self.n_nodes,))
        self.assertEqual(output.node_logits.shape, self.node_target.shape + (len(self.node_labels_to_generate) + 1,))

    def test_cuda(self):
        self.to_cuda()
        self.test_output_shapes()

    def check_exact_match(self):
        partial_adj_tensor = self.adj_tensor[:, :self.start_node_i, :self.start_node_i]
        partial_node_labels = self.node_labels[:self.start_node_i]
        generated_adj_tensor, generated_node_labels = self.graph_generator.generate(
            partial_adj_tensor=partial_adj_tensor,
            partial_node_labels=partial_node_labels,
            condition_vector=self.condition_vector,
            n_prop_steps=self.n_prop_steps,
        )
        min_len = min(len(generated_node_labels), len(self.node_labels))
        n_correct = 0
        n_total = 0
        for i in range(self.start_node_i, min_len):
            n_total += 1
            if generated_node_labels[i] == self.node_labels[i]:
                n_correct += 1
        acc = n_correct / n_total
        print(f'node_accuracy={acc}')
        match = is_exact_match(generated_adj_tensor, generated_node_labels, self.adj_tensor, self.node_labels)
        self.assertTrue(match)

    def test_overfit(self):
        optimizer = torch.optim.Adam(self.graph_generator.parameters())
        n_epochs = 1000
        best_loss = math.inf
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.run_model()
            loss = graph_generator_loss_fn(output, self.edge_target, self.node_target)
            loss.backward()
            optimizer.step()
            print(f'{epoch+1} out of {n_epochs} loss={loss.item()}')
            if best_loss > loss.item():
                best_loss = loss.item()

        self.assertLess(best_loss, 0.5)
        self.check_exact_match()


if __name__ == '__main__':
    unittest.main()
