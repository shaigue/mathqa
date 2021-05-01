import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from graph_classification.common import get_n_node_labels, get_max_n_args, get_n_categories
from graph_classification.data_utils import GraphClassificationDataset
from graph_classification.model import GraphClassifier


def train_eval(gnn: bool, gnn_bias: bool, gnn_n_iter: int, logger):
    # this simple loop gets to ~84% accuracy on dev
    desc = f"gnn={gnn}-bias={gnn_bias}-T={gnn_n_iter}"
    train_set = GraphClassificationDataset('train')
    dev_set = GraphClassificationDataset('dev')

    model = GraphClassifier(n_node_labels=get_n_node_labels(),
                            n_edge_types=get_max_n_args(),
                            hidden_dim=32,
                            n_classes=get_n_categories(),
                            gnn=gnn,
                            gnn_bias=gnn_bias,
                            gnn_n_iter=gnn_n_iter,)
    clip_grad_norm = 2
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4)

    train_samples = len(train_set)
    dev_samples = len(dev_set)

    n_epochs = 150
    batch_size = 16
    best_acc = 0
    t = 0

    for epoch in range(n_epochs):
        print(f"epoch={epoch}")
        class_scores, class_true = [], []
        train_indices = np.arange(train_samples)
        np.random.shuffle(train_indices)
        # train epoch
        for i in range(train_samples):
            adj_tensor, node_labels, category = train_set[train_indices[i]]
            class_scores.append(model(adj_tensor, node_labels))
            class_true.append(category)

            if i > 0 and (i % batch_size == 0 or i == train_samples - 1):
                class_scores = torch.stack(class_scores)
                class_true = torch.tensor(class_true, dtype=torch.long)
                loss = F.cross_entropy(class_scores, class_true)
                loss.backward()
                logger.add_scalar(f'loss/{desc}', loss.item(), t)
                t += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                class_scores, class_true = [], []

        # eval epoch
        correct = 0
        for i in range(dev_samples):
            adj_tensor, node_labels, category = train_set[i]
            class_scores = model(adj_tensor, node_labels)
            pred_category = torch.argmax(class_scores)
            if pred_category == category:
                correct += 1

        acc = correct / dev_samples
        scheduler.step(acc)
        if acc > best_acc:
            best_acc = acc
        logger.add_scalar(f'accuracy/{desc}', acc, epoch)

    print(best_acc)


if __name__ == "__main__":
    logger = SummaryWriter()
    train_eval(gnn=False, gnn_bias=False, gnn_n_iter=1, logger=logger)
    train_eval(gnn=True, gnn_bias=False, gnn_n_iter=1, logger=logger)
    train_eval(gnn=True, gnn_bias=False, gnn_n_iter=5, logger=logger)
    train_eval(gnn=True, gnn_bias=True, gnn_n_iter=5, logger=logger)
