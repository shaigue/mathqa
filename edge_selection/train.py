import numpy as np
import torch
from torch import optim, Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from edge_selection.data_utils import ArgSelectionDataset, ArgSelectionDatapoint
from edge_selection.model import EdgeSelectionModel
from graph_classification.common import get_max_n_args, get_n_node_labels


def check_correct(logits: Tensor, gt: int) -> int:
    pred = logits.argmax()
    return int(pred == gt)


def train_eval(logger):
    train_set = ArgSelectionDataset('train')
    dev_set = ArgSelectionDataset('dev')

    model = EdgeSelectionModel(get_n_node_labels(), 32, get_max_n_args())
    clip_grad_norm = 100
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8)

    train_samples = len(train_set)
    dev_samples = len(dev_set)

    n_epochs = 150
    batch_size = 16
    best_acc = 0
    t = 0

    for epoch in range(n_epochs):
        print(f"epoch={epoch}")
        loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        train_indices = np.arange(train_samples)
        np.random.shuffle(train_indices)
        # train epoch
        correct = 0
        for i in range(train_samples):
            adj_tensor, node_labels, dst_node, src_node = train_set[train_indices[i]]
            logits = model(adj_tensor, node_labels, dst_node)
            loss = loss + F.cross_entropy(logits.unsqueeze(0), torch.tensor([src_node]))
            correct += check_correct(logits, src_node)

            if i > 0 and (i % batch_size == 0 or i == train_samples - 1):
                loss.backward()
                logger.add_scalar(f'train/loss', loss.item(), t)
                t += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)

        acc = correct / train_samples
        logger.add_scalar('train/accuracy', acc, epoch)

        # eval epoch
        correct = 0
        for i in range(dev_samples):
            adj_tensor, node_labels, dst_node, src_node = train_set[i]
            logits = model(adj_tensor, node_labels, dst_node)
            correct += check_correct(logits, src_node)

        acc = correct / dev_samples
        scheduler.step(acc)
        if acc > best_acc:
            best_acc = acc
        logger.add_scalar(f'dev/accuracy', acc, epoch)

    print(best_acc)


if __name__ == "__main__":
    logger = SummaryWriter()
    train_eval(logger)