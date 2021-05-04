import torch
from torch.nn import functional

from graph_classification.common import get_n_categories
from problem_classification.model import TextClassifier
from problem_classification.data_utils import ProblemClassificationDataset, get_data_loaders
from torch import optim
from torch.utils.tensorboard import SummaryWriter


def get_n_correct(logits, targets) -> float:
    return torch.sum(logits.argmax(dim=1) == targets).item()


def train_eval():
    logger = SummaryWriter()
    data_loaders = get_data_loaders()

    train_loader = data_loaders['train']
    n_train = len(train_loader.dataset)

    eval_loader = data_loaders['dev']
    n_eval = len(eval_loader.dataset)

    model = TextClassifier(get_n_categories())
    optimizer = optim.Adam([{'params': model.transformer.parameters(), 'lr': 1e-5},
                            {'params': model.classifier.parameters(), 'lr': 1e-3}])
    n_epochs = 100
    t = 0
    device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(n_epochs):
        print(f"epoch={epoch}")
        model.train()

        n_correct = 0
        for text, cat in train_loader:
            print(f"n_batch={t}")
            cat = cat.to(device)
            outputs = model(text)
            loss = functional.cross_entropy(outputs, cat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.add_scalar('train/loss', loss.item(), t)
            t += 1

            n_correct += get_n_correct(outputs, cat)

        logger.add_scalar('train/accuracy', n_correct / n_train, epoch)

        model.eval()
        n_correct = 0
        for text, cat in eval_loader:
            cat = cat.to(device)
            outputs = model(text)
            n_correct += get_n_correct(outputs, cat)

        logger.add_scalar('eval/accuracy', n_correct / n_eval, epoch)


if __name__ == "__main__":
    train_eval()
