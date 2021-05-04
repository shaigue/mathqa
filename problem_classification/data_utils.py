import torch
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, DataLoader
from math_qa import math_qa


class ProblemClassificationDataset(Dataset):
    def __init__(self, partition: str):
        self.problems = []
        self.category = []

        entries = math_qa.load_dataset(partition)
        for e in entries:
            self.problems.append(e.problem)
            self.category.append(e.category)

        self.int_to_category = math_qa.get_categories()
        self.category_to_int = {cat: i for i, cat in enumerate(self.int_to_category)}
        self.n_categories = len(self.int_to_category)

        self.category_num = [self.category_to_int[cat] for cat in self.category]

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, item) -> tuple[str, int]:
        return self.problems[item], self.category_num[item]


def collate_fn(batch: list[tuple[str, int]]) -> tuple[list[str], Tensor]:
    sentences = []
    categories = []
    for s, c in batch:
        sentences.append(s)
        categories.append(c)
    return sentences, torch.tensor(categories)


def get_data_loaders() -> dict[str, DataLoader]:
    d = {}
    for part in ['train', 'dev', 'test']:
        dataset = ProblemClassificationDataset(part)
        if part == 'train':
            shuffle = True
        else:
            shuffle = False
        d[part] = DataLoader(dataset, batch_size=2, shuffle=shuffle, collate_fn=collate_fn)
    return d



