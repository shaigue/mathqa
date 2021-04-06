from collections import namedtuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from mathqa_processing import MathQAManager, MathQADatapoint

TrainBatch = namedtuple('TrainBatch', ['text_tokens', 'text_lens', 'code_tokens', 'code_lens'])
EvalBatch = namedtuple('EvalBatch', ['text_tokens', 'text_lens', 'inputs', 'answers'])


class MathQADataset(Dataset):
    def __init__(self, manager: MathQAManager, partition: str):
        self.manager = manager
        self.partition = partition

    def __getitem__(self, item: int) -> MathQADatapoint:
        datapoint = self.manager.get_datapoint(self.partition, item)
        return datapoint

    def __len__(self) -> int:
        return self.manager.get_partition_length(self.partition)


class CollateFn:
    def __init__(self, pad_index: int, device):
        self.pad_index = pad_index
        self.device = device

    def _pad_to_device(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(sequences, padding_value=self.pad_index).to(self.device)

    def train_collate(self, batch: list[MathQADatapoint]) -> EvalBatch:
        texts = []
        texts_len = []
        codes = []
        codes_len = []
        for entry in batch:
            text = torch.tensor(entry.text_token_indices)
            texts.append(text)
            texts_len.append(text.shape[0])

            code = torch.tensor(entry.code_token_indices)
            codes.append(code)
            codes_len.append(code.shape[0])
        # transfer them to the device
        texts = self._pad_to_device(texts)
        codes = self._pad_to_device(codes)

        return TrainBatch(
            text_tokens=texts,
            text_lens=texts_len,
            code_tokens=codes,
            code_lens=codes_len,
        )

    def eval_collate(self, batch: list[MathQADatapoint]) -> EvalBatch:
        text_tokens, text_lens, inputs, answers = [], [], [], []

        for entry in batch:
            text = torch.tensor(entry.text_token_indices)
            text_tokens.append(text)
            text_lens.append(text.shape[0])
            inputs.append(entry.extracted_numbers)
            answers.append(entry.evaluated_result)

        text_tokens = self._pad_to_device(text_tokens)

        return EvalBatch(
            text_tokens=text_tokens,
            text_lens=text_lens,
            inputs=inputs,
            answers=answers,
        )


def get_loader(manager: MathQAManager, device, partition: str, train: bool,
               batch_size: int = 32) -> DataLoader:
    # TODO: support also multiple partitions
    dataset = MathQADataset(manager, partition)
    collate_fn = CollateFn(manager.pad_index, device)
    if train:
        return DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn.train_collate)
    return DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn.eval_collate)
