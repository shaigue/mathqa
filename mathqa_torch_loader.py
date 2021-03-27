import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from mathqa_processing import MathQAManager


class MathQADataset(Dataset):
    def __init__(self, manager: MathQAManager, partition: str):
        self.manager = manager
        self.partition = partition

    def __getitem__(self, item: int):
        datapoint = self.manager.get_datapoint(self.partition, item)
        return datapoint.text_token_indices, datapoint.code_token_indices

    def __len__(self):
        return self.manager.get_partition_length(self.partition)


class PadCollateFn:
    def __init__(self, pad_index: int, device):
        self.pad_index = pad_index
        self.device = device

    def __call__(self, batch: list[tuple[list[int], list[int]]]):
        texts = []
        texts_len = []
        codes = []
        codes_len = []
        for text, code in batch:
            text = torch.tensor(text)
            code = torch.tensor(code)
            texts.append(text)
            texts_len.append(len(text))
            codes.append(code)
            codes_len.append(len(code))
        # transfer them to the device
        texts = pad_sequence(texts, padding_value=self.pad_index).to(self.device)
        codes = pad_sequence(codes, padding_value=self.pad_index).to(self.device)
        # texts_len = torch.tensor(texts_len)
        # codes_len = torch.tensor(codes_len, device=self.device)

        return texts, codes, texts_len, codes_len


def get_train_loader(manager: MathQAManager, device, batch_size: int = 32) -> DataLoader:
    dataset = MathQADataset(manager, 'train')
    collate_fn = PadCollateFn(manager.pad_index, device)
    dataloader = DataLoader(dataset, batch_size, True, collate_fn=collate_fn)
    return dataloader


def example():
    import config
    from torch import nn
    manager = MathQAManager(config.MATHQA_DIR, max_vocabulary_size=100, dummy=True, no_punctuation=True)
    dataloader = get_train_loader(manager, 'cpu', batch_size=32)
    embedding = nn.Embedding(num_embeddings=manager.text_vocabulary_size, embedding_dim=42,
                             padding_idx=manager.pad_index)
    rnn = nn.GRU(42, 42)
    linear = nn.Linear(42, 10)
    loss_fn = nn.CrossEntropyLoss(ignore_index=manager.pad_index)
    for text_pad, code_pad, text_len, code_len in dataloader:
        text_embedd = embedding(text_pad)
        text_embedd_packed = pack_padded_sequence(text_embedd, text_len, enforce_sorted=False)
        rnn_out, rnn_hidden = rnn(text_embedd_packed)
        pad_out, pad_lens = pad_packed_sequence(rnn_out, padding_value=manager.pad_index)
        seq_len, batch_size, hidden_dim = pad_out.shape
        pad_out = pad_out[pad_lens - 1, torch.arange(batch_size)]#.view(batch_size, hidden_dim)
        # will be S, B, H
        linear_out = linear(pad_out)
        dummy_label = torch.randint(10, (batch_size,))
        loss = loss_fn(linear_out, dummy_label)
        print(loss)


if __name__ == "__main__":
    example()
