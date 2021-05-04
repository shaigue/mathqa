import torch
from torch import nn, Tensor
from torch.nn import functional
from transformers import AlbertTokenizer, AlbertModel, AutoModel, AutoTokenizer

from models.common import get_device


class TextClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super(TextClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained('albert-base-v1')
        self.tokenizer = AutoTokenizer.from_pretrained('albert-base-v1')
        self.classifier = nn.Linear(self.transformer.config.hidden_size, n_classes)

    def forward(self, text: list[str]) -> Tensor:
        x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True,
                           max_length=self.transformer.config.max_position_embeddings)
        x = x.to(get_device(self))
        x = self.transformer(**x)
        x = self.classifier(x.pooler_output)
        return x


if __name__ == "__main__":
    x = 1000
    model = TextClassifier(10)
    x = [' '.join('hello' for i in range(x))]
    y = model(x)
    print(y)
