from torch import nn, Tensor

from models.common import get_module_device
from models.pretrained_lm import get_pretrained_albert


class TextClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super(TextClassifier, self).__init__()
        self.transformer, self.tokenizer = get_pretrained_albert()
        self.classifier = nn.Linear(self.transformer.config.hidden_size, n_classes)

    def forward(self, text: list[str]) -> Tensor:
        x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True,
                           max_length=self.transformer.config.max_position_embeddings)
        x = x.to(get_module_device(self))
        x = self.transformer(**x)
        x = self.classifier(x.pooler_output)
        return x


if __name__ == "__main__":
    x = 1000
    model = TextClassifier(10)
    x = [' '.join('hello' for i in range(x))]
    y = model(x)
    print(y)
