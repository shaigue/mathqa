import torch
from torch import nn

# TODO: support padding


def get_module_device(module: nn.Module):
    parameter_iterator = module.parameters()
    first_parameter = next(parameter_iterator)
    return first_parameter.device


class Encoder(nn.Module):
    def __init__(self, source_vocabulary_size: int, hidden_dim: int):
        super(Encoder, self).__init__()
        self.source_vocabulary_size = source_vocabulary_size
        self.hidden_dim = hidden_dim

        self.embedding_layer = nn.Embedding(num_embeddings=source_vocabulary_size, embedding_dim=hidden_dim)
        self.gru_layer = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)

    def forward(self, source_token_indices, prev_hidden_state=None):
        seq_len, batch_size = source_token_indices.shape

        if prev_hidden_state is None:
            prev_hidden_state = self._initial_hidden_state(batch_size)

        embedding = self.embedding_layer(source_token_indices)
        gru_outputs, next_hidden_state = self.gru_layer(embedding, prev_hidden_state)
        return gru_outputs, next_hidden_state

    def _initial_hidden_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim, device=get_module_device(self))


class Decoder(nn.Module):
    def __init__(self, target_vocabulary_size: int, hidden_dim: int):
        super(Decoder, self).__init__()
        self.target_vocabulary_size = target_vocabulary_size
        self.hidden_dim = hidden_dim

        self.embedding_layer = nn.Embedding(num_embeddings=target_vocabulary_size, embedding_dim=hidden_dim)
        self.gru_layer = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)
        self.output_linear_layer = nn.Linear(in_features=hidden_dim, out_features=target_vocabulary_size)

    def forward(self, target_token_indices, encoder_last_hidden):
        seq_len, batch_size = target_token_indices.shape

        embedding = self.embedding_layer(target_token_indices)
        gru_outputs, next_hidden_state = self.gru_layer(embedding, encoder_last_hidden)
        gru_outputs = gru_outputs.view(seq_len * batch_size, self.hidden_dim)
        token_logits = self.output_linear_layer(gru_outputs)
        token_logits = token_logits.view(seq_len, batch_size, self.target_vocabulary_size)
        return token_logits, next_hidden_state


class Seq2Seq(nn.Module):
    def __init__(self, source_vocabulary_size: int, target_vocabulary_size: int, hidden_dim: int):
        super(Seq2Seq, self).__init__()
        self.source_vocabulary_size = source_vocabulary_size
        self.target_vocabulary_size = target_vocabulary_size
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(source_vocabulary_size, hidden_dim)
        self.decoder = Decoder(target_vocabulary_size, hidden_dim)

    def forward(self, source_token_indices, target_token_indices):
        encoder_output, encoder_last_hidden_state = self.encoder(source_token_indices)
        decoder_output, decoder_last_hidden_state = self.decoder(target_token_indices, encoder_last_hidden_state)
        return decoder_output

    def generate(self, source_token_indices, start_of_string_token_index: int, end_of_string_token_index: int,
                 max_target_seq_len: int):
        seq_len, batch_size = source_token_indices.shape
        # TODO: think how to remove this by supporting masking
        assert batch_size == 1, f"on generation input sequences should be 1, got {batch_size}"

        encoder_output, encoder_last_hidden_state = self.encoder(source_token_indices)
        decoder_last_hidden_state = encoder_last_hidden_state
        next_decoder_input = torch.tensor(start_of_string_token_index, device=get_module_device(self)).view(1, 1)
        generated = []

        for i in range(max_target_seq_len):
            # decoder_output.shape = (1, 1, target_vocabulary_size)
            decoder_output, decoder_last_hidden_state = self.decoder(next_decoder_input, decoder_last_hidden_state)
            decoder_output = decoder_output.view(self.target_vocabulary_size)
            next_token = torch.argmax(decoder_output)
            if next_token == end_of_string_token_index:
                break
            # will not add the end-of-sequence and start-of-sequence tokens to the string
            generated.append(next_token.detach().cpu().item())
            next_decoder_input = next_token.view(1, 1)

        return generated


def example():
    batch_size = 1
    hidden_dim = 32

    source_vocabulary_size = 123
    source_seq_len = 22
    source_token_indices = torch.randint(high=source_vocabulary_size, size=(source_seq_len, batch_size))

    target_vocabulary_size = 321
    target_seq_len = 11
    target_token_indices = torch.randint(high=target_vocabulary_size, size=(target_seq_len, batch_size))

    seq2seq = Seq2Seq(source_vocabulary_size, target_vocabulary_size, hidden_dim)
    output = seq2seq(source_token_indices, target_token_indices)
    print(output.shape)
    print(output.shape == (target_seq_len, batch_size, target_vocabulary_size))


def example_generate():
    batch_size = 1
    hidden_dim = 32

    source_vocabulary_size = 123
    source_seq_len = 22
    source_token_indices = torch.randint(high=source_vocabulary_size, size=(source_seq_len, batch_size))

    target_vocabulary_size = 321
    target_seq_len = 11
    target_token_indices = torch.randint(high=target_vocabulary_size, size=(target_seq_len, batch_size))

    seq2seq = Seq2Seq(source_vocabulary_size, target_vocabulary_size, hidden_dim)
    generated = seq2seq.generate(source_token_indices, 1, 5, 40)
    print(generated)


if __name__ == "__main__":
    # example()
    example_generate()