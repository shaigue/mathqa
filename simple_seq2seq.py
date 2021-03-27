import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# TODO: support padding


def get_module_device(module: nn.Module):
    parameter_iterator = module.parameters()
    first_parameter = next(parameter_iterator)
    return first_parameter.device


class Encoder(nn.Module):
    def __init__(self, source_vocabulary_size: int, hidden_dim: int, pad_index: int, dropout=0.0):
        super(Encoder, self).__init__()
        self.source_vocabulary_size = source_vocabulary_size
        self.hidden_dim = hidden_dim
        self.pad_index = pad_index

        self.embedding_layer = nn.Embedding(num_embeddings=source_vocabulary_size, embedding_dim=hidden_dim,
                                            padding_idx=pad_index)
        self.gru_layer = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, dropout=dropout)

    def forward(self, source_tokens, source_lens=None, prev_hidden_state=None):
        seq_len, batch_size = source_tokens.shape
        assert source_lens is not None or batch_size == 1, "source_lens can be None only if batch_size is 1"

        if prev_hidden_state is None:
            prev_hidden_state = self._initial_hidden_state(batch_size)

        embedding = self.embedding_layer(source_tokens)
        if source_lens is not None:
            embedding = pack_padded_sequence(embedding, source_lens, enforce_sorted=False)
        gru_outputs, next_hidden_state = self.gru_layer(embedding, prev_hidden_state)
        # This is currently not used, so skip this for speed
        # gru_outputs = pad_packed_sequence(gru_outputs, padding_value=self.pad_index)
        return gru_outputs, next_hidden_state

    def _initial_hidden_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim, device=get_module_device(self))


class Decoder(nn.Module):
    def __init__(self, target_vocabulary_size: int, hidden_dim: int, dropout=0.0, pad_index: int = None):
        super(Decoder, self).__init__()
        self.target_vocabulary_size = target_vocabulary_size
        self.hidden_dim = hidden_dim
        self.pad_index = pad_index

        self.embedding_layer = nn.Embedding(num_embeddings=target_vocabulary_size, embedding_dim=hidden_dim,
                                            padding_idx=pad_index)
        self.gru_layer = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, dropout=dropout)
        self.output_linear_layer = nn.Linear(in_features=hidden_dim, out_features=target_vocabulary_size)

    def forward(self, target_tokens, encoder_last_hidden, target_lens=None):
        seq_len, batch_size = target_tokens.shape
        assert target_lens is not None or batch_size == 1, "if target_len is None, then batch_size has to be 1."

        embedding = self.embedding_layer(target_tokens)

        if target_lens is not None:
            embedding = pack_padded_sequence(embedding, target_lens, enforce_sorted=False)
        gru_outputs, next_hidden_state = self.gru_layer(embedding, encoder_last_hidden)
        if target_lens is not None:
            gru_outputs, _ = pad_packed_sequence(gru_outputs, padding_value=self.pad_index)

        gru_outputs = gru_outputs.view(seq_len * batch_size, self.hidden_dim)
        token_logits = self.output_linear_layer(gru_outputs)
        token_logits = token_logits.view(seq_len, batch_size, self.target_vocabulary_size)

        return token_logits, next_hidden_state


class Seq2Seq(nn.Module):
    def __init__(self, source_vocabulary_size: int, target_vocabulary_size: int, hidden_dim: int,
                 pad_index: int, dropout=0.0):
        super(Seq2Seq, self).__init__()
        self.source_vocabulary_size = source_vocabulary_size
        self.target_vocabulary_size = target_vocabulary_size
        self.hidden_dim = hidden_dim
        self.pad_index = pad_index

        self.encoder = Encoder(source_vocabulary_size, hidden_dim, dropout=dropout,
                               pad_index=pad_index)
        self.decoder = Decoder(target_vocabulary_size, hidden_dim, dropout=dropout,
                               pad_index=pad_index)

    def forward(self, source_tokens, target_tokens, source_lens, target_lens):
        encoder_output, encoder_last_hidden_state = self.encoder(source_tokens, source_lens)
        decoder_output, decoder_last_hidden_state = self.decoder(target_tokens, encoder_last_hidden_state, target_lens)
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


def get_simple_seq2seq():
    # TODO: implement
    return Seq2Seq(
        source_vocabulary_size=0,
        target_vocabulary_size=0,
        hidden_dim=0,
        dropout=0,
        pad_index=0,
    )


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