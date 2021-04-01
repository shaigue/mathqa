import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def get_module_device(module: nn.Module):
    parameter_iterator = module.parameters()
    first_parameter = next(parameter_iterator)
    return first_parameter.device


class Encoder(nn.Module):
    def __init__(self, source_vocabulary_size: int, hidden_dim: int, pad_index: int,
                 dropout: float = 0.0, n_gru_layers: int = 1):
        super(Encoder, self).__init__()
        self.source_vocabulary_size = source_vocabulary_size
        self.hidden_dim = hidden_dim
        self.pad_index = pad_index

        self.n_gru_layers = n_gru_layers

        # dropout in RNN is only possible when more then one layer
        dropout = dropout if n_gru_layers > 1 else 0

        self.embedding_layer = nn.Embedding(num_embeddings=source_vocabulary_size, embedding_dim=hidden_dim,
                                            padding_idx=pad_index)
        self.gru_layer = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, dropout=dropout,
                                num_layers=n_gru_layers)

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
        return torch.zeros(self.n_gru_layers, batch_size, self.hidden_dim, device=get_module_device(self))


class Decoder(nn.Module):
    def __init__(self, target_vocabulary_size: int, hidden_dim: int, pad_index: int = None,
                 dropout: float = 0, n_gru_layers: int = 1):
        super(Decoder, self).__init__()
        self.target_vocabulary_size = target_vocabulary_size
        self.hidden_dim = hidden_dim
        self.pad_index = pad_index

        self.n_gru_layers = n_gru_layers

        gru_dropout = dropout if n_gru_layers > 1 else 0
        self.embedding_layer = nn.Embedding(num_embeddings=target_vocabulary_size, embedding_dim=hidden_dim,
                                            padding_idx=pad_index)
        self.gru_layer = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, dropout=gru_dropout,
                                num_layers=n_gru_layers)
        self.dropout = nn.Dropout(dropout)
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
        gru_outputs = self.dropout(gru_outputs)
        token_logits = self.output_linear_layer(gru_outputs)
        token_logits = token_logits.view(seq_len, batch_size, self.target_vocabulary_size)

        return token_logits, next_hidden_state


class Seq2Seq(nn.Module):
    def __init__(self, source_vocabulary_size: int, target_vocabulary_size: int, hidden_dim: int,
                 pad_index: int, dropout: float = 0.0, n_gru_layers: int = 1):
        super(Seq2Seq, self).__init__()
        self.source_vocabulary_size = source_vocabulary_size
        self.target_vocabulary_size = target_vocabulary_size
        self.hidden_dim = hidden_dim
        self.pad_index = pad_index

        self.encoder = Encoder(source_vocabulary_size, hidden_dim, dropout=dropout,
                               pad_index=pad_index, n_gru_layers=n_gru_layers)
        self.dropout = nn.Dropout(dropout)
        self.decoder = Decoder(target_vocabulary_size, hidden_dim, dropout=dropout,
                               pad_index=pad_index, n_gru_layers=n_gru_layers)

    def forward(self, source_tokens, target_tokens, source_lens, target_lens):
        encoder_output, encoder_last_hidden_state = self.encoder(source_tokens, source_lens)
        encoder_last_hidden_state = self.dropout(encoder_last_hidden_state)
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


def example():
    # define parameters
    source_vocab_size = 100
    target_vocab_size = 60
    hidden_dim = 64
    pad_index = 0
    dropout = 0.2
    n_gru_layers = 2

    # crete some sequences
    n_sequences = 32
    max_seq_len = 50
    min_seq_len = 30
    source_sequences = []
    source_lens = torch.empty(n_sequences, dtype=torch.int64)
    target_sequences = []
    target_lens = torch.empty(n_sequences, dtype=torch.int64)
    import random
    for i in range(n_sequences):
        source_seq_len = random.randint(min_seq_len, max_seq_len)
        source_lens[i] = source_seq_len
        source_seq = torch.randint(low=1, high=source_vocab_size, size=(source_seq_len, ))
        source_sequences.append(source_seq)

        target_seq_len = random.randint(min_seq_len, max_seq_len)
        target_lens[i] = target_seq_len
        target_seq = torch.randint(low=1, high=target_vocab_size, size=(target_seq_len, ))
        target_sequences.append(target_seq)

    # pad the sequences before inputting to the model
    padded_source = pad_sequence(source_sequences, padding_value=pad_index)
    padded_targets = pad_sequence(target_sequences, padding_value=pad_index)

    # if you use a loss function take count that you need to ignore the pad symbol
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)

    # create the model
    model = Seq2Seq(
        source_vocabulary_size=source_vocab_size,
        target_vocabulary_size=target_vocab_size,
        hidden_dim=hidden_dim,
        pad_index=pad_index,
        dropout=dropout,
        n_gru_layers=n_gru_layers
    )

    # simple forward pass of the model
    output = model(padded_source, padded_targets, source_lens, target_lens)

    # generate text with the model
    source_to_generate = source_sequences[3].view(-1, 1)
    generated = model.generate(
        source_token_indices=source_to_generate,
        start_of_string_token_index=1,
        end_of_string_token_index=2,
        max_target_seq_len=max_seq_len
    )

    print(f"generated sequence\n{generated}")


if __name__ == "__main__":
    example()