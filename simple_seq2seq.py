import torch
from torch import nn
from torch.nn import functional as F
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
        self.source_vocab_size = source_vocabulary_size
        self.target_vocab_size = target_vocabulary_size
        self.hidden_dim = hidden_dim
        self.pad_index = pad_index
        self.n_gru_layers = n_gru_layers

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

    def _beam_search_decode(self, source_tokens, source_lens, start_of_sequence_token: int, end_of_sequence_token: int,
                            max_target_seq_len: int, beam_size: int = 1):
        # TODO: support batched beam search decoding
        pass

    def _greedy_decode(self, source_tokens, source_lens, start_of_sequence_token: int, end_of_sequence_token: int,
                       max_target_seq_len: int):
        seq_len, batch_size = source_tokens.shape
        assert len(source_lens) == batch_size

        encoder_out, encoder_hidden = self.encoder(source_tokens, source_lens)
        # assert encoder_out.shape == (seq_len, batch_size, self.hidden_dim)
        assert encoder_hidden.shape == (self.n_gru_layers, batch_size, self.hidden_dim)

        decoded = torch.full(size=(batch_size, max_target_seq_len), fill_value=self.pad_index,
                             device=get_module_device(self))

        # this tells us what batch corresponds to each entry in decoded.
        # will change when <EOS> will be emitted
        batch_mapping = torch.arange(batch_size)
        decoder_hidden = encoder_hidden
        decoder_input = torch.full(size=(1, batch_size), fill_value=start_of_sequence_token,
                                   device=get_module_device(self))
        n_active = batch_size
        for i in range(max_target_seq_len):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            assert decoder_out.shape == (1, n_active, self.target_vocab_size)
            assert decoder_hidden.shape == (self.n_gru_layers, n_active, self.hidden_dim)

            # insert the next tokens to the decoded list
            decoder_out = decoder_out.view(n_active, self.target_vocab_size)
            next_tokens = torch.argmax(decoder_out, dim=1)
            decoded[batch_mapping, i] = next_tokens

            # find out what tokens are end-of-sequence
            end_of_sequence_mask = (next_tokens != end_of_sequence_token)
            n_active = end_of_sequence_mask.sum().item()
            if n_active == 0:
                break
            batch_mapping = batch_mapping[end_of_sequence_mask]
            decoder_hidden = decoder_hidden[:, end_of_sequence_mask]
            decoder_input = next_tokens[end_of_sequence_mask].view(1, -1)

        # convert to a list of lists
        decoded_list = []
        for i in range(batch_size):
            mask = decoded[i] != self.pad_index
            decoded_list.append(decoded[i, mask].tolist())

        return decoded_list

    def generate(self, source_tokens, source_lens, start_of_sequence_token: int, end_of_sequence_token: int,
                 max_target_seq_len: int, beam_size: int = 1):
        if beam_size == 1:
            return self._greedy_decode(source_tokens, source_lens, start_of_sequence_token, end_of_sequence_token,
                                       max_target_seq_len)
        else:
            return self._beam_search_decode(source_tokens, source_lens, start_of_sequence_token, end_of_sequence_token,
                                            max_target_seq_len, beam_size)


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
    generated = model.generate(
        padded_source, source_lens,
        start_of_sequence_token=1,
        end_of_sequence_token=2,
        max_target_seq_len=max_seq_len,
        beam_size=1
    )
    print(f"generated sequence\n{generated}")


if __name__ == "__main__":
    example()