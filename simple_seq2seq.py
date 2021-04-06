import torch
from torch import Tensor
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
                            max_target_seq_len: int, beam_size: int):
        # TODO: test & beautify & review & refactor
        # get the input dimensions
        seq_len, batch_size = source_tokens.shape
        assert len(source_lens) == batch_size

        # feed the batch into the encoder
        encoder_out, encoder_hidden = self.encoder(source_tokens, source_lens)
        assert encoder_hidden.shape == (self.n_gru_layers, batch_size, self.hidden_dim)

        # initialize the best decoded sequences
        best_decoded = torch.full(size=(batch_size, max_target_seq_len), fill_value=self.pad_index,
                                  dtype=torch.long, device=get_module_device(self))
        # set their probabilities to 0
        best_probabilities = torch.zeros(size=(batch_size,), dtype=torch.float32, device=get_module_device(self))

        # for simplicity, every iteration we will pass the same size of (n_active, beam_size, *)
        n_active = batch_size
        # this is to map the active samples to the original samples.
        # a samples becomes 'inactive' if all it's running probabilities are not better then it's current best,
        # since the probability only reduces as the sequence gets longer
        active_to_orig_index = torch.arange(batch_size)
        # the decoded sequence for each of the elements in the beam
        decoded = torch.full(size=(n_active, beam_size, max_target_seq_len), fill_value=self.pad_index,
                             dtype=torch.long, device=get_module_device(self))
        # first token will be start of sequence token
        decoded[:, :, 0] = start_of_sequence_token

        # FIRST ITERATION #
        decoder_hidden = encoder_hidden
        # first input is all start_of_sequence
        decoder_inputs = torch.full((1, batch_size), fill_value=start_of_sequence_token,
                                    device=get_module_device(self))
        # feed the decoder
        next_token_logits, decoder_hidden_next = self.decoder(decoder_inputs, decoder_hidden)
        assert next_token_logits.shape == (1, batch_size, self.target_vocab_size)
        assert decoder_hidden_next.shape == (self.n_gru_layers, batch_size, self.hidden_dim)
        # reshape the logits
        next_token_logits = next_token_logits.view(batch_size, self.target_vocab_size)
        # calculate the probabilities of the next tokens
        next_probabilities = F.softmax(next_token_logits, dim=-1)
        # pick the top beam_size with the best probabilities
        probabilities, next_tokens = torch.topk(next_probabilities, beam_size, dim=1)
        assert probabilities.shape == (batch_size, beam_size)
        assert next_tokens.shape == (batch_size, beam_size)

        # place the next tokens in the correct places
        assert decoded.shape == (batch_size, beam_size, max_target_seq_len)
        assert next_tokens.shape == (batch_size, beam_size)
        decoded[:, :, 1] = next_tokens

        # deal with the case where we have <EOS>
        assert next_tokens.shape == (n_active, beam_size)
        assert probabilities.shape == (n_active, beam_size)
        finished_mask = (next_tokens == end_of_sequence_token)
        # take for each of the active sample, take the best ending sequence, if exists, else put there 0
        finished_probabilities = torch.where(finished_mask, probabilities, torch.zeros_like(probabilities))
        best_finished_probabilities, best_finished_indices = torch.max(finished_probabilities, dim=1)

        # compare the best probability for each active sample to the current best ending sequence.
        # if it is better then put this as the best sequence
        assert best_finished_probabilities.shape == (n_active,)
        assert best_probabilities.shape == (batch_size,)
        assert active_to_orig_index.shape == (n_active,)
        assert best_decoded.shape == (batch_size, max_target_seq_len)
        better_then_best_mask = (best_finished_probabilities > best_probabilities[active_to_orig_index])
        update_indices = active_to_orig_index[better_then_best_mask]
        # update the probabilities
        best_probabilities[update_indices] = best_finished_probabilities[better_then_best_mask]
        # update the sequences
        # TODO
        assert decoded.shape == (n_active, beam_size, max_target_seq_len)
        beam_index = best_finished_indices[better_then_best_mask]
        best_decoded[update_indices] = decoded[better_then_best_mask, beam_index]

        # put zero probability where the next token is <EOS>.
        assert probabilities.shape == (n_active, beam_size)
        probabilities = torch.where(finished_mask, torch.zeros_like(probabilities), probabilities)

        # remove the samples where all their probabilities are lower then the best current one from the active set
        best_intermediate_p, _ = torch.max(probabilities, dim=1)
        still_better_then_best_mask = (best_intermediate_p > best_probabilities[active_to_orig_index])
        # now filter out all the tensors that are not active
        n_active = still_better_then_best_mask.sum()
        probabilities = probabilities[still_better_then_best_mask]
        decoder_hidden = decoder_hidden[:, still_better_then_best_mask]
        decoded = decoded[still_better_then_best_mask]
        active_to_orig_index = active_to_orig_index[still_better_then_best_mask]

        # FIRST ITERATION END #

        # duplicate the same output again, for each beam
        decoder_hidden = decoder_hidden_next.view(self.n_gru_layers, n_active, 1, self.hidden_dim)
        decoder_hidden = decoder_hidden.expand(self.n_gru_layers, n_active, beam_size, self.hidden_dim)

        for i in range(1, max_target_seq_len - 1):
            # assert correct input sizes
            assert decoded.shape == (n_active, beam_size, max_target_seq_len)
            assert probabilities.shape == (n_active, beam_size)
            assert decoder_hidden.shape == (self.n_gru_layers, n_active, beam_size, self.hidden_dim)
            assert best_decoded.shape == (batch_size, max_target_seq_len)
            assert best_probabilities.shape == (batch_size,)

            # reshape the inputs before feeding them to the decoder
            decoder_input = decoded[:, :, i].view(1, beam_size * n_active)
            decoder_hidden_flat = decoder_hidden.contiguous().view(self.n_gru_layers, n_active * beam_size,
                                                                   self.hidden_dim)

            # feed the decoder
            next_token_logits, decoder_hidden_next = self.decoder(decoder_input, decoder_hidden_flat)
            assert next_token_logits.shape == (1, n_active * beam_size, self.target_vocab_size)
            assert decoder_hidden_next.shape == (self.n_gru_layers, n_active * beam_size, self.hidden_dim)
            # reshape the output tensors
            decoder_hidden_next = decoder_hidden_next.view(self.n_gru_layers, n_active, beam_size, self.hidden_dim)
            next_token_logits = next_token_logits.view(n_active, beam_size, self.target_vocab_size)

            # calculate the probabilities of the next tokens
            next_token_probabilities = F.softmax(next_token_logits, dim=-1)
            # multiply the current probabilities with the probabilities for the next tokens
            next_probabilities = probabilities.view(n_active, beam_size, 1) * next_token_probabilities
            assert next_probabilities.shape == (n_active, beam_size, self.target_vocab_size)

            # now pick for each active sample the best beam_size options
            next_probabilities = next_probabilities.view(n_active, beam_size * self.target_vocab_size)
            top_probabilities, top_indices = torch.topk(next_probabilities, beam_size, dim=-1)
            top_beams = top_indices // self.target_vocab_size
            next_tokens = top_indices % self.target_vocab_size

            # update the sequences in the decoded tensor, according to the best beams
            assert decoded.shape == (n_active, beam_size, max_target_seq_len)
            assert top_beams.shape == (n_active, beam_size)
            # add a singleton dimension at the end, and match the size of decoded tensor
            index = top_beams.view(n_active, beam_size, 1).expand_as(decoded)
            decoded = decoded.gather(dim=1, index=index)

            # update the hidden state for the next run
            assert top_beams.shape == (n_active, beam_size)
            assert decoder_hidden_next.shape == (self.n_gru_layers, n_active, beam_size, self.hidden_dim)
            # add a singleton dimension at the start, and the end, and expand to the same size as decoder_hidden
            index = top_beams.view(1, n_active, beam_size, 1).expand_as(decoder_hidden)
            decoder_hidden = decoder_hidden_next.gather(dim=2, index=index)

            # place the next tokens in the correct places
            assert decoded.shape == (n_active, beam_size, max_target_seq_len)
            assert next_tokens.shape == (n_active, beam_size)
            decoded[:, :, i + 1] = next_tokens

            # update the probabilities
            probabilities = top_probabilities

            # deal with the case where we have <EOS>
            assert next_tokens.shape == (n_active, beam_size)
            assert probabilities.shape == (n_active, beam_size)
            finished_mask = (next_tokens == end_of_sequence_token)
            # take for each of the active sample, take the best ending sequence, if exists, else put there 0
            finished_probabilities = torch.where(finished_mask, probabilities, torch.zeros_like(probabilities))
            best_finished_probabilities, best_finished_indices = torch.max(finished_probabilities, dim=1)

            # compare the best probability for each active sample to the current best ending sequence.
            # if it is better then put this as the best sequence
            assert best_finished_probabilities.shape == (n_active,)
            assert best_probabilities.shape == (batch_size,)
            assert active_to_orig_index.shape == (n_active,)
            assert best_decoded.shape == (batch_size, max_target_seq_len)
            better_then_best_mask = (best_finished_probabilities > best_probabilities[active_to_orig_index])
            update_indices = active_to_orig_index[better_then_best_mask]
            # update the probabilities
            best_probabilities[update_indices] = best_finished_probabilities[better_then_best_mask]
            # update the sequences
            assert decoded.shape == (n_active, beam_size, max_target_seq_len)
            beam_index = best_finished_indices[better_then_best_mask]
            best_decoded[update_indices] = decoded[better_then_best_mask, beam_index]

            # put zero probability where the next token is <EOS>.
            assert probabilities.shape == (n_active, beam_size)
            probabilities = torch.where(finished_mask, torch.zeros_like(probabilities), probabilities)

            # remove the samples where all their probabilities are lower then the best current one from the active set
            best_intermediate_p, _ = torch.max(probabilities, dim=1)
            still_better_then_best_mask = (best_intermediate_p > best_probabilities[active_to_orig_index])
            # now filter out all the tensors that are not active
            n_active = still_better_then_best_mask.sum()
            if n_active == 0:
                break

            probabilities = probabilities[still_better_then_best_mask]
            decoder_hidden = decoder_hidden[:, still_better_then_best_mask]
            decoded = decoded[still_better_then_best_mask]
            active_to_orig_index = active_to_orig_index[still_better_then_best_mask]

        # convert the best generated sequences into lists
        decoded_list = []
        for i in range(batch_size):
            mask = best_decoded[i] != self.pad_index
            decoded_list.append(best_decoded[i, mask].tolist())

        return decoded_list

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
        decoded[:, 0] = start_of_sequence_token
        decoder_input = decoded[:, 0].view(1, batch_size)
        n_active = batch_size

        for i in range(max_target_seq_len - 1):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            assert decoder_out.shape == (1, n_active, self.target_vocab_size)
            assert decoder_hidden.shape == (self.n_gru_layers, n_active, self.hidden_dim)

            # insert the next tokens to the decoded list
            decoder_out = decoder_out.view(n_active, self.target_vocab_size)
            next_tokens = torch.argmax(decoder_out, dim=1)
            decoded[batch_mapping, i + 1] = next_tokens

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