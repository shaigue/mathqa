import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from mathqa_torch_loader import EvalBatch
from train_mathqa import get_model, get_manager, evaluate, get_loader


class TestBeamSearch(unittest.TestCase):
    def test_beam_search(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        manager = get_manager()
        state_dict_path = Path('../training_logs/no_punc_macro_0/model.pt')
        state_dict = torch.load(state_dict_path, map_location=device)
        model = get_model(manager).to(device)
        model.load_state_dict(state_dict)
        loader = get_loader(manager, device, 'dev', train=False, batch_size=32)
        batch: EvalBatch = next(iter(loader))
        generated_beam = model._beam_search_decode(batch.text_tokens, batch.text_lens, manager.code_start_token_index,
                                                   manager.code_end_token_index, manager.code_max_len, 4)
        generated_no_beam = model._greedy_decode(batch.text_tokens, batch.text_lens, manager.code_start_token_index,
                                                 manager.code_end_token_index, manager.code_max_len)
        # make sure that there is at least one result that is different
        different_i = None
        for i in range(len(generated_beam)):
            if generated_no_beam[i] != generated_beam[i]:
                different_i = i
                break
        self.assertIsNotNone(different_i)
        # make sure that the probabilities of the one found in the beam search is greater then the other
        greedy_dec = generated_no_beam[different_i]
        beam_dec = generated_beam[different_i]

        def batch_single(seq: list[int]) -> tuple[torch.Tensor, list[int]]:
            t = torch.tensor(seq, device=device)
            t = t.unsqueeze(1)
            return t, [len(seq)]

        input_seq = batch.text_tokens[:, different_i].unsqueeze(1)
        input_len = [batch.text_lens[different_i]]
        greedy_dec, greedy_len = batch_single(greedy_dec)
        beam_dec, beam_len = batch_single(beam_dec)
        logits_greedy = model.forward(input_seq, greedy_dec, input_len, greedy_len)
        logits_beam = model.forward(input_seq, beam_dec, input_len, beam_len)
        p_greedy = F.softmax(logits_greedy, dim=-1).view(greedy_len[0], model.target_vocab_size)
        p_beam = F.softmax(logits_beam, dim=-1).view(beam_len[0], model.target_vocab_size)
        p_g = 1
        for i, token in enumerate(greedy_dec[1:]):
            p_g *= p_greedy[i, token]
        p_b = 1
        for i, token in enumerate(beam_dec[1:]):
            p_b *= p_beam[i, token]
        self.assertGreater(p_b, p_g)


if __name__ == '__main__':
    unittest.main()
