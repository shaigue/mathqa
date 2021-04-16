import unittest

from preprocessing.bpe import byte_pair_encode, byte_pair_decode


class TestBPE(unittest.TestCase):
    def test_something(self):
        n_src_tokens = 5
        n_dst_tokens = 8
        sequence_list = [
            [0, 1, 2, 1, 2, 4, 3, 3],
            [3, 3, 1, 2, 4, 1, 2, 4],
            [2, 1, 3, 3, 3, 1, 2, 4]
        ]
        encoded_sequence, bpe_record = byte_pair_encode(sequence_list, n_src_tokens, n_dst_tokens)
        self.assertEqual(bpe_record, [
            (5, (1, 2)),
            (6, (5, 4)),
            (7, (3, 3))
        ])
        decoded_sequence = byte_pair_decode(encoded_sequence, bpe_record)
        self.assertEqual(decoded_sequence, sequence_list)


if __name__ == '__main__':
    unittest.main()
