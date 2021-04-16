"""BPE stands for Byte Pair Encoding"""
from collections import Counter
from copy import deepcopy

# TODO: run training with BPE on the input, BPE on the output, BPE on both
# TODO: run training with macro and without macros on the programs with a certain complexity level
# TODO: find a way to calculate exactly how much symbols will we have to generate after macro sub, and before, and
#   choose the most useful one
# TODO: find other datasets that you can test on


def get_pair_counts(sequence_list: list[list[int]]) -> Counter:
    """Returns a mapping between pairs and their count in the sequences"""
    counter = Counter()
    for sequence in sequence_list:
        for i in range(len(sequence) - 1):
            counter[(sequence[i], sequence[i+1])] += 1
    return counter


def encode_pair(sequence_list: list[list[int]], token_to_encode: int, pair: tuple[int, int]) -> None:
    """Replaces inplace all the occurrences of the pair with the new token."""
    for sequence in sequence_list:
        i = 0
        while i < len(sequence) - 1:
            if pair == (sequence[i], sequence[i+1]):
                sequence[i] = token_to_encode
                sequence.pop(i+1)
            i += 1


def byte_pair_encode(sequence_list: list[list[int]], n_src_tokens: int, n_dst_tokens: int) -> \
        tuple[list[list[int]], list[tuple[int, tuple[int, int]]]]:
    """encodes the input sequences with byte pair encoding"""
    assert n_src_tokens < n_dst_tokens
    sequence_list = deepcopy(sequence_list)
    new_token_src_pair: list[tuple[int, tuple[int, int]]] = []
    for new_token in range(n_src_tokens, n_dst_tokens):
        # count the pairs
        counter = get_pair_counts(sequence_list)
        # choose the pair that is most frequent
        most_common_list = counter.most_common(1)
        most_common_pair, count = most_common_list[0]
        # substitute this pair in the sequence list
        encode_pair(sequence_list, new_token, most_common_pair)
        # add this encoding
        new_token_src_pair.append((new_token, most_common_pair))

    return sequence_list, new_token_src_pair


def decode_pair(sequence_list: list[list[int]], token_to_decode: int, pair: tuple[int, int]) -> None:
    """Replaces the occurrences of token_to_decode with the pair, changes the list inplace."""
    for sequence in sequence_list:
        i = 0
        while i < len(sequence):
            if sequence[i] == token_to_decode:
                sequence[i] = pair[0]
                sequence.insert(i+1, pair[1])
                i += 1 # we assume that the token is not recursive :)
            i += 1


def byte_pair_decode(sequence_list: list[list[int]], new_token_src_pair: list[tuple[int, tuple[int, int]]]) -> \
        list[list[int]]:
    """Decodes the byte pair encoding. returns a new list"""
    sequence_list = deepcopy(sequence_list)
    # needs to decode in the reversed order
    for new_token, pair in reversed(new_token_src_pair):
        decode_pair(sequence_list, new_token, pair)

    return sequence_list
