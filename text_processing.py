from typing import List, Iterable
from collections import Counter, defaultdict


def default_split_fn(s: str) -> list[str]:
    return s.split(' ')


def default_normalize_fn(s: str) -> str:
    return s.lower()


def default_join_fn(s: list[str]) -> str:
    return ' '.join(s)


# TODO: add the ability to state in advance all the tokens, without reading them from data
class TextVectorizer:
    pad_token = '<PAD>'
    unknown_token = '<UNK>'
    end_of_sequence_token = '<EOS>'
    start_of_sequence_token = '<SOS>'

    special_tokens = [unknown_token, pad_token, end_of_sequence_token, start_of_sequence_token]

    def __init__(self, strings: Iterable[str], max_tokens: int = None,
                 normalize_fn=default_normalize_fn, split_fn=default_split_fn, join_fn=default_join_fn,):

        assert split_fn is not None, "The split function cannot be None."

        self.max_tokens = max_tokens
        self.normalize_fn = normalize_fn
        self.split_fn = split_fn
        self.join_fn = join_fn


        # create the mapping from tokens to indices
        counter = Counter()
        max_seq_len = 0
        for s in strings:
            s = self._normalize_and_split(s)
            max_seq_len = max(max_seq_len, len(s))
            counter.update(s)

        tokens_ordered_by_frequency = []
        for token, count in counter.most_common():
            if token not in self.special_tokens:
                tokens_ordered_by_frequency.append(token)

        self.index_to_token = self.special_tokens + tokens_ordered_by_frequency
        if max_tokens is not None:
            self.index_to_token = self.index_to_token[:max_tokens]

        # for addition of <sos> and <eos> in the beginning and the end
        self.max_sequence_len = max_seq_len + 2

        unknown_token_index = self.index_to_token.index(self.unknown_token)
        self.token_to_index = defaultdict(lambda: unknown_token_index)
        for index, token in enumerate(self.index_to_token):
            self.token_to_index[token] = index

    def _normalize_and_split(self, text: str) -> list[str]:
        if self.normalize_fn is not None:
            text = self.normalize_fn(text)
        return self.split_fn(text)

    def convert_token_to_index(self, token: str) -> int:
        return self.token_to_index[token]

    def convert_index_to_token(self, index: int) -> str:
        return self.index_to_token[index]

    def string_to_token_index_list(self, s: str) -> list[int]:
        # process the string
        s = self._normalize_and_split(s)
        # add start and end of sequence tokens
        s.insert(0, self.start_of_sequence_token)
        s.append(self.end_of_sequence_token)
        # convert into index list
        return [self.token_to_index[token] for token in s]

    def token_index_list_to_token_list(self, index_list: List[int]) -> List[str]:
        if len(index_list) == 0:
            return []
        token_list = [self.index_to_token[token_index] for token_index in index_list]
        # remove <sos>, <eos> if in the start or end
        if token_list[0] == self.start_of_sequence_token:
            del token_list[0]
        if len(index_list) == 0:
            return []
        if token_list[-1] == self.end_of_sequence_token:
            del token_list[-1]
        return token_list

    def token_list_to_string(self, token_list: list[str]) -> str:
        return self.join_fn(token_list)

    def token_index_list_to_string(self, index_list: list[int]) -> str:
        token_list = self.token_index_list_to_token_list(index_list)
        return self.token_list_to_string(token_list)

    @property
    def pad_token_index(self):
        return self.token_to_index[self.pad_token]

    @property
    def unknown_token_index(self):
        return self.token_to_index[self.unknown_token]

    @property
    def end_of_sequence_token_index(self):
        return self.token_to_index[self.end_of_sequence_token]

    @property
    def start_of_sequence_token_index(self):
        return self.token_to_index[self.start_of_sequence_token]

    @property
    def vocabulary_size(self):
        return len(self.index_to_token)
