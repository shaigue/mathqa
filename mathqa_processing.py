"""Processing methods for mathqa dataset"""
from pathlib import Path
from typing import List, Union, Iterator
import re

import numpy as np

from math_qa import dataset as mathqa
from math_qa.linear_formula import BadProgram
from text_processing import TextVectorizer

# TODO:
#   - build the code vocabulary from known symbols, not from data


class MathQADatapoint:
    def __init__(self, text_token_indices: list[int], code_token_indices: list[int], extracted_numbers: list[int],
                 linear_formula: str):
        self.text_token_indices = text_token_indices
        self.code_token_indices = code_token_indices
        self.extracted_numbers = extracted_numbers
        self.linear_formula = linear_formula


class MathQAManager:
    partitions = ['train', 'dev', 'test']

    def __init__(self, root_dir: Path, max_vocabulary_size: int, dummy=False):
        self.dummy = dummy
        self.mathqa_entries = {}
        for partition in self.partitions:
            self.mathqa_entries[partition] = mathqa.load_dataset(root_dir, partition)

        # small dummy dataset to test the pipeline
        if dummy:
            dummy_data = self.mathqa_entries['train'][:200]
            for partition in self.partitions:
                self.mathqa_entries[partition] = dummy_data

        train_texts = self._get_problem_texts('train')
        train_codes = self._get_program_strings('train')

        self.text_vectorizer = TextVectorizer(train_texts, max_vocabulary_size)
        self.code_vectorizer = TextVectorizer(train_codes)
        self.operators_descriptors = mathqa.get_operators_descriptors()
        self.operator_name_to_n_args = {od.name: od.n_args for od in self.operators_descriptors}

    def get_dataset_iterator(self, partition: Union[str, List[str]], shuffle=False) -> Iterator[MathQADatapoint]:
        """If it is a list of strings, then concatenate the partitions"""
        if isinstance(partition, str):
            partition = [partition]

        code_vectors = []
        text_vectors = []
        extracted_numbers = []
        linear_formulas = []
        for part in partition:
            code_vectors += self._get_program_vectors(part)
            text_vectors += self._get_problem_vectors(part)
            extracted_numbers += self._get_problem_numbers(part)
            linear_formulas += self._get_linear_formulas(part)

        assert len(code_vectors) == len(text_vectors) == len(extracted_numbers) == len(linear_formulas), \
            "Got uneven sizes. something is wrong."

        n_samples = len(code_vectors)
        indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(indices)

        for i in indices:
            yield MathQADatapoint(
                code_token_indices=code_vectors[i],
                text_token_indices=text_vectors[i],
                extracted_numbers=extracted_numbers[i],
                linear_formula=linear_formulas[i],
            )

    @property
    def text_vocabulary_size(self):
        return self.text_vectorizer.vocabulary_size

    @property
    def code_vocabulary_size(self):
        return self.code_vectorizer.vocabulary_size

    @property
    def code_start_token_index(self):
        return self.code_vectorizer.start_of_sequence_token_index

    @property
    def code_end_token_index(self):
        return self.code_vectorizer.end_of_sequence_token_index

    @property
    def code_max_len(self):
        return self.code_vectorizer.max_sequence_len

    def check_generated_code_correctness(self, code_token_indices: list[int], datapoint: MathQADatapoint) -> bool:
        original_linear_formula = datapoint.linear_formula
        extracted_numbers = datapoint.extracted_numbers
        # transform to tokens into a linear formula
        try:
            generated_linear_formula = self.token_sequence_to_linear_formula(code_token_indices, extracted_numbers)
            generated_linear_formula = mathqa.LinearFormula(generated_linear_formula)
            generated_evaluation = generated_linear_formula.eval(extracted_numbers)
        # if the generated program is invalid then count that as an error
        except BadProgram:
            return False
        # evaluate the 2 formulas with the same inputs and see if they are equal
        original_linear_formula = mathqa.LinearFormula(original_linear_formula)
        original_evaluation = original_linear_formula.eval(extracted_numbers)
        return original_evaluation == generated_evaluation

    def token_sequence_to_linear_formula(self, token_vector: list[int], extracted_numbers: list) -> str:
        """Converts a sequence of tokens to a linear formula

        :param token_vector: a vector of generated tokens
        :return: a linear formula string
        """
        text_vector = self.code_vectorizer.token_index_list_to_token_list(token_vector)
        # first gather all the (op, arguments) pairs
        operation_argument_list = []
        last_operation = None
        last_operation_arguments = None
        for token in text_vector:
            if self._is_arg(token):
                if last_operation_arguments is None:
                    # got argument before any operation
                    raise BadProgram
                last_operation_arguments.append(token)
            else:
                if last_operation is not None:
                    # not the first operation to insert
                    operation_argument_list.append((last_operation, last_operation_arguments))
                # start a new argument list
                last_operation = token
                last_operation_arguments = []
        # empty program is also bad
        if last_operation is None:
            raise BadProgram
        # add the last pair
        operation_argument_list.append((last_operation, last_operation_arguments))
        # check that they are legal
        for i, (operation, arguments) in enumerate(operation_argument_list):
            # check that the number of arguments for each function are ok
            if self.operator_name_to_n_args[operation] != len(arguments):
                raise BadProgram
            for arg in arguments:
                # check that number arguments are not out of range
                if arg.startswith('n'):
                    num = int(arg[1:])
                    if num >= len(extracted_numbers):
                        raise BadProgram
                # check that the temp arguments are smaller then the index
                if arg.startswith('#'):
                    num = int(arg[1:])
                    if num >= i:
                        raise BadProgram
        # the program is OK!
        linear_formula = ''
        first_exp = text_vector.pop(0)
        linear_formula += first_exp + '('

        last_is_arg = False
        for exp in text_vector:
            if self._is_arg(exp):
                if last_is_arg:
                    linear_formula += ','
                linear_formula += exp
                last_is_arg = True
            else:
                linear_formula += ')|' + exp + '('
                last_is_arg = False

        linear_formula += ')'
        return linear_formula

    def _get_program_vectors(self, partition: str) -> List[List[int]]:
        program_strings = self._get_program_strings(partition)
        program_vectors = [self.code_vectorizer.string_to_token_index_list(s) for s in program_strings]
        return program_vectors

    def _get_problem_vectors(self, partition: str) -> List[List[int]]:
        problem_texts = self._get_problem_texts(partition)
        problem_vectors = [self.text_vectorizer.string_to_token_index_list(s) for s in problem_texts]
        return problem_vectors

    def _get_problem_numbers(self, partition: str) -> List[List[Union[int, float]]]:
        problem_numbers = []
        for entry in self.mathqa_entries[partition]:
            problem_numbers.append(entry.processed_problem.numbers)
        return problem_numbers

    def _get_linear_formulas(self, partition: str) -> List[str]:
        linear_formulas = []
        for entry in self.mathqa_entries[partition]:
            linear_formulas.append(entry.linear_formula)
        return linear_formulas

    def _get_problem_texts(self, partition: str) -> List[str]:
        problem_texts = []
        for entry in self.mathqa_entries[partition]:
            problem_texts.append(entry.processed_problem.text)
        return problem_texts

    def _get_program_strings(self, partition: str) -> List[str]:
        program_strings = []
        for entry in self.mathqa_entries[partition]:
            program_strings.append(entry.processed_linear_formula.get_program_string())
        return program_strings

    @staticmethod
    def _is_arg(exp: str) -> bool:
        arg_regexp = re.compile(r'const_\w+|n\d+|#\d+')
        if arg_regexp.match(exp):
            return True
        return False

