"""Processing methods for mathqa dataset"""
from collections import defaultdict
import math
from enum import Enum
from pathlib import Path
from typing import Union, Iterator, Optional

import numpy as np

from math_qa import math_qa as mathqa
from math_qa.math_qa import RawMathQAEntry
from text_processing import TextVectorizer
from program_graph.program import Program
from program_graph.macro_substitution import MacroData


# TODO: build the code vocabulary from known symbols, not from data
Number = Union[int, float]


class ErrorType(Enum):
    no_error = 0
    syntax = 1
    value = 2


class MathQADatapoint:
    def __init__(self, text_token_indices: list[int], code_token_indices: list[int], extracted_numbers: list[int],
                 program: Program, evaluated_result: Number):
        self.text_token_indices = text_token_indices
        self.code_token_indices = code_token_indices
        self.extracted_numbers = extracted_numbers
        self.program = program
        self.evaluated_result = evaluated_result


class ProcessedMathQAEntry:
    def __init__(self, raw_entry: RawMathQAEntry, refactored_program: Optional[Program] = None):
        self.processed_problem, self.extracted_numbers = mathqa.extract_numbers_from_problem(raw_entry.problem)
        if refactored_program is not None:
            self.linear_formula = str(refactored_program)
            self.program = refactored_program
        else:
            self.linear_formula = raw_entry.linear_formula
            self.program = Program.from_linear_formula(self.linear_formula)


def _check_programs_match(raw: RawMathQAEntry, processed: ProcessedMathQAEntry, macro_data: MacroData) -> bool:
    """Checks if the original and refactored program produce the same value"""
    inputs = processed.extracted_numbers
    original_program = Program.from_linear_formula(raw.linear_formula)
    original_value = original_program.eval(inputs)
    refactored_program = processed.program
    refactored_value = refactored_program.eval(inputs, macro_data.macro_dict)
    return original_value == refactored_value


def _process_raw_mathqa_entries(raw_entries: dict[str, list[RawMathQAEntry]],
                                macro_data: Optional[MacroData] = None) -> dict[str, list[ProcessedMathQAEntry]]:
    """Processes the raw mathqa entries, and replaces the macros in train if required."""
    processed_entries = defaultdict(list)
    for part, raw_entries in raw_entries.items():
        if part == 'train' and macro_data is not None:
            assert len(raw_entries) == len(macro_data.modified_programs), "There should be exactly the same number of" \
                                                                          " programs"
            # use the macro_10 programs instead here
            for i, raw in enumerate(raw_entries):
                processed = ProcessedMathQAEntry(raw, macro_data.modified_programs[i])
                assert _check_programs_match(raw, processed, macro_data), "Program evaluation does not match"
                processed_entries[part].append(processed)
        else:
            for raw in raw_entries:
                processed = ProcessedMathQAEntry(raw)
                processed_entries[part].append(processed)
    return processed_entries


def _get_processed_problems(processed_entries: dict[str, list[ProcessedMathQAEntry]], partition: str) -> list[str]:
    processed_problems = []
    partition_entries = processed_entries[partition]
    for entry in partition_entries:
        processed_problems.append(entry.processed_problem)
    return processed_problems


def _get_linear_formulas(processed_entries: dict[str, list[ProcessedMathQAEntry]], partition: str) -> list[str]:
    processed_problems = []
    partition_entries = processed_entries[partition]
    for entry in partition_entries:
        processed_problems.append(entry.linear_formula)
    return processed_problems


class ErrorReport:
    def __init__(self, error_type: ErrorType, generated_tokens: list[str]):
        self.error_type = error_type
        self.generated_tokens = generated_tokens


class MathQAManager:
    partitions = ['train', 'dev', 'test']

    def __init__(self, root_dir: Path, max_vocabulary_size: int, dummy=False, macro_file: Optional[Path] = None,
                 no_punctuation: bool = False):
        self.dummy = dummy

        self.macro_data = None
        self.macro_dict = None
        if macro_file is not None:
            self.macro_data = MacroData.from_file(macro_file)
            self.macro_dict = self.macro_data.macro_dict

        raw_entries = {}
        for partition in self.partitions:
            raw_entries[partition] = mathqa.load_dataset(root_dir, partition)

        processed_entries = _process_raw_mathqa_entries(raw_entries, self.macro_data)

        # small dummy dataset to test the pipeline
        if dummy:
            dummy_data = processed_entries['train'][:200]
            for partition in self.partitions:
                processed_entries[partition] = dummy_data

        train_processed_problems = _get_processed_problems(processed_entries, 'train')
        train_linear_formulas = _get_linear_formulas(processed_entries, 'train')

        # if padding is stated then add text padding at the start

        self.text_vectorizer = TextVectorizer(train_processed_problems, max_vocabulary_size)

        if no_punctuation:
            join_fn = mathqa.join_tokenized_linear_formula_no_punctuations
            split_fn = mathqa.tokenize_linear_formula_no_punctuations
        else:
            join_fn = mathqa.join_tokenized_linear_formula
            split_fn = mathqa.tokenize_linear_formula

        self.code_vectorizer = TextVectorizer(train_linear_formulas, normalize_fn=mathqa.normalize_linear_formula,
                                              split_fn=split_fn,
                                              join_fn=join_fn)

        self.datapoints = self._get_datapoints(processed_entries)

    def get_datapoint(self, partition: str, index: int) -> MathQADatapoint:
        return self.datapoints[partition][index]

    def get_partition_length(self, partition: str) -> int:
        return len(self.datapoints[partition])

    @property
    def text_vocabulary_size(self):
        return self.text_vectorizer.vocabulary_size

    @property
    def pad_index(self):
        return self.code_vectorizer.pad_token_index

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

    def get_error_report(self, generated: list[int], inputs: list[Number], correct_answer: Number) -> ErrorReport:
        generated = self.code_vectorizer.token_index_list_to_token_list(generated)
        try:
            linear_formula = self.code_vectorizer.token_list_to_string(generated)
            program = Program.from_linear_formula(linear_formula)
            value = program.eval(inputs, self.macro_dict)
            correct = math.isclose(value, correct_answer)
            if correct:
                return ErrorReport(ErrorType.no_error, generated)
            return ErrorReport(ErrorType.value, generated)

        except:
            return ErrorReport(ErrorType.syntax, generated)

    def _get_datapoints(self, processed_entries: dict[str, list[ProcessedMathQAEntry]]) -> dict[str, list[MathQADatapoint]]:
        datapoints = defaultdict(list)
        for part, entries in processed_entries.items():
            for entry in entries:
                evaluated_result = entry.program.eval(entry.extracted_numbers, macro_dict=self.macro_dict)
                datapoint = MathQADatapoint(
                    text_token_indices=self.text_vectorizer.string_to_token_index_list(entry.processed_problem),
                    code_token_indices=self.code_vectorizer.string_to_token_index_list(entry.linear_formula),
                    extracted_numbers=entry.extracted_numbers,
                    program=entry.program,
                    evaluated_result=evaluated_result,
                )
                datapoints[part].append(datapoint)

        return datapoints
