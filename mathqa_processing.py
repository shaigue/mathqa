"""Processing methods for mathqa dataset"""
from pathlib import Path
from typing import Union, Iterator, Optional
import math
from collections import defaultdict

import numpy as np

from math_qa import dataset as mathqa
from math_qa.dataset import RawMathQAEntry
from text_processing import TextVectorizer
from program_graph.extract_dags import Program
from program_graph.macro_substitution import MacroData


# TODO: build the code vocabulary from known symbols, not from data
class MathQADatapoint:
    def __init__(self, text_token_indices: list[int], code_token_indices: list[int], extracted_numbers: list[int],
                 program: Program):
        self.text_token_indices = text_token_indices
        self.code_token_indices = code_token_indices
        self.extracted_numbers = extracted_numbers
        self.program = program


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


class MathQAManager:
    partitions = ['train', 'dev', 'test']

    def __init__(self, root_dir: Path, max_vocabulary_size: int, dummy=False, macro_file: Optional[Path] = None):
        self.dummy = dummy

        self.macro_data = None
        if macro_file is not None:
            self.macro_data = MacroData.from_file(macro_file)

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

        self.text_vectorizer = TextVectorizer(train_processed_problems, max_vocabulary_size)
        self.code_vectorizer = TextVectorizer(train_linear_formulas, normalize_fn=mathqa.normalize_linear_formula,
                                              split_fn=mathqa.tokenize_linear_formula,
                                              join_fn=mathqa.join_tokenized_linear_formula)

        self.datapoints = self._get_datapoints(processed_entries)

    def iter_dataset(self, partition: Union[str, list[str]], shuffle=False) -> Iterator[MathQADatapoint]:
        """If it is a list of strings, then concatenate the partitions"""
        if isinstance(partition, str):
            partition = [partition]

        datapoints = []
        for part in partition:
            datapoints += self.datapoints[part]

        n_samples = len(datapoints)
        indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(indices)

        for i in indices:
            yield datapoints[i]

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
        try:
            linear_formula = self.code_vectorizer.token_index_list_to_string(code_token_indices)
            program = Program.from_linear_formula(linear_formula)
            inputs = datapoint.extracted_numbers
            if self.macro_data is not None:
                macro_dict = self.macro_data.macro_dict
            else:
                macro_dict = None
            value = program.eval(inputs, macro_dict)
            target_value = datapoint.program.eval(inputs, macro_dict)
            return math.isclose(value, target_value)
        except:
            return False

    def _get_datapoints(self, processed_entries: dict[str, list[ProcessedMathQAEntry]]) -> dict[str, list[MathQADatapoint]]:
        datapoints = defaultdict(list)
        for part, entries in processed_entries.items():
            for entry in entries:
                datapoint = MathQADatapoint(
                    text_token_indices=self.text_vectorizer.string_to_token_index_list(entry.processed_problem),
                    code_token_indices=self.code_vectorizer.string_to_token_index_list(entry.linear_formula),
                    extracted_numbers=entry.extracted_numbers,
                    program=entry.program
                )
                datapoints[part].append(datapoint)
        return datapoints
