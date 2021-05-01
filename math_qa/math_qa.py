"""Entry point for the module"""
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import re
from typing import Union

import config
from math_qa import operations
from math_qa import constants


class RawMathQAEntry:
    def __init__(self, problem: str, rationale: str, options: str, correct: str, annotated_formula: str,
                 linear_formula: str, category: str):
        self.problem = problem
        self.rationale = rationale
        self.options = options
        self.correct = correct
        self.annotated_formula = annotated_formula
        self.linear_formula = linear_formula
        self.category = category

    @classmethod
    def from_dict(cls, d: dict[str, str]):
        kwargs = {}
        for k, v in d.items():
            kwargs[k.lower()] = v
        return cls(**kwargs)


def _load_json(json_path: Path) -> list[RawMathQAEntry]:
    """Reads the contents of a json file."""
    with json_path.open('r') as f:
        json_list: list[dict[str, str]] = json.load(f)

    return list(map(RawMathQAEntry.from_dict, json_list))


def _load_all_raw(math_qa_path: Path) -> list[RawMathQAEntry]:
    """Loads all the partitions of the MathQA dataset."""
    all_raw = []
    for partition in math_qa_path.glob('*.json'):
        all_raw += _load_json(partition)
    return all_raw


def load_dataset(root_dir: Path, partition: str) -> list[RawMathQAEntry]:
    """Loads the correct partition of the dataset"""
    assert partition in ['train', 'test', 'dev', 'all'], f"got bad partition {partition}"
    assert root_dir.is_dir(), f"{root_dir} is not a directory"
    part_path = root_dir / f"{partition}.json"
    assert part_path.is_file(), f"{part_path} is not a file"
    return _load_json(part_path)


def load_all_dataset(root_dir: Path = config.MATHQA_DIR) -> dict[str, list[RawMathQAEntry]]:
    partitions = ['train', 'dev', 'test']
    return {part: load_dataset(root_dir, part) for part in partitions}

@dataclass
class OperatorDescription:
    name: str
    n_args: int


@dataclass
class ConstantDescriptor:
    name: str
    value: float


def get_operators_descriptors() -> list[OperatorDescription]:
    """Returns the MathQA operator descriptors"""
    operation_names = dir(operations)
    operation_names.remove('math')
    operation_names = filter(lambda x: not x.startswith('_'), operation_names)
    descriptors = []
    for name in operation_names:
        signature = inspect.signature(getattr(operations, name))
        n_args = len(signature.parameters)
        descriptors.append(OperatorDescription(name, n_args))
    return descriptors


def get_constants_descriptors() -> list[ConstantDescriptor]:
    const_dict = constants.const_dict
    descriptors = []
    for name, value in const_dict.items():
        descriptors.append(ConstantDescriptor(name, value))
    return descriptors


def get_categories() -> list[str]:
    return ['other', 'general', 'physics', 'gain', 'geometry', 'probability']


def is_commutative(op) -> bool:
    commutative_ops = (
        operations.multiply,
        operations.triangle_perimeter,
        operations.surface_rectangular_prism,
        operations.volume_rectangular_prism,
        operations.add,
        operations.speed_in_still_water,
        operations.rectangle_area,
        operations.max,
        operations.min,
        operations.gcd,
        operations.rectangle_perimeter,
        operations.lcm,
        operations.rhombus_area,
        operations.diagonal,
        operations.triangle_area_three_edges,
        operations.triangle_area,
    )
    commutative_ops_names = tuple(map(lambda x: x.__name__, commutative_ops))

    if isinstance(op, str):
        return op in commutative_ops_names
    return op in commutative_ops


def extract_numbers_from_problem(problem: str) -> tuple[str, list[Union[int, float]]]:
    """Takes a problem text, finds the numbers in it, and replaces them with <num> token.

    :returns the processed string and the list of extracted numbers.
    """
    number_regexp = re.compile(r'\d+\.?\d*')
    number_token = '<num>'
    problem = problem.replace(',', '')
    extracted_numbers = []
    for n in number_regexp.findall(problem):
        if '.' in n:
            n = float(n)
        else:
            n = int(n)
        extracted_numbers.append(n)
    problem = number_regexp.sub(number_token, problem)
    return problem, extracted_numbers


def normalize_linear_formula(linear_formula: str) -> str:
    """Removes spaces from the linear formula"""
    return linear_formula.replace(' ', '')


def tokenize_linear_formula(linear_formula: str) -> list[str]:
    """Splits the linear formula into individual tokens."""
    # reaching this
    split_chars = '(),|'
    tokens = []
    last_token = ''
    for char in linear_formula:
        if char in split_chars:
            if last_token != '':
                tokens.append(last_token)
            tokens.append(char)
            last_token = ''
        else:
            last_token += char
    return tokens


def get_n_ops(dp: RawMathQAEntry) -> int:
    lf = dp.linear_formula
    if lf[-1] == '|':
        lf = lf[:-1]
    return lf.count('|')


def get_n_temps(dp: RawMathQAEntry) -> int:
    return get_n_ops(dp) - 1


def get_n_inputs(dp: RawMathQAEntry) -> int:
    number_regexp = re.compile(r'\d+\.?\d*')
    matches = number_regexp.findall(dp.problem)
    return len(matches)


def get_max_train_temps() -> int:
    all_data = load_all_dataset()
    return max(get_n_temps(dp) for dp in all_data['train'])


def get_max_train_inputs() -> int:
    all_data = load_all_dataset()
    return max(get_n_inputs(dp) for dp in all_data['train'])


def join_tokenized_linear_formula(tokenized_linear_formula: list[str]) -> str:
    return ''.join(tokenized_linear_formula)


def tokenize_linear_formula_no_punctuations(linear_formula: str) -> list[str]:
    """Splits the linear formula into individual tokens."""
    symbol_list = re.split(r'[,|() ]', linear_formula)
    symbol_list = list(filter(lambda x: x != '', symbol_list))
    return symbol_list


def join_tokenized_linear_formula_no_punctuations(tokenized_linear_formula: list[str]) -> str:
    def is_arg(token: str) -> bool:
        m = re.match(r'n\d+|#\d+|const_\w+', token)
        return m is not None

    last_type = None
    linear_formula = ''
    for token in tokenized_linear_formula:
        if is_arg(token):
            if last_type is None:
                pass  # raise RuntimeError('should not start with argument')
            elif last_type == 'arg':
                linear_formula += ','
            elif last_type == 'op':
                linear_formula += '('
            else:
                assert False, "should not get here"
            last_type = 'arg'
        else:
            if last_type is None:
                pass
            elif last_type == 'arg':
                linear_formula += ')|'
            elif last_type == 'op':
                linear_formula += '()|'
            else:
                assert False, "should not get here"
            last_type = 'op'

        linear_formula += token

    linear_formula += ')'
    return linear_formula
