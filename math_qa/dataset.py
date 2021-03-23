"""Entry point for the module"""
from pathlib import Path
import json
from typing import Union
from dataclasses import dataclass
import inspect
import re

from math_qa import operations
from math_qa.operations import *
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


def is_commutative(op) -> bool:
    commutative_ops = (
        multiply,
        triangle_perimeter,
        surface_rectangular_prism,
        volume_rectangular_prism,
        add,
        speed_in_still_water,
        rectangle_area,
        max,
        min,
        gcd,
        rectangle_perimeter,
        lcm,
        rhombus_area,
        diagonal,
        triangle_area_three_edges,
        triangle_area,
    )
    commutative_ops_names = tuple(map(lambda x: x.__name__, commutative_ops))

    if isinstance(op, str):
        return op in commutative_ops_names
    return op in commutative_ops


# TODO: maybe move to somewhere else, all bellow


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


def join_tokenized_linear_formula(tokenized_linear_formula: list[str]) -> str:
    return ''.join(tokenized_linear_formula)
