"""Entry point for the module"""
from pathlib import Path
import json
from typing import List, Dict
from dataclasses import dataclass
import inspect

from math_qa import operations
from math_qa.operations import *
from math_qa import constants
from math_qa.linear_formula import LinearFormula
from math_qa.problem import Problem


class MathQAEntry:
    def __init__(self, problem: str, rationale: str, options: str, correct: str, annotated_formula: str,
                 linear_formula: str, category: str):
        self.problem = problem
        self.rationale = rationale
        self.options = options
        self.correct = correct
        self.annotated_formula = annotated_formula
        self.linear_formula = linear_formula
        self.category = category
        # processed values
        self.processed_linear_formula = LinearFormula(linear_formula)
        self.processed_problem = Problem(problem)

    @classmethod
    def from_dict(cls, d: Dict[str, str]):
        kwargs = {}
        for k, v in d.items():
            kwargs[k.lower()] = v
        return cls(**kwargs)


def load_json(json_path: Path) -> List[MathQAEntry]:
    """Reads the contents of a json file."""
    with json_path.open('r') as f:
        json_list: List[Dict[str, str]] = json.load(f)

    return list(map(MathQAEntry.from_dict, json_list))


def load_all_raw(math_qa_path: Path) -> List[MathQAEntry]:
    """Loads all the partitions of the MathQA dataset."""
    all_raw = []
    for partition in math_qa_path.glob('*.json'):
        all_raw += load_json(partition)
    return all_raw


def load_dataset(root_dir: Path, partition: str) -> List[MathQAEntry]:
    """Loads the correct partition of the dataset"""
    assert partition in ['train', 'test', 'dev', 'all'], f"got bad partition {partition}"
    assert root_dir.is_dir(), f"{root_dir} is not a directory"
    part_path = root_dir / f"{partition}.json"
    assert part_path.is_file(), f"{part_path} is not a file"
    return load_json(part_path)


@dataclass
class OperatorDescription:
    name: str
    n_args: int


@dataclass
class ConstantDescriptor:
    name: str
    value: float


def get_operators_descriptors() -> List[OperatorDescription]:
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


def get_constants_descriptors() -> List[ConstantDescriptor]:
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
