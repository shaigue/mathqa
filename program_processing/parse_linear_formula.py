from dataclasses import dataclass
from typing import Union

from program_processing.common import ArgType


@dataclass(frozen=True)
class Arg:
    arg_type: ArgType
    key: Union[str, int]

    @classmethod
    def from_str(cls, s: str):
        if s.startswith('n'):
            key = int(s[1:])
            t = ArgType.input

        elif s.startswith('#'):
            key = int(s[1:])
            t = ArgType.op

        elif s.startswith('const'):
            key = s
            t = ArgType.const
        else:
            assert False

        return cls(t, key)

    def __str__(self) -> str:
        if self.arg_type == ArgType.const:
            return self.key
        if self.arg_type == ArgType.op:
            return f"#{self.key}"
        if self.arg_type == ArgType.input:
            return f"n{self.key}"


@dataclass()
class ParsedLinearFormula:
    op_list: list[str]
    arg_list_list: list[list[Arg]]

    def __post_init__(self):
        assert len(self.op_list) == len(self.arg_list_list)

    def __len__(self) -> int:
        return len(self.op_list)


def parse_linear_formula(linear_formula: str) -> ParsedLinearFormula:
    op_arg_list = linear_formula.split('|')
    op_list = []
    arg_list_list = []

    for op_args in op_arg_list:
        if op_args == '':
            continue

        op_args = op_args.replace(')', '')
        op, args = op_args.split('(')
        op_list.append(op)

        args = args.split(',')
        processed_args = []

        for arg in args:
            if arg == '':
                continue
            arg = Arg.from_str(arg)
            processed_args.append(arg)

        arg_list_list.append(processed_args)

    return ParsedLinearFormula(op_list, arg_list_list)
