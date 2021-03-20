"""This is a class to handle the linear formula feild in the MathQA dataset."""
from enum import Enum
from typing import Union

from math_qa.constants import const_dict
from math_qa import operations


class BadProgram(Exception):
    pass


class ArgType(Enum):
    const = 0
    temp = 1
    from_text = 2


class Arg:
    def __init__(self, text: str):
        self.text = text
        if text.startswith('#'):
            self.type = ArgType.temp
            self.value = int(text[1:])
        elif text.startswith('n'):
            self.type = ArgType.from_text
            self.value = int(text[1:])
        elif text.startswith('const_'):
            self.type = ArgType.const
            self.value = const_dict[text]
        else:
            raise BadProgram(f'got bad argument {text}')

    def eval(self, numbers_from_text: list, temps: list) -> Union[float, int]:
        if self.type == ArgType.temp:
            return temps[self.value]
        if self.type == ArgType.const:
            return self.value
        if self.type == ArgType.from_text:
            return numbers_from_text[self.value]
        raise BadProgram(f'could not get the value with {numbers_from_text}, {temps}, {self.type}, {self.value}')

    def __hash__(self):
        return hash((self.type, self.value))

    def __eq__(self, other):
        return self.type == other.type and self.value == other.value

    def __str__(self):
        return f"<Arg: type={self.type.name}, value={self.value}>"

    def __repr__(self):
        return self.__str__()


class Operation:
    def __init__(self, text: str):
        text = text.replace(')', '')
        self.func_name, args = text.split('(')
        try:
            self.func = getattr(operations, self.func_name)
        except AttributeError as e:
            print(e)
        args = args.split(',')
        self.args = list(map(Arg, args))

    def eval(self, numbers_from_text: list, temps: list) -> Union[float, int]:
        args_values = []
        for arg in self.args:
            args_values.append(arg.eval(numbers_from_text, temps))
        return self.func(*args_values)


class LinearFormula:
    def __init__(self, linear_formula: str):
        self.text = linear_formula
        ops_text = linear_formula.split('|')
        self.ops = []
        for op_text in ops_text:
            if '(' not in op_text:
                continue
            if op_text == '':
                continue
            self.ops.append(Operation(op_text))

    def eval(self, numbers_from_text: list) -> Union[float, int]:
        t = None
        temps = []
        for op in self.ops:
            t = op.eval(numbers_from_text, temps)
            temps.append(t)
        return t

    def get_program_string(self) -> str:
        program_string = []
        for op in self.ops:
            program_string.append(op.func_name)
            for arg in op.args:
                program_string.append(arg.text)
        return ' '.join(program_string)

