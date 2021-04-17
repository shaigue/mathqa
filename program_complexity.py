# complexity 1: number of tokens in a program
import math

from math_qa import math_qa


def num_tokens(linear_formula: str) -> int:
    """Calculates the number of tokens that need to be generated for the program"""
    tokenized = math_qa.tokenize_linear_formula_no_punctuations(linear_formula)
    return len(tokenized)


def log1_num_tokens(linear_formula: str) -> float:
    """Gives a smoother distribution for the values"""
    return math.log2(1 + num_tokens(linear_formula))


def log1_num_tokens_on_entry(entry: math_qa.RawMathQAEntry) -> float:
    return log1_num_tokens(entry.linear_formula)
