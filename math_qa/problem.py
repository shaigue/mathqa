"""This is a class for handeling the problem field in the MathQA dataset."""
from typing import List, Union
import re


class Problem:
    num_token = '<num>'
    number_regexp = re.compile(r'\d+\.?\d*')

    def __init__(self, problem: str):
        self.numbers = self._extract_numbers(problem)
        self.text = self._replace_numbers(problem)

    @classmethod
    def _extract_numbers(cls, text: str) -> List[Union[int, float]]:
        text = text.replace(',', '')
        numbers = []
        for n in cls.number_regexp.findall(text):
            if '.' in n:
                n = float(n)
            else:
                n = int(n)
            numbers.append(n)
        return numbers

    @classmethod
    def _replace_numbers(cls, text: str) -> str:
        text = text.replace(',', '')
        text = cls.number_regexp.sub(cls.num_token, text)
        return text
