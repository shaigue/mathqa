import unittest
from math_qa.math_qa import tokenize_linear_formula, normalize_linear_formula, join_tokenized_linear_formula
from math_qa import math_qa


class TestMathQA(unittest.TestCase):
    def test_linear_formula_processing(self):
        lf = 'add(n1,n2)|add(#0,const_pi)|macro_1(#0,#0,n0)|'
        tokenized = ['add', '(', 'n1', ',', 'n2', ')', '|', 'add', '(', '#0', ',', 'const_pi', ')',
                     '|', 'macro_1', '(', '#0', ',', '#0', ',', 'n0', ')', '|']
        value = tokenize_linear_formula(lf)
        self.assertEqual(tokenized, value)
        self.assertEqual(join_tokenized_linear_formula(tokenized), lf)

    def test_no_punctuation(self):
        lf = 'add(n1,n2)|add(#0,const_pi)|macro_1(#0,#0,n0)'
        tokenized = ['add', 'n1', 'n2', 'add', '#0', 'const_pi', 'macro_1', '#0', '#0', 'n0']
        tokens = math_qa.tokenize_linear_formula_no_punctuations(lf)
        self.assertEqual(tokenized, tokens)
        joined = math_qa.join_tokenized_linear_formula_no_punctuations(tokens)
        self.assertEqual(joined, lf)


if __name__ == '__main__':
    unittest.main()
