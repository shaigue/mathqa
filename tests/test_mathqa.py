import unittest
from math_qa.dataset import tokenize_linear_formula, normalize_linear_formula, join_tokenized_linear_formula


class TestMathQA(unittest.TestCase):
    def test_linear_formula_processing(self):
        lf = 'add(n1,n2)|add(#0,const_pi)|macro_1(#0,#0,n0)|'
        tokenized = ['add', '(', 'n1', ',', 'n2', ')', '|', 'add', '(', '#0', ',', 'const_pi', ')',
                     '|', 'macro_1', '(', '#0', ',', '#0', ',', 'n0', ')', '|']
        value = tokenize_linear_formula(lf)
        self.assertEqual(tokenized, value)
        self.assertEqual(join_tokenized_linear_formula(tokenized), lf)


if __name__ == '__main__':
    unittest.main()
