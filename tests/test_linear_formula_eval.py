import unittest
import math_qa.math_qa as mathqa
import config


class TestAllEval(unittest.TestCase):
    def test_all_eval(self):
        partitions = ['train', 'test', 'dev']
        for part in partitions:
            data = mathqa.load_dataset(config.MATHQA_DIR, part)
            for dp in data:
                value = dp.processed_linear_formula.eval(dp.processed_problem.numbers)
                self.assertIsNotNone(value)
                self.assertIsInstance(value, (int, float))


if __name__ == '__main__':
    unittest.main()
