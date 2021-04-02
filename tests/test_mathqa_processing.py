import unittest
import itertools

from mathqa_processing import MathQAManager
import config

# TODO: do thorough checks before submitting the paper


class TestMathQAProcessing(unittest.TestCase):
    def test_initialization(self):
        manager = MathQAManager(config.MATHQA_DIR, config.MAX_VOCABULARY_SIZE, dummy=False)
        manager_dummy = MathQAManager(config.MATHQA_DIR, config.MAX_VOCABULARY_SIZE, dummy=True)

    def test_dataset_iterator(self):
        manager = MathQAManager(config.MATHQA_DIR, config.MAX_VOCABULARY_SIZE, dummy=False)
        manager_dummy = MathQAManager(config.MATHQA_DIR, config.MAX_VOCABULARY_SIZE, dummy=True)

        managers = [manager, manager_dummy]
        partitions = ['train', 'test', 'dev', ['train', 'dev']]
        shuffle = [True, False]
        batch_size = [31, 65]
        for m, p, s, b in itertools.product(managers, partitions, shuffle, batch_size):
            m: MathQAManager
            got_smaller = False
            for i, datapoint in enumerate(m.iter_dataset(p, b, s)):
                # make sure that only the last batch can be smaller
                self.assertFalse(got_smaller)
                n = len(datapoint[0])
                if n != b:
                    self.assertLess(n, b)
                    got_smaller = True

                for entry in datapoint:
                    self.assertEqual(len(entry), n)

    def test_check_correctness(self):
        pass




if __name__ == '__main__':
    unittest.main()
