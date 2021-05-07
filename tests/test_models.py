import unittest

import torch

from models.common import one_pad


class MyTestCase(unittest.TestCase):
    def test_one_pad(self):
        x = torch.randn(10, 11, 12)
        x_pad = one_pad(x, 1)
        self.assertEqual(x_pad.shape, (10, 12, 12))
        self.assertTrue(torch.all(x_pad[:, -1, :] == 1))


if __name__ == '__main__':
    unittest.main()
