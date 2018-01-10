import unittest
import sys
sys.path.append('../preprocess')
import align_faces
import numpy as np
import math
from numpy.random import RandomState

class AlignFacesTest(unittest.TestCase):
    """Test face alignment moduke."""

    @classmethod
    def setUpClass(cls, random_seed=42):
        cls.rng = RandomState(seed=random_seed)

    def test_missing_equilateral_triangle_vertex(self,low=-20.0, high=20.0):
        x1, y1, x2, y2 = self.rng.uniform(low=low, high=high, size=4)
        x3, y3 = align_faces.missing_equilateral_triangle_vertex(x1, y1, x2, y2)
        dx12 = x1 - x2
        dy12 = y1 - y2
        dx23 = x2 - x3
        dy23 = y2 - y3
        dx31 = x3 - x1
        dy31 = y3 - y1
        l12_sq = dx12 * dx12 + dy12 * dy12
        l23_sq = dx23 * dx23 + dy23 * dy23
        l31_sq = dx31 * dx31 + dy31 * dy31
        msg1 = 'test l12 == l23 failed for ({:f}, {:f}), ({:f}, {:f})'.format(x1, y1, x2, y2)
        msg2 = 'test l23 == l31 failed for ({:f}, {:f}), ({:f}, {:f})'.format(x1, y1, x2, y2)
        msg3 = 'test l31 == l12 failed for ({:f}, {:f}), ({:f}, {:f})'.format(x1, y1, x2, y2)
        self.assertAlmostEqual(l12_sq, l23_sq, places=7, msg=msg1)
        self.assertAlmostEqual(l23_sq, l31_sq, places=7, msg=msg2)
        self.assertAlmostEqual(l31_sq, l12_sq, places=7, msg=msg3)


def suite(num_repeats=100):
    suite = unittest.TestSuite()
    for i in range(num_repeats):
        suite.addTest(AlignFacesTest('test_missing_equilateral_triangle_vertex'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = suite()
    runner.run(test_suite)