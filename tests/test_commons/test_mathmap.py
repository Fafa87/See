import numpy as np

import see._commons.mathmap as mathmap
import tests.testbase


class TestMathMap(tests.testbase.TestBase):
    def setUp(self):
        super().setUp()

    def test_linear_mapping_on_scalars(self):
        # edges
        self.assertEqual(0.0, mathmap.multi_linear_mapping_2d(0.0, lx=0, ly=0, rx=1, ry=1))
        self.assertEqual(1.0, mathmap.multi_linear_mapping_2d(1.0, lx=0, ly=0, rx=1, ry=1))

        # center of the ranges
        self.assertEqual(2, mathmap.multi_linear_mapping_2d(0.5, lx=0, ly=1, rx=1, ry=3))
        self.assertEqual(2.5, mathmap.multi_linear_mapping_2d(1.5, lx=1, ly=2, rx=2, ry=3))
        self.assertEqual(0.5, mathmap.multi_linear_mapping_2d(0.5, lx=0, ly=0, rx=1, ry=1))
        self.assertEqual(1.5, mathmap.multi_linear_mapping_2d(0.5, lx=0, ly=0, rx=1, ry=3))

    def test_linear_mapping_on_image(self):
        xs = np.array([[1, 3],
                       [1, 1]], dtype=np.uint8)
        lxs = np.array([[0.0, 2.0],
                        [0.0, 0.0]], dtype=np.uint8)
        rxs = np.array([[2.0, 4.0],
                        [2.0, 1.0]], dtype=np.uint8)
        prob = mathmap.multi_linear_mapping_2d(xs, lx=lxs, rx=rxs, ly=1.0, ry=0.5)
        self.assertEqual(prob[0, 0], 0.75)
        self.assertEqual(prob[0, 1], 0.75)
        self.assertEqual(prob[1, 0], 0.75)
        self.assertEqual(prob[1, 1], 0.5)
