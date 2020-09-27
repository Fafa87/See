import os
import pathlib
import shutil
import tempfile
import unittest

import imageio
import numpy as np


class TestBase(unittest.TestCase):
    def setUp(self):
        self.to_clear = []

    def tearDown(self):
        for path in self.to_clear:
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)

    def save_temp(self, path, image):
        self.to_clear.append(path)
        imageio.imsave(path, image)

    def create_temp(self, path):
        self.to_clear.append(path)
        return open(path, "w")

    def root_test_dir(self, *path_components):
        return str(pathlib.Path(__file__).parent.joinpath(*path_components))

    def create_temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.to_clear.append(temp_dir)
        return temp_dir

    def draw_cell(self, image, position, radius, value):
        left = max(0, position[0] - radius)
        top = max(0, position[1] - radius)
        right = position[0] + radius
        bottom = position[1] + radius
        image[top: bottom + 1, left: right + 1] = value

    def np_assert_not_equal(self, expected, actual):
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(expected, actual)

    def np_assert_equal(self, expected, actual):
        np.testing.assert_array_equal(expected, actual)
