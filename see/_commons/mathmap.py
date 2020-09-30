import numpy as np


def multi_linear_mapping_2d(xs, lx, ly, rx, ry):
    ys = ly + (xs - lx) / np.maximum(rx - lx, 0.0001) * (ry - ly)
    return ys
