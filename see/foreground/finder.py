import numpy as np


class ForegroundFinder:
    def __init__(self):
        pass

    def update(self, image, rough_foreground_mask):
        pass

    def calc_prob(self, image) -> np.ndarray:
        pass

    def rectify_mask(self,image, foreground_mask,
                     remove_lower = 0.01,
                     add_higher = 0.99) -> np.ndarray:
        pass

    def verify_static(self, image, diff) -> bool:
        pass

    def _get_confident_background(self, image, foreground_mask):
        pass

    def _clean_image(self, image):
        pass
