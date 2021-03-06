import typing as t

import cv2
import numpy as np


class StaticBackgroundModel:
    UNKNOWN_PIXEL = (128, 128, 128)
    UNKNOWN_PIXEL_ERROR = 64

    def __init__(self, update_inertia=None, error_inertia=None, diff_method=None):
        """
        Args:
            update_inertia [0-inf): the more the more time has to pass for background to adapt
            error_inertia [0-inf): the more the longer it takes to the error to adapt
            diff_method:
                - rgb (simple average of red, green, blue)
                - rgs (red, green, lightness)
        """
        self.background_mean = None
        self.background_error = None
        self.background_mask = None
        self.config = {"update_inertia" : update_inertia, "error_inertia": error_inertia, "diff_method": diff_method}

    def update(self, image: np.ndarray, background_mask: np.ndarray):
        assert image.ndim == 3 and image.dtype == np.uint8, f"incorrect ndim={image.ndim}, dtype={image.dtype}"
        assert background_mask.dtype == np.bool, f"dtype={image.dtype}"

        if self.background_mask is None:
            self.background_mean = image.copy().astype(np.float)
            self.background_mask = background_mask.copy()
            # TODO return only when the error is sensible (at least two values)
            self.background_error = np.ones(background_mask.shape, dtype=np.float) * self.UNKNOWN_PIXEL_ERROR
            self.background_error[background_mask] = 1
        else:
            new_background_areas = background_mask > self.background_mask
            self.background_mean[new_background_areas] = image[new_background_areas]
            self.background_mask = background_mask | self.background_mask

            update_inertia = self.config['update_inertia']
            self.background_mean[background_mask] = \
                (update_inertia * self.background_mean[background_mask]
                 + image[background_mask]) / (1 + update_inertia)

            error_inertia = self.config['error_inertia']
            new_diff = self.calc_diff(self.background_mean, image)
            self.background_error[new_background_areas] = np.maximum(1, new_diff[new_background_areas])  # it should never be zero
            self.background_error[background_mask] = \
                (error_inertia * self.background_error[background_mask]
                 + new_diff[background_mask]) / (1 + error_inertia)

    def get(self) -> t.Optional[np.ndarray]:
        if self.background_mean is None:
            return None
        res = self.background_mean.copy()
        res[self.background_mask == 0] = self.UNKNOWN_PIXEL
        return res

    def get_details(self) -> t.Optional[dict]:
        if self.background_mean is None:
            return None
        return {'background': self.background_mean, 'error': self.background_error, 'mask': self.background_mask}

    def reset(self):
        self.background_mean = None
        self.background_error = None
        self.background_mask = None

    def calc_diff(self, image1, image2, method=None) -> np.ndarray:
        method = method or self.config['diff_method']
        if method == 'rgb':
            diff_per_channel = cv2.absdiff(image1.astype(np.int8), image2.astype(np.int8))
            mean_diff_per_pixel = np.mean(diff_per_channel, axis=-1)
            return mean_diff_per_pixel
        elif method == 'rgs':
            raise NotImplementedError(method)
        else:
            raise NotImplementedError(method)
