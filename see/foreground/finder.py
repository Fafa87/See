import cv2
import numpy as np

import see._commons.mathmap as mathmap
from see.foreground.backgrounds import StaticBackgroundModel


class ForegroundFinder:
    def __init__(self, background_model: StaticBackgroundModel, confident_size=1, cleaning=None):
        """
        Args:
            background_model: background model
            confident_size: size of the erosion of the background mask to get confident (should be odd)
            cleaning: dict with cleaning params
                method and specific params
        """
        cleaning = cleaning or {}
        self.static_background = background_model
        self.config = {"confident_size": confident_size, "cleaning": cleaning}

    @staticmethod
    def create_from_dicts(config_background_model: dict = None, confident_size=1, cleaning=None):
        """
        Args:
            config_background_model: dictionary with parameters to static background model
            confident_size: size of the erosion of the background mask to get confident (should be odd)
            cleaning: dict with cleaning params
                method and specific params
        """
        config_background_model = config_background_model or {}
        return ForegroundFinder(StaticBackgroundModel(**config_background_model),
                                confident_size=confident_size, cleaning=cleaning)

    def update(self, image, rough_foreground_mask):
        assert image.dtype == np.uint8
        assert rough_foreground_mask.dtype == np.bool

        image_clean = self._clean_image(image)
        background_mask = self._get_confident_background(image_clean, foreground_mask=rough_foreground_mask)
        self.static_background.update(image_clean, background_mask)

    def calc_prob(self, image) -> np.ndarray:
        assert image.dtype == np.uint8

        image_clean = self._clean_image(image)
        background_info = self.static_background.get_details()
        difference = self.static_background.calc_diff(image_clean, background_info['background'])
        foreground_probability = 1 - self._convert_to_percentage(difference, background_info['error'])
        foreground_probability[background_info['mask'] == 0] = 0.5
        if self.verify_static(image, foreground_probability):
            return foreground_probability
        else:
            self.static_background.reset()
            return np.ones_like(foreground_probability) * 0.5

    def rectify_mask(self, image, foreground_mask,
                     remove_lower=0.01,
                     add_higher=0.99) -> np.ndarray:
        pass

    def verify_static(self, image, diff) -> bool:
        # TODO
        return True

    def _get_confident_background(self, image, foreground_mask) -> np.ndarray:
        assert foreground_mask.dtype == np.bool
        erosion_size = self.config['confident_size']
        kernel = np.ones((erosion_size, erosion_size), np.uint8)
        confident_background = cv2.erode((1 - foreground_mask).astype(np.uint8), kernel).astype(np.bool)
        return confident_background

    def _clean_image(self, image) -> np.ndarray:
        assert image.dtype == np.uint8

        if self.config['cleaning']['method'] == 'median':
            median_size = self.config['cleaning']['size']
            return cv2.medianBlur(image, median_size)
        else:
            raise NotImplementedError(self.config['cleaning_method']['method'])

    def _convert_to_percentage(self, difference, errors) -> np.ndarray:
        # TODO now i really do not like it
        # TODO it does work with initial error (big error means that everything can be background)
        prob = np.ones(difference.shape, dtype=np.float)
        top_fraction_on_2e = 0.8
        bottom_fraction_on_3e = 0.05

        step_1_pixels = difference <= errors
        step_2_pixels = (errors < difference) & (difference <= 3 * errors)
        step_3_pixels = 3 * errors < difference

        step_1_probs = mathmap.multi_linear_mapping_2d(difference[step_1_pixels],
                                                       lx=0, rx=errors[step_1_pixels],
                                                       ly=1.0, ry=top_fraction_on_2e)

        step_2_probs = mathmap.multi_linear_mapping_2d(difference[step_2_pixels],
                                                       lx=errors[step_2_pixels], rx=3 * errors[step_2_pixels],
                                                       ly=top_fraction_on_2e, ry=bottom_fraction_on_3e)

        step_3_probs = mathmap.multi_linear_mapping_2d(difference[step_3_pixels],
                                                       lx=3 * errors[step_3_pixels], rx=5 * errors[step_3_pixels],
                                                       ly=bottom_fraction_on_3e, ry=0.0)

        prob[step_1_pixels] = step_1_probs
        prob[step_2_pixels] = step_2_probs
        prob[step_3_pixels] = step_3_probs

        prob = np.minimum(np.maximum(prob, 0), 1)
        return prob
