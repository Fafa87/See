import cv2
import numpy as np
import numpy.testing as nptest

import see.foreground
import see.foreground.backgrounds as backgrounds
import tests.testbase


class TestForegroundFinder(tests.testbase.TestBase):
    def setUp(self):
        super().setUp()
        self.background = np.ones((50, 50, 3), dtype=np.uint8) * 120
        self.foreground = np.zeros_like(self.background)
        self.foreground[22:26, 10:24][0] = 3
        self.foreground[22:26, 10:24][1] = 55
        self.foreground[22:26, 10:24][2] = 129
        self.foreground_mask = np.zeros(self.background.shape[:2], dtype=np.bool)
        self.foreground_mask[22:26, 10:24] = 1

    def noise(self, image, max_noise):
        return ((np.random.random(image.shape) * max_noise * 2 - max_noise) + image).astype(np.uint8)

    def test_clean_image(self):
        finder = see.foreground.ForegroundFinder.create_from_dicts(cleaning={'method': 'median', 'size': 5})
        noisy_background = ((np.random.random(self.background.shape) * 10 - 5) + self.background).astype(np.uint8)
        cleaned_noise = finder._clean_image(noisy_background)
        noisy_diff = cv2.absdiff(noisy_background, self.background).mean()
        cleaned_diff = cv2.absdiff(cleaned_noise, self.background).mean()

        self.assertTrue(cleaned_diff * 2 < noisy_diff)

    def test_confident_background(self):
        finder = see.foreground.ForegroundFinder.create_from_dicts(confident_size=3)
        confident_background_mask = finder._get_confident_background(self.background, self.foreground_mask)
        self.assertTrue(confident_background_mask.sum() < (self.foreground_mask == 0).sum())

    def test_convert_to_probs(self):
        finder = see.foreground.ForegroundFinder.create_from_dicts()
        diffs = np.array([0, 1, 1, 2, 3, 4, 3, 6, 7, 8], dtype=np.uint8)
        errors = np.array([0, 1, 0, 0, 1, 1, 2, 2, 3, 255], dtype=np.uint8)
        prob = finder._convert_to_percentage(diffs, errors=errors)

        expected = np.array([1, 0.8, 0.0, 0.0, 0.05, 0.03, 0.61, 0.05, 0.3, 0.99])
        nptest.assert_almost_equal(prob, expected, decimal=2)

    def test_finder_no_object(self):
        background_model = backgrounds.StaticBackgroundModel(update_inertia=1.0, error_inertia=1.0, diff_method='rgb')
        finder = see.foreground.ForegroundFinder(background_model,
                                                 cleaning={'method': 'median', 'size': 5}, confident_size=3)

        noisy_background = self.noise(self.background, 10)
        finder.update(noisy_background, np.zeros_like(self.foreground_mask))
        prob_1 = finder.calc_prob(noisy_background)
        nptest.assert_array_less(prob_1, 0.01)

        noisy_background = self.noise(self.background, 10)
        finder.update(noisy_background, np.zeros_like(self.foreground_mask))
        prob_2 = finder.calc_prob(noisy_background)
        nptest.assert_array_less(prob_2.mean(), 0.23)

        noisy_background = self.noise(self.background, 10)
        finder.update(noisy_background, np.zeros_like(self.foreground_mask))
        prob_3 = finder.calc_prob(noisy_background)
        nptest.assert_array_less(prob_3.mean(), 0.21)

        noisy_background = self.noise(self.background, 10)
        finder.update(noisy_background, np.zeros_like(self.foreground_mask))
        prob_4 = finder.calc_prob(noisy_background)
        nptest.assert_array_less(prob_4.mean(), 0.21)

        noisy_background = self.noise(self.background, 10)
        finder.update(noisy_background, np.zeros_like(self.foreground_mask))
        prob_5 = finder.calc_prob(noisy_background)
        nptest.assert_array_less(prob_5.mean(), 0.21)

    def test_finder_for_unknown_area(self):
        # check what is the probability of unknown area
        # and what happens when it is first discovered
        # error should be mean of all pixels?
        background_model = backgrounds.StaticBackgroundModel(update_inertia=1.0, error_inertia=1.0, diff_method='rgb')
        finder = see.foreground.ForegroundFinder(background_model,
                                                 cleaning={'method': 'median', 'size': 5}, confident_size=1)

        finder.update(self.background, self.foreground_mask)

        noisy_image = self.noise(self.background, 2)
        prob = finder.calc_prob(noisy_image)

        prob_unknown = prob[self.foreground_mask]
        prob_background = prob[self.foreground_mask == 0]

        nptest.assert_equal(prob_unknown.mean(), 0.5)
        nptest.assert_array_less(prob_background.mean(), 0.2)
