import numpy as np
import numpy.testing as nptest

import see.foreground.backgrounds
import tests.testbase


class TestStaticBackgroundModel(tests.testbase.TestBase):
    def setUp(self):
        super().setUp()
        self.background = np.ones((50, 50, 3), dtype=np.uint8) * 120
        self.foreground = np.zeros_like(self.background)
        self.foreground[22:26, 10:24][0] = 3
        self.foreground[22:26, 10:24][1] = 55
        self.foreground[22:26, 10:24][2] = 129
        self.foreground_mask = np.zeros(self.background.shape[:2], dtype=np.bool)
        self.foreground_mask[22:26, 10:24] = 1

    def test_update_background(self):
        model = see.foreground.backgrounds.StaticBackgroundModel(update_inertia=1.0, error_inertia=1.0,
                                                                 diff_method='rgb')
        self.assertIsNone(model.get())  # nothing yet

        # all is background
        model.update(self.background, np.invert(np.zeros_like(self.foreground_mask)))
        model_background = model.get()
        nptest.assert_equal(model_background, self.background)

        # update other but not foreground
        shifted_background = (self.background + (10, 20, 7)).astype(np.uint8)
        model.update(shifted_background, np.invert(self.foreground_mask))
        model_background = model.get()
        self.np_assert_not_equal(model_background[self.foreground_mask == 0],
                                 self.background[self.foreground_mask == 0])
        self.np_assert_equal(model_background[self.foreground_mask != 0],
                             self.background[self.foreground_mask != 0])

    def test_never_seen_background(self):
        model = see.foreground.backgrounds.StaticBackgroundModel(update_inertia=1.0, error_inertia=1.0,
                                                                 diff_method='rgb')
        model.update(self.background, np.invert(self.foreground_mask))

        model_background = model.get()
        nptest.assert_equal(model_background[self.foreground_mask == 0], self.background[self.foreground_mask == 0])
        nptest.assert_equal(model_background[self.foreground_mask != 0], 128)

    def test_get_all_background_info(self):
        model = see.foreground.backgrounds.StaticBackgroundModel(update_inertia=1.0, error_inertia=1.0,
                                                                 diff_method='rgb')
        model.update(self.background, np.invert(self.foreground_mask))
        background_mask = self.foreground_mask == 0

        model_details = model.get_details()
        self.assertIn('background', model_details)
        self.assertIn('error', model_details)
        self.assertIn('mask', model_details)
        nptest.assert_equal(model_details['background'][background_mask], self.background[background_mask])
        nptest.assert_equal(model_details['error'][self.foreground_mask == 0], 1)  # we assume it is never perfect
        nptest.assert_equal(model_details['error'][self.foreground_mask != 0], 64)
        nptest.assert_equal(model_details['mask'], self.foreground_mask == 0)

    def test_diff_rgb(self):
        pass

    def test_background_and_error_progression(self):
        pass
