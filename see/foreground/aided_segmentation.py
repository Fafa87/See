from abc import abstractmethod, ABC

import numpy as np

import vendor.SEP.sep as sep


class AidedSegmentation(ABC):
    @abstractmethod
    def update(self, image, rough_foreground_mask):
        pass

    @abstractmethod
    def calc_prob(self, image) -> np.ndarray:
        pass


def run_on_video(video_path, aided_segmentator: AidedSegmentation, output_path):
    video_loader = MovieLoader