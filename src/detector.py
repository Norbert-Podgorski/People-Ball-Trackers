import abc
from typing import NamedTuple

import numpy as np
import torch


class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, images: torch.Tensor) -> torch.Tensor:
        pass


class Detection(NamedTuple):
    bounding_box: np.ndarray
    detected_class: str
    confidence: float
