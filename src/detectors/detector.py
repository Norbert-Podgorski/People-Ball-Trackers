from __future__ import annotations

import abc
from typing import NamedTuple, List, Optional

import numpy as np
import torch


class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, images: torch.Tensor) -> List[Detection]:
        pass


class Detection(NamedTuple):
    bounding_box: torch.Tensor
    confidence: float
    detected_class: str
    idx: Optional[int] = None
    size: Optional[np.ndarray] = None
