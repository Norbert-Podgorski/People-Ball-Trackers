from __future__ import annotations

import abc
from typing import NamedTuple, List, Optional

import torch


class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, images: torch.Tensor) -> List[List[Detection]]:
        pass


class Detection(NamedTuple):
    bounding_box: torch.Tensor
    detected_class: str
    confidence: float
    idx: Optional[int] = None
