from __future__ import annotations

import abc
from typing import List, Optional
import dataclasses
import numpy as np
import torch


class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, images: torch.Tensor) -> List[Detection]:
        pass


@dataclasses.dataclass
class Detection:
    def __init__(
        self,
        bounding_box: torch.Tensor,
        confidence: float,
        detected_class: str,
        idx: Optional[int] = None,
        size: Optional[torch.Tensor] = None,
        last_frame_number: Optional[int] = None
    ):
        self.bounding_box = bounding_box
        self.confidence = confidence
        self.detected_class = detected_class
        self.size = size
        self.idx = idx
        self.last_frame_number = last_frame_number
