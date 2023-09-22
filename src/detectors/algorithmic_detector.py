from typing import List

import torch

from src.detectors.detector import Detector, Detection


class AlgorithmicDetector(Detector):
    def __init__(self, base_detector: Detector, distance_to_players: int, fps: int):
        self.base_detector = base_detector
        self.distance_to_players = distance_to_players
        self.fps = fps

    def detect(self, images: torch.Tensor) -> List[List[Detection]]:
        pass