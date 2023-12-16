from typing import List

import torch

from src.detectors.detector import Detector, Detection
from src.dasiam_rpn_net.dasiam import net as dasiam_net
from src.dasiam_rpn_net.dasiam import run_SiamRPN

class DaSiamRPNDetector(Detector):
    def __init__(self, base_detector: Detector, path: str):
        self.base_detector = base_detector
        self.model: dasiam_net.SiamRPNvot = dasiam_net.SiamRPNvot()
        self.model.load_state_dict(torch.load(path))


    def detect(self, images: torch.Tensor) -> List[Detection]:
        print(self.model.state_dict())
