from typing import Dict, Callable, Any

from src.detectors.detector import Detector
from src.detectors.pretrained_yolo_detector import PretrainedYOLODetector


def create_pretrained_yolo_detector(**detector_config: Dict[str, Any]) -> Detector:
    return PretrainedYOLODetector(**detector_config["pretrained_yolo_detector_config"])


DETECTOR_CREATORS: Dict[str, Callable] = {
    "PretrainedYOLODetector": create_pretrained_yolo_detector,
}


class DetectorFactory:
    @staticmethod
    def create(name: str, **detector_config: Dict[str, Any]) -> Detector:
        created_detector = DETECTOR_CREATORS[name](**detector_config)
        return created_detector
