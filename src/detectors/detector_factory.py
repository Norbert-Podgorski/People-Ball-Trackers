from typing import Dict, Callable, Any

from src.detectors.detector import Detector
from src.detectors.yolo_detector import YOLODetector


def create_yolo_detector(**detector_config: Dict[str, Any]) -> Detector:
    return YOLODetector(**detector_config["yolo_detector_config"])


DETECTOR_CREATORS: Dict[str, Callable] = {
    "YOLODetector": create_yolo_detector,
}


class DetectorFactory:
    @staticmethod
    def create(name: str, **detector_config: Dict[str, Any]) -> Detector:
        created_detector = DETECTOR_CREATORS[name](**detector_config)
        return created_detector
