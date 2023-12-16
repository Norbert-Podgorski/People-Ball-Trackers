from typing import Dict, Callable, Any

import yaml
from yaml import SafeLoader

from src.detectors.algorithmic_detector import AlgorithmicDetector
from src.detectors.dasiam_rpn_detector import DaSiamRPNDetector
from src.detectors.detector import Detector
from src.detectors.yolo_detector import YOLODetector


def create_yolo_detector(**detector_config: Dict[str, Any]) -> Detector:
    return YOLODetector(**detector_config)

def create_algorithmic_detector(base_detector_config_path: str, **detector_config: Dict[str, Any]) -> Detector:
    with open(base_detector_config_path) as detector_config_file:
        base_detector_config = yaml.load(detector_config_file, Loader=SafeLoader)
    base_detector = DetectorFactory.create(**base_detector_config)
    return AlgorithmicDetector(base_detector, **detector_config)

def create_dasiam_rpn_detector(base_detector_config_path: str, **detector_config: Dict[str, Any]) -> Detector:
    with open(base_detector_config_path) as detector_config_file:
        base_detector_config = yaml.load(detector_config_file, Loader=SafeLoader)
    base_detector = DetectorFactory.create(**base_detector_config)
    return DaSiamRPNDetector(base_detector, **detector_config)

DETECTOR_CREATORS: Dict[str, Callable] = {
    "YOLODetector": create_yolo_detector,
    "AlgorithmicDetector": create_algorithmic_detector,
    "DaSiamRPNDetector": create_dasiam_rpn_detector
}


class DetectorFactory:
    @staticmethod
    def create(name: str, **detector_config: Dict[str, Any]) -> Detector:
        created_detector = DETECTOR_CREATORS[name](**detector_config)
        return created_detector
