from typing import Dict, Any

import numpy as np
import yaml
from yaml import SafeLoader

from src.detector_factory import DetectorFactory
from src.frames_loader import FramesLoader


def detect(config: Dict[str, Any]):
    frames_loader = FramesLoader(**config["frames_loader_config"])
    detector = DetectorFactory.create(**config["used_detector"])

    frames = frames_loader.load_subset_frames(subset=5)
    detections = detector.detect(frames)
    print(detections)


if __name__ == "__main__":
    config_path = "../configs/detection_config.yaml"
    with open(config_path) as yaml_file:
        config = yaml.load(yaml_file, Loader=SafeLoader)
    detect(config)
