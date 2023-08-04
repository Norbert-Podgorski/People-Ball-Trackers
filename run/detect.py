from typing import Dict, Any

import yaml
from yaml import SafeLoader

from src.detectors.detector_factory import DetectorFactory
from src.frames_loader import FramesLoader
from src.visualizer import Visualizer


def detect(config: Dict[str, Any]):
    frames_loader = FramesLoader(**config["frames_loader_config"])
    detector = DetectorFactory.create(**config["used_detector"])
    visualizer = Visualizer(**config["visualizer_config"])

    frames = frames_loader.load_all_frames()
    detections = detector.detect(frames)
    visualizer.visualize(frames, detections)


if __name__ == "__main__":
    config_path = "../configs/detection_config.yaml"
    with open(config_path) as yaml_file:
        loaded_config = yaml.load(yaml_file, Loader=SafeLoader)
    detect(loaded_config)
