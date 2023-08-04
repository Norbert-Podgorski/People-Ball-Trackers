from typing import Dict, Any

import cv2
import numpy as np
import torch
import yaml
from yaml import SafeLoader

from src.detector_factory import DetectorFactory
from src.frames_loader import FramesLoader


def detect(config: Dict[str, Any]):
    frames_loader = FramesLoader(**config["frames_loader_config"])
    detector = DetectorFactory.create(**config["used_detector"])

    frames = frames_loader.load_subset_frames(subset=5)
    detections = detector.detect(frames)
    for frame, detections_for_frame in zip(frames, detections):
        frame = np.transpose((frame.numpy() * 255).astype(np.uint8), (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        for detection in detections_for_frame:
            bbox = detection.bounding_box.to(torch.int32)
            start_point = (bbox[0][0].item(), bbox[0][1].item())
            end_point = (bbox[1][0].item(), bbox[1][1].item())
            cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 3)
            cv2.imwrite("C:/Users/Norbertson/Desktop/Praca Magisterska/People-Ball-Trackers/example_data/a.jpg", frame)
            return



if __name__ == "__main__":
    config_path = "../configs/detection_config.yaml"
    with open(config_path) as yaml_file:
        loaded_config = yaml.load(yaml_file, Loader=SafeLoader)
    detect(loaded_config)
