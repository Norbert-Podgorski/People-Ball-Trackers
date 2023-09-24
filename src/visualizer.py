import os
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch

from src.detectors.detector import Detection

VISUALIZATION_COLORS: Dict[str, Dict[int, Tuple[int, int, int]]] = {
    "ball":   {
        0: (255, 0, 0),
        1: (150, 0, 250),
        None: (255, 0, 0)
    },
    "person": {
        0: (0, 255, 0),
        1: (0, 0, 255),
        2: (0, 255, 255,),
        None: (0, 255, 0)
    }
}


class Visualizer:
    def __init__(self, frames_path: str, video_path: str, fps: int, size: List[int]):
        self.frames_path = frames_path
        self._create_path_if_not_exists(self.frames_path)
        self._create_path_if_not_exists(video_path)
        self.writer = cv2.VideoWriter(f'{video_path}/video.avi', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def visualize(self, frames: torch.Tensor, detections: List[List[Detection]]):
        for frame_idx, (frame, detections_for_frame) in enumerate(zip(frames, detections)):
            frame = self._prepare_frame_to_save(frame)
            for detection in detections_for_frame:
                bbox = detection.bounding_box.to(torch.int32)
                start_point = (bbox[0][0].item(), bbox[0][1].item())
                end_point = (bbox[1][0].item(), bbox[1][1].item())
                color = VISUALIZATION_COLORS[detection.detected_class][detection.idx]
                thickness = 5
                cv2.rectangle(frame, start_point, end_point, color, thickness)
            cv2.imwrite(f'{self.frames_path}/{frame_idx}.jpg', frame)
            self.writer.write(frame)
        self.writer.release()

    @staticmethod
    def _prepare_frame_to_save(frame: torch.Tensor) -> np.ndarray:
        frame = np.transpose((frame.numpy() * 255).astype(np.uint8), (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        return frame

    @staticmethod
    def _create_path_if_not_exists(path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
