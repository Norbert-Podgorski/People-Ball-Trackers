from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

from src.detectors.detector import Detector, Detection


class AlgorithmicDetector(Detector):
    def __init__(self, base_detector: Detector, ball_trust_area_scale: float, people_trust_area_scale: float):
        self.base_detector = base_detector
        self.trust_area_scales: Dict[str, float] = {
            "ball": ball_trust_area_scale,
            "person": people_trust_area_scale
        }
        self.last_detections: Dict[str, List[Detection]] = {"ball": [], "person": []}

    def detect(self, images: torch.Tensor) -> List[Detection]:
        base_detector_detections = self.base_detector.detect(images)
        ball_detections, people_detections = self._separate_detections(base_detector_detections)
        tracked_detections = []
        new_last_detections = {"ball": [], "person": []}
        if not (self.last_detections["ball"] and self.last_detections["person"]):
            self._save_first_detections(ball_detections, people_detections)
            return base_detector_detections
        else:
            for detection in ball_detections:
                tracked_detection = self._track_detection(detection)
                if tracked_detection:
                    tracked_detections.append(tracked_detection)
                    new_last_detections["ball"].append(tracked_detection)
            for detection in people_detections:
                tracked_detection = self._track_detection(detection)
                if tracked_detection:
                    tracked_detections.append(tracked_detection)
                    new_last_detections["person"].append(tracked_detection)
        return tracked_detections

    @staticmethod
    def _separate_detections(detections: List[Detection]) -> Tuple[List[Detection], List[Detection]]:
        ball_detections = []
        people_detections = []
        for detection in detections:
            if detection.detected_class == "ball":
                ball_detections.append(detection)
            else:
                people_detections.append(detection)
        return ball_detections, people_detections

    def _save_first_detections(self, ball_detections: List[Detection], people_detections: List[Detection]) -> None:
        for ball_detection_idx, ball_detection in enumerate(ball_detections):
            self.last_detections["ball"].append(
                Detection(
                    bounding_box=ball_detection.bounding_box,
                    confidence=ball_detection.confidence,
                    detected_class=ball_detection.detected_class,
                    idx=ball_detection_idx,
                    size=self._calculate_box_size(ball_detection.bounding_box)
                )
            )
        for person_detection_idx, person_detection in enumerate(people_detections):
            self.last_detections["person"].append(
                Detection(
                    bounding_box=person_detection.bounding_box,
                    confidence=person_detection.confidence,
                    detected_class=person_detection.detected_class,
                    idx=person_detection_idx,
                    size=self._calculate_box_size(person_detection.bounding_box)
                )
            )

    @staticmethod
    def _calculate_box_size(bounding_box: torch.Tensor) -> torch.Tensor:
        size_x = bounding_box[1][0] - bounding_box[0][0]
        size_y = bounding_box[1][1] - bounding_box[0][1]
        return torch.tensor([size_x, size_y])

    def _track_detection(self, detection: Detection) -> Optional[Detection]:
        detection.size = self._calculate_box_size(detection.bounding_box)
        nearest_detection = self._find_nearest_detection(detection)
        if nearest_detection:
            updated_detection = Detection(
                bounding_box=detection.bounding_box,
                confidence=detection.confidence,
                detected_class=detection.detected_class,
                idx=nearest_detection.idx,
                size=detection.size
            )
            return updated_detection

    def _find_nearest_detection(self, detection: Detection) -> Optional[Detection]:
        min_distance = None
        nearest_detection = None
        acceptable_distance = torch.min(detection.size).item() * self.trust_area_scales[detection.detected_class]
        for last_detection in self.last_detections[detection.detected_class]:
            first_center_x = np.mean([detection.bounding_box[1][0], detection.bounding_box[0][0]])
            first_center_y = np.mean([detection.bounding_box[1][1], detection.bounding_box[0][1]])
            second_center_x = np.mean([last_detection.bounding_box[1][0], last_detection.bounding_box[0][0]])
            second_center_y = np.mean([last_detection.bounding_box[1][1], last_detection.bounding_box[0][1]])
            distance_x = abs(first_center_x - second_center_x)
            distance_y = abs(first_center_y - second_center_y)
            if distance_x < acceptable_distance and distance_y < acceptable_distance:
                lower_distance = min([distance_x, distance_y])
                if min_distance is None:
                    min_distance = lower_distance
                    nearest_detection = last_detection
                elif lower_distance < min_distance:
                    min_distance = lower_distance
                    nearest_detection = last_detection
        return nearest_detection
