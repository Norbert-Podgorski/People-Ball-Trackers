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
        self.frame_number = 0

    def detect(self, image: torch.Tensor) -> List[Detection]:
        base_detector_detections = self.base_detector.detect(image)
        ball_detections, people_detections = self._separate_detections(base_detector_detections)
        tracked_detections = []
        new_last_detections = {"ball": [], "person": []}
        new_last_detections_idx =  {"ball": [], "person": []}
        if not self.last_detections["ball"] and not self.last_detections["person"]:
            tracked_detections = self._save_first_detections(ball_detections, people_detections)
            self.frame_number += 1
            return tracked_detections
        else:
            for detection in ball_detections:
                tracked_detection = self._track_detection(detection)
                if tracked_detection:
                    tracked_detections.append(tracked_detection)
                    new_last_detections_idx["ball"].append(tracked_detection.idx)
                    new_last_detections["ball"].append(tracked_detection)
            for detection in people_detections:
                tracked_detection = self._track_detection(detection)
                if tracked_detection:
                    tracked_detections.append(tracked_detection)
                    new_last_detections_idx["person"].append(tracked_detection.idx)
                    new_last_detections["person"].append(tracked_detection)

        for key in ["ball", "person"]:
            for detection in self.last_detections[key]:
                if detection.idx not in new_last_detections_idx[key]:
                    new_last_detections[key].append(detection)

        self.last_detections = new_last_detections
        self.frame_number += 1
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

    def _save_first_detections(self, ball_detections: List[Detection], people_detections: List[Detection]) -> List[Detection]:
        detections = []
        for ball_detection_idx, ball_detection in enumerate(ball_detections):
            detection = Detection(
                bounding_box=ball_detection.bounding_box,
                confidence=ball_detection.confidence,
                detected_class=ball_detection.detected_class,
                size=self._calculate_box_size(ball_detection.bounding_box),
                idx=ball_detection_idx,
                last_frame_number=self.frame_number
                )
            self.last_detections["ball"].append(detection)
            detections.append(detection)
        for person_detection_idx, person_detection in enumerate(people_detections):
            detection = Detection(
                bounding_box=person_detection.bounding_box,
                confidence=person_detection.confidence,
                detected_class=person_detection.detected_class,
                size=self._calculate_box_size(person_detection.bounding_box),
                idx=person_detection_idx,
                last_frame_number=self.frame_number,
            )
            self.last_detections["person"].append(detection)
            detections.append(detection)
        return detections

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
                size=detection.size,
                idx=nearest_detection.idx,
                last_frame_number=self.frame_number
            )
            return updated_detection
        else:
            if detection.confidence >= 0.7:
                idx = self._find_first_available_idx(detection)
                return Detection(
                bounding_box=detection.bounding_box,
                confidence=detection.confidence,
                detected_class=detection.detected_class,
                size=detection.size,
                idx=idx,
                last_frame_number=self.frame_number
            )

    def _find_nearest_detection(self, detection: Detection) -> Optional[Detection]:
        acceptable_distance_detections = []
        for last_detection in self.last_detections[detection.detected_class]:
            acceptable_distance = torch.min(detection.size).item() * self.trust_area_scales[detection.detected_class] * (self.frame_number - last_detection.last_frame_number)
            first_center_x = np.mean([detection.bounding_box[1][0], detection.bounding_box[0][0]])
            first_center_y = np.mean([detection.bounding_box[1][1], detection.bounding_box[0][1]])
            second_center_x = np.mean([last_detection.bounding_box[1][0], last_detection.bounding_box[0][0]])
            second_center_y = np.mean([last_detection.bounding_box[1][1], last_detection.bounding_box[0][1]])
            distance_x = abs(first_center_x - second_center_x)
            distance_y = abs(first_center_y - second_center_y)
            if distance_x < acceptable_distance and distance_y < acceptable_distance:
                acceptable_distance_detections.append(last_detection)
        most_accurate_detection = self._find_most_accurate_detection(detection, acceptable_distance_detections)
        return most_accurate_detection

    @staticmethod
    def _find_most_accurate_detection(detection: Detection, acceptable_distance_detections: List[Detection]) -> Detection:
        detection_with_minimum_maximum_size_difference = None
        minimum_maximum_size_difference = None
        for proposed_detection in acceptable_distance_detections:
            x_size_diff = abs(proposed_detection.size[0] - detection.size[0])
            y_size_diff = abs(proposed_detection.size[1] - detection.size[1])
            max_diff = x_size_diff if x_size_diff >= y_size_diff else y_size_diff
            if minimum_maximum_size_difference is None:
                minimum_maximum_size_difference = max_diff
                detection_with_minimum_maximum_size_difference = proposed_detection
            elif max_diff < minimum_maximum_size_difference:
                minimum_maximum_size_difference = max_diff
                detection_with_minimum_maximum_size_difference = proposed_detection
        return detection_with_minimum_maximum_size_difference

    def _find_first_available_idx(self, detection: Detection) -> int:
        used_idx = []
        for detection in self.last_detections[detection.detected_class]:
            used_idx.append(detection.idx)
        idx = 0
        while True:
            if idx not in used_idx:
                return idx
            idx += 1


