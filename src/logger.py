from typing import List, Tuple, Dict

from src.detectors.detector import Detection

import os


class Logger:
    def __init__(self, path: str):
        self.path = path

    def log(self, detections: List[List[Detection]], frames_number: int) -> None:
        self._create_path_if_not_exists()
        if detections[0][0].idx is None:
            self._log_yolo_detections(detections, frames_number)
        else:
            self._log_tracked_detections(detections, frames_number)


    def _log_yolo_detections(self, detections: List[List[Detection]], frames_number: int):
        detected_people_number, detected_balls_number = self._count_detections(detections)
        with open(self.path + "/info.txt", 'w') as file:
            file.write(f"The recording contains {frames_number} scenes.\n")
            file.write(f"Each shows two people and one ball.\n")
            file.write(f"Detected: \n")
            file.write(f"- {detected_people_number} people.\n")
            file.write(f"- {detected_balls_number} balls.")

    def _log_tracked_detections(self, detections: List[List[Detection]], frames_number: int):
        detected_people_numbers, detected_balls_numbers = self._count_tracked_detections(detections)
        with open(self.path + "/info.txt", 'w') as file:
            file.write(f"The recording contains {frames_number} scenes.\n")
            file.write(f"Each shows two people and one ball.\n")
            file.write(f"Detected: \n")
            file.write(f"People: \n")
            for idx in list(detected_people_numbers.keys()):
                file.write(f"- {idx}: {detected_people_numbers[idx]}\n")
            file.write(f"Balls: \n")
            for idx in list(detected_balls_numbers.keys()):
                file.write(f"- {idx}: {detected_balls_numbers[idx]}\n")

    @staticmethod
    def _count_detections(detections: List[List[Detection]]) -> Tuple[int, int]:
        detected_people_number = 0
        detected_balls_number = 0
        for detections_for_scene in detections:
            for detection in detections_for_scene:
                if detection.detected_class == "person":
                    detected_people_number += 1
                else:
                    detected_balls_number += 1
        return detected_people_number, detected_balls_number

    @staticmethod
    def _count_tracked_detections(detections: List[List[Detection]]) -> Tuple[Dict[int, int], Dict[int, int]]:
        detected_people_numbers = {}
        detected_balls_numbers = {}
        for detections_for_scene in detections:
            for detection in detections_for_scene:
                if detection.detected_class == "person":
                    if detection.idx not in detected_people_numbers.keys():
                        detected_people_numbers[detection.idx] = 0
                    detected_people_numbers[detection.idx] += 1
                else:
                    if detection.idx not in detected_balls_numbers.keys():
                        detected_balls_numbers[detection.idx] = 0
                    detected_balls_numbers[detection.idx] += 1
        return detected_people_numbers, detected_balls_numbers

    def _create_path_if_not_exists(self) -> None:
        if not os.path.exists(self.path):
            os.makedirs(self.path)
