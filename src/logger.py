from typing import List, Tuple

from src.detectors.detector import Detection

import os


class Logger:
    def __init__(self, path: str):
        self.path = path

    def log(self, detections: List[List[Detection]], frames_number: int) -> None:
        self._create_path_if_not_exists()
        detected_people_number, detected_balls_number = self._count_detections(detections)
        with open(self.path + "/info.txt", 'w') as file:
            file.write(f"The recording contains {frames_number} scenes.\n")
            file.write(f"Each shows two people and one ball.\n")
            file.write(f"Detected: \n")
            file.write(f"- {detected_people_number} people.\n")
            file.write(f"- {detected_balls_number} balls.")

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

    def _create_path_if_not_exists(self) -> None:
        if not os.path.exists(self.path):
            os.makedirs(self.path)
