from typing import Dict, Any, List

import torch
from torch import nn

from src.detectors.detector import Detector, Detection


class PretrainedYOLODetector(Detector):
    def __init__(self, torchhub_yolo_config: Dict[str, Any], selected_classes: Dict[str, Dict[str, Any]]):
        self.yolo = self._create_torchub_yolo(torchhub_yolo_config)
        self.selected_classes = selected_classes
        self.yolo_class_to_name = self._create_yolo_class_to_name_dict(selected_classes)

    @staticmethod
    def _create_torchub_yolo(torchhub_yolo_config: Dict[str, Any]) -> nn.Module:
        yolo = torch.hub.load(**torchhub_yolo_config).eval()
        return yolo

    @staticmethod
    def _create_yolo_class_to_name_dict(selected_classes: Dict[str, Dict[str, Any]]) -> Dict[int, str]:
        class_to_name_dict = {selected_classes[name]["class_id"]: name for name in selected_classes}
        return class_to_name_dict

    def detect(self, images: torch.Tensor) -> List[List[Detection]]:
        # import straight from yolo repo, it must be here
        from utils.general import non_max_suppression

        yolo_outputs = self.yolo(images)
        yolo_outputs_for_all_scenes = non_max_suppression(yolo_outputs)
        return self._yolo_output_to_valid_representation_for_all_scenes(yolo_outputs_for_all_scenes)

    def _yolo_output_to_valid_representation_for_all_scenes(self, yolo_outputs_for_all_scenes: torch.Tensor) -> List[List[Detection]]:
        return [
            self._yolo_output_to_valid_representation_for_single_scene(yolo_outputs_for_single_scene)
            for yolo_outputs_for_single_scene in yolo_outputs_for_all_scenes
        ]

    def _yolo_output_to_valid_representation_for_single_scene(self, yolo_outputs_for_single_scenes: torch.Tensor) -> List[Detection]:
        valid_yolo_detections = [
            self._detection_from_torchub_yolo_output(yolo_output)
            for yolo_output in yolo_outputs_for_single_scenes
            if self._is_yolo_detection_valid(yolo_output)
        ]
        return valid_yolo_detections

    def _detection_from_torchub_yolo_output(self, yolo_output: torch.Tensor) -> Detection:
        yolo_output_bbox_indexes = slice(4)
        yolo_output_confidence_index = 4
        yolo_output_detected_class_index = 5
        min_x, min_y, max_x, max_y = yolo_output[yolo_output_bbox_indexes]
        bounding_box_coordinates = torch.tensor([[min_x, min_y], [max_x, max_y]])
        detected_class = self.yolo_class_to_name.get(yolo_output[yolo_output_detected_class_index].item())
        confidence = yolo_output[yolo_output_confidence_index].item()
        return Detection(bounding_box=bounding_box_coordinates, detected_class=detected_class, confidence=confidence)

    def _is_yolo_detection_valid(self, detection: Detection) -> bool:
        detection_confidence_index = 4
        confidence = detection[detection_confidence_index].item()
        detection_class_index = 5
        detected_class = self.yolo_class_to_name.get(detection[detection_class_index].item())

        invalid_class = detected_class not in self.selected_classes
        if invalid_class:
            return False
        is_confidence_sufficient = confidence >= self.selected_classes[detected_class]["confidence_threshold"]

        return is_confidence_sufficient
