from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch

from src.detectors.detector import Detector, Detection
from src.dasiam_rpn_net.dasiam import net as dasiam_net
from src.dasiam_rpn_net.dasiam import run_SiamRPN


@dataclass
class DaSiamRPNTrackedObject:
    centers: np.ndarray
    sizes: np.ndarray

# class DaSiamRPNDetectorStrategy(ABC):
#     def __init__(self, dasiam_rpn_model: dasiam_net.SiamRPNvot, states: List[Dict[str, Any]]):
#         self.model: dasiam_net.SiamRPNvot = dasiam_rpn_model
#         self.states: List[Dict[str, Any]] = states
#
#     @abstractmethod
#     def dasiam_rpn_step(self, tracked_balls: TrackedBalls, scene: SceneData, camera_idx: int) -> None:
#         pass
#
#     @abstractmethod
#     def is_condition_satisfied(self, tracked_balls: TrackedBalls, camera_idx: int) -> bool:
#         pass
#
#     @staticmethod
#     def _get_image_size_wh(image_tensor: torch.Tensor) -> np.ndarray:
#         image_size_hw = np.array(image_tensor.shape[1:])
#         image_size_wh = image_size_hw[::-1]
#         return image_size_wh
#
#     @staticmethod
#     def _is_ball_found_by_base_tracker(balls_centers: np.ndarray, camera_idx: int) -> bool:
#         ball_center = balls_centers[camera_idx]
#         ball_center_contain_nans = np.any(np.isnan(ball_center))
#         return not ball_center_contain_nans
#
#
# class UpdatingDaSiamRPNStateStrategy(DaSiamRPNDetectorStrategy):
#     def dasiam_rpn_step(self, tracked_balls: TrackedBalls, scene: SceneData, camera_idx: int) -> None:
#         image = convert_tensor_image_to_numpy_bgr(scene.images[camera_idx])
#
#         image_size = self._get_image_size_wh(image_tensor=scene.images[camera_idx])
#         camera_resolution = scene.cameras.resolution[camera_idx].cpu().numpy()
#
#         scaling_factor = image_size / camera_resolution
#         center_position = tracked_balls.centers[camera_idx] * scaling_factor
#         bbox_size = tracked_balls.sizes[camera_idx] * scaling_factor
#
#         new_state = run_SiamRPN.SiamRPN_init(image, center_position, bbox_size, self.model)
#         self.states[camera_idx] = new_state
#
#     def is_condition_satisfied(self, tracked_balls: TrackedBalls, camera_idx: int) -> bool:
#         ball_found_by_base_tracker = self._is_ball_found_by_base_tracker(tracked_balls.centers, camera_idx)
#         return ball_found_by_base_tracker
#
#
# class GettingPredictionFromDaSiamRPNStrategy(DaSiamRPNDetectorStrategy):
#     def __init__(self, dasiam_rpn_model: dasiam_net.SiamRPNvot, states: List[Dict[str, Any]], score_threshold: float):
#         super().__init__(dasiam_rpn_model, states)
#         self.score_threshold: float = score_threshold
#
#     def dasiam_rpn_step(self, tracked_balls: TrackedBalls, scene: SceneData, camera_idx: int) -> None:
#         image = convert_tensor_image_to_numpy_bgr(scene.images[camera_idx])
#         new_state = run_SiamRPN.SiamRPN_track(self.states[camera_idx], image)
#         self._try_fill_missing_ball_detection_with_tracked_ball(tracked_ball_state=new_state, detected_balls=tracked_balls, scene=scene, camera_idx=camera_idx)
#         self.states[camera_idx] = new_state
#
#     def is_condition_satisfied(self, tracked_balls: TrackedBalls, camera_idx: int) -> bool:
#         ball_not_found_by_base_tracker = not self._is_ball_found_by_base_tracker(tracked_balls.centers, camera_idx)
#         ball_found_ever_before = self._is_ball_found_ever_before(camera_idx)
#         return ball_not_found_by_base_tracker and ball_found_ever_before
#
#     def _try_fill_missing_ball_detection_with_tracked_ball(
#         self, tracked_ball_state: Dict[str, Any], detected_balls: TrackedBalls, scene: SceneData, camera_idx: int
#     ) -> None:
#         image_size = self._get_image_size_wh(image_tensor=scene.images[camera_idx])
#         camera_resolution = scene.cameras.resolution[camera_idx].cpu().numpy()
#         if tracked_ball_state["score"] > self.score_threshold:
#             scaling_factor = camera_resolution / image_size
#             center_position = tracked_ball_state["target_pos"] * scaling_factor
#             bbox_size = tracked_ball_state["target_sz"] * scaling_factor
#             detected_balls.centers[camera_idx] = center_position
#             detected_balls.sizes[camera_idx] = bbox_size
#
#     def _is_ball_found_ever_before(self, camera_idx: int) -> bool:
#         return self.states[camera_idx] is not None

class DaSiamRPNDetector(Detector):
    def __init__(self, base_detector: Detector, path: str):
        self.base_detector = base_detector
        self.ball_model: dasiam_net.SiamRPNvot = dasiam_net.SiamRPNvot()
        self.ball_model.load_state_dict(torch.load(path))
        self.people_model: dasiam_net.SiamRPNvot = dasiam_net.SiamRPNvot()
        self.people_model.load_state_dict(torch.load(path))


    def detect(self, image: torch.Tensor) -> List[Detection]:
        base_detector_detections = self.base_detector.detect(image)
        ball_detections, people_detections = self._separate_detections(base_detector_detections)
        print("\nBall Detetctions: ")
        if ball_detections:
            for detection in ball_detections:
                if detection:
                    print(detection.bounding_box)
        else:
            print(ball_detections)
        # ball_detections = self._convert_detections_to_dasiam_tracked_objects(ball_detections)


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

    # def _convert_detections_to_dasiam_tracked_objects(self, detections: List[Detection]):
