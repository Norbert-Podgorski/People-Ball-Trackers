from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import cv2

from src.detectors.detector import Detector, Detection
from src.dasiam_rpn_net.dasiam import net as dasiam_net
from src.dasiam_rpn_net.dasiam import run_SiamRPN


class DaSiamRPNTrackingStrategy(ABC):
    STATE: Dict[str, Any] = {}

    def __init__(self, dasiam_rpn_model: dasiam_net.SiamRPNBIG):
        self.model: dasiam_net.SiamRPNBIG = dasiam_rpn_model

    @abstractmethod
    def dasiam_rpn_step(self, detected_ball: Detection, image: torch.Tensor) -> Detection:
        pass

    @abstractmethod
    def is_condition_satisfied(self, detected_ball: Detection) -> bool:
        pass

    def convert_tensor_image_to_numpy_bgr(self, image: torch.Tensor) -> np.ndarray:
        image_np_hwc = self.convert_tensor_image_chw_to_numpy_hwc(image)
        image_bgr = cv2.cvtColor(image_np_hwc, cv2.COLOR_RGB2BGR)
        return image_bgr

    def convert_tensor_image_chw_to_numpy_hwc(self, image: torch.Tensor) -> np.ndarray:
        image_np = np.round(image.cpu().numpy() * 255).astype("uint8")
        return self.convert_from_chw_to_hwc(image_np)

    @staticmethod
    def convert_from_chw_to_hwc(image: np.ndarray) -> np.ndarray:
        channel_dim = -3
        height_dim = -2
        width_dim = -1
        converted_image = np.moveaxis(image, (height_dim, width_dim, channel_dim), (-3, -2, -1))
        return converted_image

    @staticmethod
    def _is_ball_found_by_base_tracker(detected_ball: Detection) -> bool:
        return detected_ball is not None


class UpdatingDaSiamRPNStateStrategy(DaSiamRPNTrackingStrategy):
    def dasiam_rpn_step(self, detected_ball: Detection, image: torch.Tensor) -> Detection:
        image_np = self.convert_tensor_image_to_numpy_bgr(image)

        center_x = (detected_ball.bounding_box[0][0] + detected_ball.bounding_box[1][0]) / 2
        center_y = (detected_ball.bounding_box[0][1] + detected_ball.bounding_box[1][1]) / 2
        bbox_center = np.array([center_x, center_y])
        bbox_size = detected_ball.size.numpy()

        new_state = run_SiamRPN.SiamRPN_init(image_np, bbox_center, bbox_size, self.model)
        DaSiamRPNTrackingStrategy.STATE = new_state

        return detected_ball

    def is_condition_satisfied(self, detected_ball: Detection) -> bool:
        ball_found_by_base_tracker = self._is_ball_found_by_base_tracker(detected_ball)
        return ball_found_by_base_tracker


class GettingPredictionFromDaSiamRPNStrategy(DaSiamRPNTrackingStrategy):
    def __init__(self, dasiam_rpn_model: dasiam_net.SiamRPNBIG, score_threshold: float):
        super().__init__(dasiam_rpn_model)
        self.score_threshold: float = score_threshold

    def dasiam_rpn_step(self, detected_ball: Detection, image: torch.Tensor) -> Detection:
        image_np = self.convert_tensor_image_to_numpy_bgr(image)
        new_state = run_SiamRPN.SiamRPN_track(DaSiamRPNTrackingStrategy.STATE, image_np)
        DaSiamRPNTrackingStrategy.STATE = new_state
        return self._try_fill_missing_ball_detection_with_dasiam_ball(new_state)

    def is_condition_satisfied(self, detected_ball: Detection) -> bool:
        ball_not_found_by_base_tracker = not self._is_ball_found_by_base_tracker(detected_ball)
        ball_found_ever_before = self._is_ball_found_ever_before()
        return ball_not_found_by_base_tracker and ball_found_ever_before

    def _try_fill_missing_ball_detection_with_dasiam_ball(self, dasiam_ball_state: Dict[str, Any]) -> Detection:
        score = dasiam_ball_state["score"]
        if score > self.score_threshold:
            center_position = dasiam_ball_state["target_pos"]
            bbox_size = dasiam_ball_state["target_sz"]
            bbox_x_min = center_position[0] - bbox_size[0] / 2
            bbox_x_max = center_position[0] + bbox_size[0] / 2
            bbox_y_min = center_position[1] - bbox_size[1] / 2
            bbox_y_max = center_position[1] + bbox_size[1] / 2
            bbox = torch.tensor([[bbox_x_min, bbox_y_min], [bbox_x_max, bbox_y_max]])
            detected_ball = Detection(bbox, score, "ball", idx=0)
            return detected_ball

    @staticmethod
    def _is_ball_found_ever_before() -> bool:
        return len(list(DaSiamRPNTrackingStrategy.STATE.keys())) != 0


class DaSiamRPNDetector(Detector):
    def __init__(self, base_detector: Detector, path: str, score_threshold: float):
        self.base_detector = base_detector
        self.model: dasiam_net.SiamRPNBIG = dasiam_net.SiamRPNBIG()
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval()
        self.ball_tracking_strategies: List[DaSiamRPNTrackingStrategy] = [
            UpdatingDaSiamRPNStateStrategy(dasiam_rpn_model=self.model),
            GettingPredictionFromDaSiamRPNStrategy(dasiam_rpn_model=self.model, score_threshold=score_threshold)
        ]

    def detect(self, image: torch.Tensor) -> List[Detection]:
        base_detector_detections = self.base_detector.detect(image)
        ball_detection, people_detections = self._separate_detections(base_detector_detections)

        ball_detection = self._try_to_fill_missing_ball_detection(ball_detection, image)
        all_detections = people_detections
        if ball_detection:
            all_detections.append(ball_detection)
        return all_detections

    @staticmethod
    def _separate_detections(detections: List[Detection]) -> Tuple[Detection, List[Detection]]:
        ball_detection = None
        people_detections = []
        for detection in detections:
            if detection.detected_class == "ball":
                ball_detection = detection
            else:
                people_detections.append(detection)
        return ball_detection, people_detections

    def _try_to_fill_missing_ball_detection(self, detected_ball: Detection, image: torch.Tensor) -> Detection:
        dasiam_ball_detection = None
        for tracking_strategy in self.ball_tracking_strategies:
            strategy_condition_is_satisfied = tracking_strategy.is_condition_satisfied(detected_ball=detected_ball)
            if strategy_condition_is_satisfied:
                dasiam_ball_detection = tracking_strategy.dasiam_rpn_step(detected_ball, image)
        return dasiam_ball_detection
