import yaml
from yaml import SafeLoader

from src.detectors.detector import Detector
from src.detectors.detector_factory import DetectorFactory
from src.frames_loader import FramesLoader
from src.visualizer import Visualizer


def detect(detector: Detector, frames_loader: FramesLoader, visualizer: Visualizer) -> None:
    frames = frames_loader.load_all_frames()

    batch_size = config["batch_size"]
    scenes_number = frames.shape[0]
    detections = []
    batch_low_idx = 0
    batch_high_idx = 0
    while batch_high_idx <= scenes_number:
        batch_high_idx += batch_size
        if batch_high_idx >= scenes_number:
            batch_high_idx = scenes_number + 1
        batch = frames[batch_low_idx: batch_high_idx]
        batch_detections = detector.detect(batch)
        detections.append(batch_detections)
        batch_low_idx += batch_size
    detections = [detection for detection_for_batch in detections for detection in detection_for_batch]

    visualizer.visualize(frames, detections)


if __name__ == "__main__":
    config_path = "../configs/detection_config.yaml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=SafeLoader)
    detector_config_path = config["detector_config_path"]
    with open(detector_config_path) as detector_config_file:
        detector_config = yaml.load(detector_config_file, Loader=SafeLoader)

    _detector = DetectorFactory.create(**detector_config)
    _frames_loader = FramesLoader(**config["frames_loader_config"])
    _visualizer = Visualizer(**config["visualizer_config"])

    detect(detector=_detector, frames_loader=_frames_loader, visualizer=_visualizer)
