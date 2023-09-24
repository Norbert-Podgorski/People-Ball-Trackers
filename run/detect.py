import yaml
from yaml import SafeLoader

from src.detectors.detector import Detector
from src.detectors.detector_factory import DetectorFactory
from src.frames_loader import FramesLoader
from src.logger import Logger
from src.visualizer import Visualizer


def detect(detector: Detector, frames_loader: FramesLoader, visualizer: Visualizer, logger: Logger) -> None:
    batch_size = config["batch_size"]
    frames = frames_loader.load_all_frames(batch_size=batch_size)
    # frames = frames_loader.load_subset_frames(100)
    detections = []
    for frame in frames:
        detections_for_frame = detector.detect(frame)
        detections.append(detections_for_frame)

    logger.log(detections=detections, frames_number=len(frames))

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
    _logger = Logger(**config["logger_config"])

    detect(detector=_detector, frames_loader=_frames_loader, visualizer=_visualizer, logger=_logger)
