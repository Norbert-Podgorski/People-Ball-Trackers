import os

import cv2


def video_splitter(video_path: str, destination_path: str) -> None:
    create_path_if_not_exists(path=destination_path)

    video_capture = cv2.VideoCapture(video_path)
    frame_number = 0
    success = True
    while success:
        success, frame = video_capture.read()
        if success:
            frame_path = f'{destination_path}/frame_{frame_number}.jpg'
            cv2.imwrite(frame_path, frame)
        frame_number += 1
    video_capture.release()


def create_path_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    destination = "example_data/test/frames/passes_run"
    video = "example_data/test/recordings/passes_run.mp4"
    video_splitter(video_path=video, destination_path=destination)
