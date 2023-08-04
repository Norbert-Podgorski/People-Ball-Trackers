import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms


class FramesLoader:
    def __init__(self, frames_path: str, resize: List[int]):
        self.frames_path = frames_path
        self.to_tensor_converter = transforms.ToTensor()
        self.resizer = transforms.Resize(resize)

    def load_all_frames(self) -> torch.Tensor:
        frames_names = sorted(Path(self.frames_path).glob('*.jpg'), key=os.path.getmtime)
        frames = torch.stack([self.resizer(self.to_tensor_converter(Image.open(frame_name))) for frame_name in frames_names])
        return frames

    def load_subset_frames(self, subset: int) -> torch.Tensor:
        frames_names = sorted(Path(self.frames_path).glob('*.jpg'), key=os.path.getmtime)
        frames = []
        for frame_name in frames_names:
            frame = self.resizer(self.to_tensor_converter(Image.open(frame_name)))
            frames.append(frame)

            if len(frames) == subset:
                return torch.stack(frames)
