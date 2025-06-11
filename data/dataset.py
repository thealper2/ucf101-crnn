import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_video

logger = logging.getLogger("DQN_GridWorld")


class VideoDataset(Dataset):
    """Custom dataset class for loading and preprocessing video data."""

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        sequence_length: int = 16,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (112, 112),
    ) -> None:
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_size = target_size

        if len(video_paths) != len(labels):
            raise ValueError("Number of video paths must match number of labels")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            # Load video using torchvision
            video_path = self.video_paths[idx]
            video, _, _ = read_video(video_path, pts_unit="sec")

            # Convert from (T, H, W, C) to (T, C, H, W)
            video = video.permute(0, 3, 1, 2).float() / 255.0

            # Sample frames uniformly if video is longer than sequence_length
            if video.shape[0] > self.sequence_length:
                indices = torch.linspace(
                    0, video.shape[0] - 1, self.sequence_length
                ).long()
                video = video[indices]
            elif video.shape[0] < self.sequence_length:
                # Repeat frames if video is shorter than sequence_length
                repeat_factor = self.sequence_length // video.shape[0] + 1
                video = video.repeat(repeat_factor, 1, 1, 1)[: self.sequence_length]

            # Resize frames to target size
            video = torch.nn.functional.interpolate(
                video, size=self.target_size, mode="bilinear", align_corners=False
            )

            # Apply transformations if provided
            if self.transform:
                transformed_frames = []
                for frame in video:
                    transformed_frame = self.transform(frame)
                    transformed_frames.append(transformed_frame)
                video = torch.stack(transformed_frames)

            label = self.labels[idx]
            return video, label

        except Exception as e:
            logger.error(f"Error loading video {self.video_paths[idx]}: {str(e)}")
            # Return a dummy tensor and label in case of error
            dummy_video = torch.zeros(self.sequence_length, 3, *self.target_size)
            return dummy_video, self.labels[idx]
