import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torchvision.transforms as transforms

logger = logging.getLogger("DQN_GridWorld")


def load_ucf101_data(
    data_dir: str, samples_per_class: Optional[int] = None
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Load UCF101 dataset from directory structure."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    video_paths = []
    labels = []
    class_names = []

    # Get all class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    class_dirs.sort()  # Ensure consistent ordering

    # Create class to index mapping
    class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}

    logger.info(f"Found {len(class_dirs)} classes in dataset")

    # Load videos from each class directory
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]

        # Find all video files in the class directory
        video_files = []
        for ext in ["*.avi", "*.mp4", "*.mov", "*.mkv"]:
            video_files.extend(class_dir.glob(ext))

        # If samples_per_class is specified, take that many samples
        if samples_per_class is not None:
            if len(video_files) > samples_per_class:
                video_files = random.sample(video_files, samples_per_class)

        logger.info(f"Class '{class_name}': {len(video_files)} videos")

        for video_file in video_files:
            video_paths.append(str(video_file))
            labels.append(class_idx)
            class_names.append(class_name)

    logger.info(f"Total videos loaded: {len(video_paths)}")
    return video_paths, labels, class_to_idx


def create_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Create data transformations for training and validation."""
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    return train_transform, val_transform
