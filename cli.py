import logging
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import VideoDataset
from data.preprocessing import create_data_transforms, load_ucf101_data
from models.crnn import CRNN
from utils.evaluation import evaluate_model
from utils.training import train_model
from utils.video_utils import test_video_with_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger("training")


def main(
    data_dir: str = typer.Argument(..., help="Path to UCF101 dataset directory"),
    output_dir: str = typer.Option("./output", help="Directory to save outputs"),
    batch_size: int = typer.Option(8, help="Batch size for training"),
    num_epochs: int = typer.Option(50, help="Number of training epochs"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    sequence_length: int = typer.Option(16, help="Number of frames per video"),
    test_size: float = typer.Option(0.2, help="Proportion of data for testing"),
    val_size: float = typer.Option(
        0.1, help="Proportion of training data for validation"
    ),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility"),
    device: str = typer.Option("auto", help="Device to use (auto, cpu, cuda)"),
    samples_per_class: Optional[int] = typer.Option(
        None, help="Number of samples per class (None for all)"
    ),
    test_video: Optional[str] = typer.Option(
        None, help="Path to external video file for testing"
    ),
    model_path: Optional[str] = typer.Option(
        None, help="Path to trained model for testing"
    ),
) -> None:
    """
    Train and evaluate CRNN model on UCF101 dataset.
    """
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # If test_video and model_path are provided, run video testing
    if test_video is not None and model_path is not None:
        logger.info(f"Testing model on external video: {test_video}")
        test_video_with_model(
            model_path=model_path,
            video_path=test_video,
            sequence_length=sequence_length,
            device=device,
        )
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset
        logger.info("Loading UCF101 dataset...")
        video_paths, labels, class_to_idx = load_ucf101_data(
            data_dir, samples_per_class
        )
        class_names = list(class_to_idx.keys())
        num_classes = len(class_names)

        # Split data into train/test
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            video_paths,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_seed,
        )

        # Split training data into train/validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths,
            train_labels,
            test_size=val_size,
            stratify=train_labels,
            random_state=random_seed,
        )

        logger.info(
            f"Dataset split - Train: {len(train_paths)}, "
            f"Val: {len(val_paths)}, Test: {len(test_paths)}"
        )

        # Create data transforms
        train_transform, val_transform = create_data_transforms()

        # Create datasets
        train_dataset = VideoDataset(
            train_paths, train_labels, sequence_length, train_transform
        )
        train_dataset.class_to_idx = class_to_idx  # Save class mapping

        val_dataset = VideoDataset(
            val_paths, val_labels, sequence_length, val_transform
        )

        test_dataset = VideoDataset(
            test_paths, test_labels, sequence_length, val_transform
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # Initialize model
        model = CRNN(num_classes=num_classes, sequence_length=sequence_length).to(
            device
        )

        logger.info(f"Model initialized with {num_classes} classes")

        # Train model
        model_save_path = output_path / "best_model.pth"
        history = train_model(
            model,
            train_loader,
            val_loader,
            device,
            num_epochs,
            learning_rate,
            str(model_save_path),
        )

        # Load best model for evaluation
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate on test set
        test_results = evaluate_model(model, test_loader, device, class_names)

        # Save results
        results_path = output_path / "results.txt"
        with open(results_path, "w") as f:
            f.write(f"Test Accuracy: {test_results['accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(
                classification_report(
                    test_results["labels"],
                    test_results["predictions"],
                    target_names=class_names,
                )
            )

        logger.info(f"Training completed successfully. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
