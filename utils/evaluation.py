import logging
from typing import Any, Dict, List

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("DQN_GridWorld")


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    """Evaluate the trained model on test data."""
    model.eval()
    all_predictions = []
    all_labels = []

    logger.info("Evaluating model...")

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for videos, labels in test_pbar:
            try:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                logger.error(f"Error in evaluation: {str(e)}")
                continue

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(
        all_labels, all_predictions, target_names=class_names, output_dict=True
    )

    logger.info(f"Test Accuracy: {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "predictions": all_predictions,
        "labels": all_labels,
    }
