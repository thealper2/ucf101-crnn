import logging
from typing import Dict, List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("DQN_GridWorld")


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    save_path: str,
) -> Dict[str, List[float]]:
    """Train the CRNN model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    logger.info("Starting training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False
        )

        for batch_idx, (videos, labels) in enumerate(train_pbar):
            try:
                videos, labels = videos.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Update progress bar
                train_pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * train_correct / train_total:.2f}%",
                    }
                )

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False
        )

        with torch.no_grad():
            for batch_idx, (videos, labels) in enumerate(val_pbar):
                try:
                    videos, labels = videos.to(device), labels.to(device)
                    outputs = model(videos)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    # Update progress bar
                    val_pbar.set_postfix(
                        {
                            "Loss": f"{loss.item():.4f}",
                            "Acc": f"{100.0 * val_correct / val_total:.2f}%",
                        }
                    )

                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue

        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print epoch results
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_to_idx": train_loader.dataset.class_to_idx
                    if hasattr(train_loader.dataset, "class_to_idx")
                    else None,
                },
                save_path,
            )
            logger.info(
                f"New best model saved with validation accuracy: {val_acc:.2f}%"
            )

        scheduler.step()

    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return history
