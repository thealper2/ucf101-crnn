import logging
from typing import Tuple

import cv2
import torch
import torchvision.transforms as transforms

from models.crnn import CRNN

logger = logging.getLogger("DQN_GridWorld")


def test_video_with_model(
    model_path: str,
    video_path: str,
    sequence_length: int = 16,
    target_size: Tuple[int, int] = (112, 112),
    device: str = "auto",
) -> None:
    """Test a trained model with an external video file."""
    # Load the model
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx")
    if class_to_idx is None:
        raise ValueError("Model checkpoint does not contain class_to_idx mapping")

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Initialize model
    model = CRNN(num_classes=num_classes, sequence_length=sequence_length).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create transform
    transform = transforms.Compose(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Initialize frame buffer
    frame_buffer = []
    frame_count = 0

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create window
    cv2.namedWindow("Video Classification", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Add to frame buffer
        frame_buffer.append(frame_tensor)
        if len(frame_buffer) > sequence_length:
            frame_buffer.pop(0)

        # When we have enough frames, make a prediction
        if len(frame_buffer) == sequence_length:
            # Stack frames and apply transform
            video_tensor = torch.stack(frame_buffer)
            video_tensor = transform(video_tensor).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(video_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred_prob, pred_class = torch.max(probs, 1)

            # Get class name and confidence
            class_name = idx_to_class.get(pred_class.item(), "Unknown")
            confidence = pred_prob.item()

            # Display prediction on frame
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(
                frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        # Display frame
        cv2.imshow("Video Classification", frame)

        # Check for quit key
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
