import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """
    CNN component of CRNN for spatial feature extraction from video frames.
    """

    def __init__(
        self, input_channels: int = 3, hidden_dim: int = 512, dropout_rate: float = 0.3
    ) -> None:
        """
        Initialize CNN feature extractor.

        Args:
            input_channels (int): Number of input channels (3 for RGB)
            hidden_dim (int): Dimension of output features
            dropout_rate (float): Dropout rate for regularization
        """
        super(CNNFeatureExtractor, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128, stride=2)
        self.res_block2 = self._make_residual_block(128, 256, stride=2)
        self.res_block3 = self._make_residual_block(256, 512, stride=2)

        # Global average pooling and final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, hidden_dim)

    def _make_residual_block(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> nn.Sequential:
        """
        Create a residual block with skip connection.

        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            stride (int): Convolution stride

        Returns:
            Sequential residual block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CNN feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Feature tensor of shape (batch_size, hidden_dim)
        """
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)

        return x


class RNNTemporalModel(nn.Module):
    """
    RNN component of CRNN for temporal sequence modeling.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
        """
        Initialize RNN temporal model.

        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden dimension of LSTM
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
        """
        super(RNNTemporalModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RNN temporal model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Attention-weighted feature tensor of shape (batch_size, hidden_dim * 2)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)

        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum of LSTM outputs
        attended_output = torch.sum(
            lstm_out * attention_weights, dim=1
        )  # (batch_size, hidden_dim * 2)

        return self.dropout(attended_output)


class CRNN(nn.Module):
    """
    Complete CRNN model combining CNN feature extraction and RNN temporal modeling
    for video action recognition.
    """

    def __init__(
        self,
        num_classes: int,
        sequence_length: int = 16,
        input_channels: int = 3,
        cnn_hidden_dim: int = 512,
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
        """
        Initialize CRNN model.

        Args:
            num_classes (int): Number of action classes
            input_channels (int): Number of input channels (3 for RGB)
            cnn_hidden_dim (int): CNN output dimension
            rnn_hidden_dim (int): RNN hidden dimension
            rnn_num_layers (int): Number of RNN layers
            dropout_rate (int): Dropout rate for regularization
        """
        super(CRNN, self).__init__()

        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # CNN for spatial feature extraction
        self.cnn = CNNFeatureExtractor(
            input_channels=input_channels,
            hidden_dim=cnn_hidden_dim,
            dropout_rate=dropout_rate,
        )

        # RNN for temporal modeling
        self.rnn = RNNTemporalModel(
            input_dim=cnn_hidden_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            dropout_rate=dropout_rate,
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_dim * 2, rnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CRNN model.

        Args:
            x (torch.Tensor): Input video tensor of shape (batch_size, sequence_length, channels, height, width)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.size()

        # Reshape for CNN processing: (batch_size * seq_len, channels, height, width)
        x = x.view(batch_size * seq_len, channels, height, width)

        # Extract spatial features using CNN
        cnn_features = self.cnn(x)  # (batch_size * seq_len, cnn_hidden_dim)

        # Reshape back for RNN: (batch_size, seq_len, cnn_hidden_dim)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)

        # Model temporal dependencies using RNN
        rnn_features = self.rnn(cnn_features)  # (batch_size, rnn_hidden_dim * 2)

        # Final classification
        logits = self.classifier(rnn_features)  # (batch_size, num_classes)

        return logits

    def get_feature_extractor(self) -> nn.Module:
        """
        Get the CNN feature extractor for transfer learning or feature extraction.

        Returns:
            CNN feature extractor module
        """
        return self.cnn

    def freeze_cnn(self) -> None:
        """
        Freeze CNN parameters for fine-tuning only the RNN and classifier.
        """
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self) -> None:
        """
        Unfreeze CNN parameters for end-to-end training.
        """
        for param in self.cnn.parameters():
            param.requires_grad = True
