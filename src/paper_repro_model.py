"""Model definitions for the strict CMOSE comparison pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    """Describe how a model consumes preprocessed tensors."""

    name: str
    input_kind: str


class PaperEngagementCNN(nn.Module):
    """4-block square-matrix CNN matching the paper summary."""

    def __init__(self, *, input_size: int = 300, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flat_size = self._infer_flat_size(input_size=input_size)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(flat_size, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)

    def _infer_flat_size(self, *, input_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size, input_size)
            output = self.features(dummy)
            return int(output.numel())


class RawTemporalCNN(nn.Module):
    """1-D temporal CNN over the original frame sequence."""

    def __init__(self, *, input_features: int = 709, num_classes: int = 4) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_features, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.temporal(x)
        return self.classifier(x)


class RectangularFilterCNN(nn.Module):
    """2-D CNN with asymmetric kernels for frame-feature inputs."""

    def __init__(self, *, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 15), padding=(2, 7)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(5, 9), padding=(2, 4)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SpectralConvNet(nn.Module):
    """CNN over TES time-frequency tensors."""

    def __init__(self, *, n_input_features: int, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_input_features, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SequenceLSTM(nn.Module):
    """LSTM classifier over frame-level OpenFace features."""

    def __init__(
        self,
        *,
        input_features: int = 709,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3 if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        return self.classifier(hidden[-1])


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SequenceTransformer(nn.Module):
    """Small transformer encoder for raw OpenFace sequences."""

    def __init__(
        self,
        *,
        input_features: int = 709,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_features, d_model)
        self.position = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.position(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)


class DualStreamOpenFaceI3DModel(nn.Module):
    """Late-fusion temporal model over OpenFace and precomputed I3D features."""

    def __init__(
        self,
        *,
        openface_features: int = 709,
        i3d_features: int = 1024,
        hidden_dim: int = 128,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.openface_encoder = nn.Sequential(
            nn.Conv1d(openface_features, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.i3d_projection = nn.Sequential(
            nn.Linear(i3d_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.i3d_temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, openface_x: torch.Tensor, i3d_x: torch.Tensor) -> torch.Tensor:
        openface_seq = self.openface_encoder(openface_x.transpose(1, 2)).transpose(1, 2)

        i3d_seq = self.i3d_projection(i3d_x)
        i3d_seq = self.i3d_temporal(i3d_seq.transpose(1, 2)).transpose(1, 2)

        fused_input = torch.cat([openface_seq, i3d_seq], dim=-1)
        gate = torch.sigmoid(self.gate(fused_input))
        fused = gate * openface_seq + (1.0 - gate) * i3d_seq
        pooled = fused.mean(dim=1)
        return self.classifier(pooled)


def build_model(
    model_name: str,
    *,
    input_size: int = 300,
    input_features: int = 709,
    i3d_input_features: int | None = None,
    num_classes: int = 4,
) -> tuple[nn.Module, ModelSpec]:
    """Create a model and describe its expected input format."""
    if model_name == "paper_cnn":
        return (
            PaperEngagementCNN(input_size=input_size, num_classes=num_classes),
            ModelSpec(name=model_name, input_kind="square_matrix"),
        )
    if model_name == "temporal_cnn":
        return (
            RawTemporalCNN(input_features=input_features, num_classes=num_classes),
            ModelSpec(name=model_name, input_kind="sequence"),
        )
    if model_name == "rectangular_cnn":
        return (
            RectangularFilterCNN(num_classes=num_classes),
            ModelSpec(name=model_name, input_kind="frame_feature_map"),
        )
    if model_name == "spectral_cnn":
        return (
            SpectralConvNet(n_input_features=input_features, num_classes=num_classes),
            ModelSpec(name=model_name, input_kind="spectral_tensor"),
        )
    if model_name == "lstm":
        return (
            SequenceLSTM(input_features=input_features, num_classes=num_classes),
            ModelSpec(name=model_name, input_kind="sequence"),
        )
    if model_name == "transformer":
        return (
            SequenceTransformer(input_features=input_features, num_classes=num_classes),
            ModelSpec(name=model_name, input_kind="sequence"),
        )
    if model_name == "openface_i3d_temporal_fusion":
        if i3d_input_features is None:
            raise ValueError("i3d_input_features is required for openface_i3d_temporal_fusion")
        return (
            DualStreamOpenFaceI3DModel(
                openface_features=input_features,
                i3d_features=i3d_input_features,
                num_classes=num_classes,
            ),
            ModelSpec(name=model_name, input_kind="multimodal_sequence"),
        )
    raise ValueError(f"Unknown model: {model_name}")
