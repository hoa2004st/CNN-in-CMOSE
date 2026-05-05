"""Model definitions for the narrowed CMOSE comparison pipeline."""

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


class RawTemporalCNN(nn.Module):
    """1-D temporal CNN over frame-level OpenFace features."""

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
    """Small transformer encoder for frame-level OpenFace features."""

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


class FlattenMLP(nn.Module):
    """Simple MLP baseline that flattens the input tensor for classification."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OpenFaceTCNI3DFusionModel(nn.Module):
    """Shared-fusion model with an OpenFace TCN branch and an I3D branch."""

    def __init__(
        self,
        *,
        openface_features: int = 709,
        i3d_features: int = 1024,
        hidden_dim: int = 128,
        num_classes: int = 4,
        reconstruction_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = float(reconstruction_weight)
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
        self.shared_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.openface_reconstruction = nn.Linear(hidden_dim, hidden_dim)
        self.i3d_reconstruction = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def _encode_streams(
        self,
        openface_x: torch.Tensor,
        i3d_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        openface_seq = self.openface_encoder(openface_x.transpose(1, 2)).transpose(1, 2)
        i3d_seq = self.i3d_projection(i3d_x)
        i3d_seq = self.i3d_temporal(i3d_seq.transpose(1, 2)).transpose(1, 2)
        return openface_seq, i3d_seq

    def _fuse_sequences(self, openface_seq: torch.Tensor, i3d_seq: torch.Tensor) -> torch.Tensor:
        fused_input = torch.cat([openface_seq, i3d_seq], dim=-1)
        return self.shared_fusion(fused_input)

    def forward_with_aux(
        self,
        openface_x: torch.Tensor,
        i3d_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        openface_seq, i3d_seq = self._encode_streams(openface_x, i3d_x)
        fused = self._fuse_sequences(openface_seq, i3d_seq)
        openface_recon = self.openface_reconstruction(fused)
        i3d_recon = self.i3d_reconstruction(fused)
        aux_loss = self.reconstruction_weight * (
            torch.mean((openface_recon - openface_seq) ** 2)
            + torch.mean((i3d_recon - i3d_seq) ** 2)
        )
        pooled = fused.mean(dim=1)
        return self.classifier(pooled), aux_loss

    def forward(self, openface_x: torch.Tensor, i3d_x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_aux(openface_x, i3d_x)
        return logits


def build_model(
    model_name: str,
    *,
    input_features: int = 709,
    i3d_input_features: int | None = None,
    num_classes: int = 4,
) -> tuple[nn.Module, ModelSpec]:
    """Create a model and describe its expected input format."""
    if model_name == "openface_mlp":
        return (
            FlattenMLP(num_classes=num_classes),
            ModelSpec(name=model_name, input_kind="openface_flat_mlp"),
        )
    if model_name == "temporal_cnn":
        return (
            RawTemporalCNN(input_features=input_features, num_classes=num_classes),
            ModelSpec(name=model_name, input_kind="sequence"),
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
    if model_name == "i3d_mlp":
        return (
            FlattenMLP(num_classes=num_classes),
            ModelSpec(name=model_name, input_kind="i3d_flat_mlp"),
        )
    if model_name == "openface_tcn_i3d_fusion":
        if i3d_input_features is None:
            raise ValueError("i3d_input_features is required for openface_tcn_i3d_fusion")
        return (
            OpenFaceTCNI3DFusionModel(
                openface_features=input_features,
                i3d_features=i3d_input_features,
                num_classes=num_classes,
            ),
            ModelSpec(name=model_name, input_kind="multimodal_sequence"),
        )
    raise ValueError(f"Unknown model: {model_name}")
