"""Model definitions for the narrowed CMOSE comparison pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    """Describe how a model consumes preprocessed tensors."""

    name: str
    input_kind: str


class Chomp1d(nn.Module):
    """Remove right-side padding so temporal convolutions remain causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Canonical TCN residual block with causal dilated convolutions."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.activation = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(self.network(x) + residual)


class TCNEncoder(nn.Module):
    """Stack residual causal dilated TCN blocks over a temporal sequence."""

    def __init__(
        self,
        *,
        input_channels: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for idx, out_channels in enumerate(channels):
            in_channels = input_channels if idx == 0 else channels[idx - 1]
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RawTemporalCNN(nn.Module):
    """Canonical TCN classifier over frame-level OpenFace features."""

    def __init__(self, *, input_features: int = 709, num_classes: int = 4) -> None:
        super().__init__()
        self.temporal = TCNEncoder(
            input_channels=input_features,
            channels=[256, 128, 128],
            kernel_size=3,
            dropout=0.2,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
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
        x = self.pool(x)
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
    """Simple MLP that flattens the input tensor for classification."""

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


class I3DVectorMLP(nn.Module):
    """MLP classifier over the pooled clip-level I3D vector.

    CMOSE/DAiSEE ship a single 1024-d I3D embedding per clip (the loader tiles it over time to a
    fixed length). Mean-pooling over time recovers that vector, which a small MLP
    (1024 -> 256 -> 128 -> num_classes) then classifies. This matches the head shape used by the
    other baselines (the FlattenMLP 256 -> 128 -> num_classes), with a 128-wide hidden layer.
    """

    def __init__(
        self,
        *,
        input_features: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)   # (B, T, 1024) → (B, 1024)
        return self.network(x)


class OpenFaceTCNI3DFusionModel(nn.Module):
    """Shared-fusion model with canonical TCN branches for temporal encoding."""

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
        self.openface_encoder = TCNEncoder(
            input_channels=openface_features,
            channels=[256, 128, hidden_dim],
            kernel_size=3,
            dropout=0.2,
        )
        self.i3d_projection = nn.Sequential(
            nn.Linear(i3d_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.i3d_temporal = TCNEncoder(
            input_channels=hidden_dim,
            channels=[hidden_dim, hidden_dim],
            kernel_size=3,
            dropout=0.2,
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


GROUP_ENCODER_ARCHS = ("transformer", "tcn", "lstm")


class GroupEncoder(nn.Module):
    """Temporal encoder for one semantic feature group.

    Wraps a small Transformer, a TCN, or an LSTM and always produces a fixed
    (batch, embed_dim) representation via mean-pool over time + LayerNorm.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        embed_dim: int,
        arch: str,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if arch not in GROUP_ENCODER_ARCHS:
            raise ValueError(
                f"GroupEncoder arch must be one of {GROUP_ENCODER_ARCHS}, got '{arch}'"
            )
        self.arch = arch
        self.embed_dim = embed_dim
        if arch == "transformer":
            self.proj = nn.Linear(in_dim, embed_dim)
            self.pos = PositionalEncoding(d_model=embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder: nn.Module = nn.TransformerEncoder(encoder_layer, num_layers=2)
        elif arch == "lstm":
            # Two-layer LSTM whose per-timestep hidden state has width embed_dim,
            # so mean-pooling over time yields a fixed embed_dim vector — parallel
            # to the TCN/Transformer branches.
            self.encoder = nn.LSTM(
                input_size=in_dim,
                hidden_size=embed_dim,
                num_layers=2,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.encoder = TCNEncoder(
                input_channels=in_dim,
                channels=[embed_dim, embed_dim],
                kernel_size=3,
                dropout=dropout,
            )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        if self.arch == "transformer":
            x = self.proj(x)
            x = self.pos(x)
            x = self.encoder(x)
            x = x.mean(dim=1)
        elif self.arch == "lstm":
            x, _ = self.encoder(x)               # → (B, T, embed_dim)
            x = x.mean(dim=1)                    # → (B, embed_dim)
        else:
            x = self.encoder(x.transpose(1, 2))  # → (B, embed_dim, T)
            x = x.mean(dim=2)                    # → (B, embed_dim)
        return self.norm(x)


class OpenFaceTemporalHybrid(nn.Module):
    """Hybrid model: each OpenFace semantic group gets its own Transformer or TCN encoder.

    Each GroupEncoder produces a fixed embed_dim vector.  All group vectors are
    concatenated and passed through a shared MLP classifier.  Each group also has
    an auxiliary classification head trained jointly via a weighted auxiliary loss.

    Args:
        group_indices: maps group name → int64 index array into the 709-dim feature
                       axis (built by build_group_column_indices).
        group_architectures: maps group name → "transformer" | "tcn" | "lstm".
        embed_dim: output dimension of each GroupEncoder (default 64).
        aux_weight: weight applied to the mean of auxiliary group losses (default 0.2).
        num_classes: number of engagement classes (default 4).
    """

    def __init__(
        self,
        *,
        group_indices: dict[str, "np.ndarray"],
        group_architectures: dict[str, str],
        embed_dim: int = 64,
        aux_weight: float = 0.2,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.group_names: list[str] = list(group_indices.keys())

        self.encoders = nn.ModuleDict()
        self.aux_heads = nn.ModuleDict()
        for name in self.group_names:
            idxs = group_indices[name]
            arch = group_architectures.get(name, "tcn")
            self.encoders[name] = GroupEncoder(
                in_dim=len(idxs),
                embed_dim=embed_dim,
                arch=arch,
            )
            self.aux_heads[name] = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(embed_dim, num_classes),
            )
            # Register index buffer for device-agnostic slicing
            self.register_buffer(
                f"_idx_{name}",
                torch.from_numpy(np.asarray(idxs, dtype=np.int64)),
                persistent=True,
            )

        fusion_dim = embed_dim * len(self.group_names)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def _encode_groups(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings: dict[str, torch.Tensor] = {}
        for name in self.group_names:
            idx = getattr(self, f"_idx_{name}")
            group_x = x[:, :, idx]          # (B, T, group_dim)
            embeddings[name] = self.encoders[name](group_x)
        return embeddings

    def forward_with_aux(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = self._encode_groups(x)
        fused = torch.cat([embeddings[n] for n in self.group_names], dim=-1)
        main_logits = self.fusion(fused)
        aux_logits = {name: self.aux_heads[name](emb) for name, emb in embeddings.items()}
        return main_logits, aux_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self._encode_groups(x)
        fused = torch.cat([embeddings[n] for n in self.group_names], dim=-1)
        return self.fusion(fused)


# Backward-compatibility alias
SemanticGroupFusionModel = OpenFaceTemporalHybrid


class OpenFaceTemporalI3DHybrid(nn.Module):
    """Hybrid model: OpenFace semantic group encoders + I3D temporal stream.

    Extends OpenFaceTemporalHybrid by adding an I3D encoder (always TCN) as an
    additional stream.  All group embeddings and the I3D embedding are concatenated
    and passed through a shared MLP classifier.  Each group and the I3D stream each
    have an auxiliary classification head.

    Args:
        group_indices: maps group name → int64 index array (from build_group_column_indices).
        group_architectures: maps group name → "transformer" | "tcn" | "lstm".
        i3d_input_dim: I3D feature dimension (typically 1024).
        embed_dim: embedding size for every encoder (default 64).
        aux_weight: auxiliary loss weight (default 0.2).
        num_classes: number of engagement classes (default 4).
    """

    def __init__(
        self,
        *,
        group_indices: dict[str, "np.ndarray"],
        group_architectures: dict[str, str],
        i3d_input_dim: int = 1024,
        embed_dim: int = 64,
        aux_weight: float = 0.2,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.group_names: list[str] = list(group_indices.keys())

        # OpenFace group encoders + aux heads
        self.encoders = nn.ModuleDict()
        self.aux_heads = nn.ModuleDict()
        for name in self.group_names:
            idxs = group_indices[name]
            arch = group_architectures.get(name, "tcn")
            self.encoders[name] = GroupEncoder(in_dim=len(idxs), embed_dim=embed_dim, arch=arch)
            self.aux_heads[name] = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(embed_dim, num_classes))
            self.register_buffer(
                f"_idx_{name}",
                torch.from_numpy(np.asarray(idxs, dtype=np.int64)),
                persistent=True,
            )

        # I3D encoder: an MLP on the pooled 1024-d clip vector (1024 -> 256 -> 128) + aux head.
        # CMOSE/DAiSEE provide one clip-level I3D vector per clip (the loader tiles it over time);
        # _encode_i3d mean-pools that tiled sequence back to the single vector before projecting.
        # The I3D stream gets a wider 128-d embedding (vs the 64-d group embeddings) because the
        # 1024-d I3D descriptor is richer; this 128-d embedding is concatenated with the group ones.
        i3d_embed_dim = 128
        self.i3d_projection = nn.Sequential(
            nn.Linear(i3d_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, i3d_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.i3d_norm = nn.LayerNorm(i3d_embed_dim)
        self.aux_heads["i3d"] = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(i3d_embed_dim, num_classes))

        fusion_dim = embed_dim * len(self.group_names) + i3d_embed_dim  # 5*64 + 128 = 448
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def _encode_openface_groups(self, x_of: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings: dict[str, torch.Tensor] = {}
        for name in self.group_names:
            idx = getattr(self, f"_idx_{name}")
            embeddings[name] = self.encoders[name](x_of[:, :, idx])
        return embeddings

    def _encode_i3d(self, x_i3d: torch.Tensor) -> torch.Tensor:
        # x_i3d: (B, T, 1024) — the clip-level I3D vector tiled over time. Mean-pooling over
        # time recovers the single 1024-d vector, which the MLP then projects to the embedding.
        h = x_i3d.mean(dim=1)                          # → (B, 1024)
        h = self.i3d_projection(h)                     # → (B, embed_dim)
        return self.i3d_norm(h)

    def forward_with_aux(
        self, x_of: torch.Tensor, x_i3d: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = self._encode_openface_groups(x_of)
        embeddings["i3d"] = self._encode_i3d(x_i3d)
        all_names = self.group_names + ["i3d"]
        fused = torch.cat([embeddings[n] for n in all_names], dim=-1)
        main_logits = self.fusion(fused)
        aux_logits = {name: self.aux_heads[name](emb) for name, emb in embeddings.items()}
        return main_logits, aux_logits

    def forward(self, x_of: torch.Tensor, x_i3d: torch.Tensor) -> torch.Tensor:
        embeddings = self._encode_openface_groups(x_of)
        embeddings["i3d"] = self._encode_i3d(x_i3d)
        all_names = self.group_names + ["i3d"]
        fused = torch.cat([embeddings[n] for n in all_names], dim=-1)
        return self.fusion(fused)


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
            I3DVectorMLP(
                input_features=i3d_input_features if i3d_input_features else 1024,
                num_classes=num_classes,
            ),
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


def build_openface_temporal_hybrid(
    *,
    group_indices: "dict[str, np.ndarray]",
    group_architectures: "dict[str, str]",
    embed_dim: int = 64,
    aux_weight: float = 0.2,
    num_classes: int = 4,
) -> tuple["OpenFaceTemporalHybrid", ModelSpec]:
    """Construct an OpenFaceTemporalHybrid (OpenFace-only semantic group fusion)."""
    model = OpenFaceTemporalHybrid(
        group_indices=group_indices,
        group_architectures=group_architectures,
        embed_dim=embed_dim,
        aux_weight=aux_weight,
        num_classes=num_classes,
    )
    return model, ModelSpec(name="openface_temporal_hybrid", input_kind="sequence")


# Backward-compatibility alias
build_semantic_group_model = build_openface_temporal_hybrid


def build_openface_temporal_i3d_hybrid(
    *,
    group_indices: "dict[str, np.ndarray]",
    group_architectures: "dict[str, str]",
    i3d_input_dim: int = 1024,
    embed_dim: int = 64,
    aux_weight: float = 0.2,
    num_classes: int = 4,
) -> tuple["OpenFaceTemporalI3DHybrid", ModelSpec]:
    """Construct an OpenFaceTemporalI3DHybrid (OpenFace semantic groups + I3D)."""
    model = OpenFaceTemporalI3DHybrid(
        group_indices=group_indices,
        group_architectures=group_architectures,
        i3d_input_dim=i3d_input_dim,
        embed_dim=embed_dim,
        aux_weight=aux_weight,
        num_classes=num_classes,
    )
    return model, ModelSpec(name="openface_temporal_i3d_hybrid", input_kind="multimodal_sequence")
