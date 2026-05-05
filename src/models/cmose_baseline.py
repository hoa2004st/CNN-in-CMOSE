"""CMOSE baseline model (OpenFace + I3D) aligned with the implementation spec."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import MLPBlock, TemporalConvBackbone


def score_to_class(score: float) -> int:
    if score < -0.5:
        return 0
    if score < 0.0:
        return 1
    if score < 0.5:
        return 2
    return 3


class CMOSEBaselineModel(nn.Module):
    """Spec-shaped baseline with temporal attention and normalized score head."""

    def __init__(self, *, i3d_dim: int = 1024, openface_dim: int = 147, c: int = 128, t: int = 10) -> None:
        super().__init__()
        self.t = int(t)
        self.attn_mlp = nn.Sequential(
            MLPBlock(i3d_dim, c, dropout=0.5),
            nn.Linear(c, self.t),
        )
        self.tcn = TemporalConvBackbone(
            in_channels=openface_dim,
            hidden_channels=c,
            layers=4,
            kernel_size=3,
            dropout=0.2,
        )
        self.i3d_projection = nn.Sequential(
            nn.Linear(i3d_dim, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, c),
        )
        self.score_weight = nn.Parameter(torch.empty(1, c * 2))
        nn.init.xavier_uniform_(self.score_weight)

    def forward(
        self,
        openface_features: torch.Tensor,
        i3d_features: torch.Tensor,
        *,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if openface_features.ndim != 3:
            raise ValueError(
                f"Expected openface_features shape (batch, channels, time), got {openface_features.shape}"
            )
        if i3d_features.ndim != 2:
            raise ValueError(f"Expected i3d_features shape (batch, features), got {i3d_features.shape}")

        attn_logits = self.attn_mlp(i3d_features)
        attn = torch.softmax(attn_logits, dim=-1).unsqueeze(1)  # B x 1 x T
        tcn_out = self.tcn(openface_features)  # B x C x T
        hl = torch.matmul(tcn_out, attn.transpose(1, 2)).squeeze(-1)  # B x C
        i3d_proj = self.i3d_projection(i3d_features)  # B x C
        embedding = torch.cat([i3d_proj, hl], dim=-1)  # B x 2C
        score = F.linear(F.normalize(embedding), F.normalize(self.score_weight)).squeeze(-1)

        if return_embedding:
            return score, embedding
        return score
