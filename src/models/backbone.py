"""Shared model building blocks used by CMOSE model variants."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """Linear -> ReLU -> Dropout helper block."""

    def __init__(self, in_dim: int, out_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalConvBackbone(nn.Module):
    """Simple temporal stack mapping ``B x C x T`` -> ``B x hidden x T``."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = max(1, int(layers))
        blocks: list[nn.Module] = []
        current_in = in_channels
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Conv1d(
                        current_in,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=float(dropout)),
                ]
            )
            current_in = hidden_channels
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
