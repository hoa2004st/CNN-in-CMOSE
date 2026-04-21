"""CNN model for student engagement detection.

Architecture
------------
The model takes a single-channel 2-D feature map (height × width) produced by
reshaping the PCA-projected clip-level features and classifies it into one of
four engagement levels (0 = not engaged … 3 = highly engaged).

The network consists of two convolutional blocks followed by a fully connected
head.  Each convolutional block applies ``Conv2d → BatchNorm → ReLU →
MaxPool2d``.  The head applies ``Dropout → Linear → ReLU → Linear → Softmax``.

This matches the design described in
*"CNN Model based Student Engagement Detection in Imbalanced DAiSEE Dataset"*
and is adapted here for features of arbitrary spatial size produced from the
PCA step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 4  # 0: not engaged, 1: barely, 2: engaged, 3: highly engaged


class EngagementCNN(nn.Module):
    """Convolutional Neural Network for engagement classification.

    Parameters
    ----------
    grid_h, grid_w:
        Height and width of the 2-D feature map input (after PCA reshape).
        Together they define the spatial input size: ``(1, grid_h, grid_w)``.
    num_classes:
        Number of output classes (default: 4).
    dropout_p:
        Dropout probability applied before the first fully connected layer
        (default: 0.5).
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        num_classes: int = NUM_CLASSES,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_classes = num_classes

        # ── Convolutional block 1 ─────────────────────────────────────────
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32,
            kernel_size=3, padding=1,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # ── Convolutional block 2 ─────────────────────────────────────────
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=3, padding=1,
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # ── Fully connected head ──────────────────────────────────────────
        flat_size = self._get_flat_size(grid_h, grid_w)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, 1, grid_h, grid_w)``.

        Returns
        -------
        torch.Tensor of shape ``(batch, num_classes)`` (raw logits).
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    # ------------------------------------------------------------------

    def _get_flat_size(self, h: int, w: int) -> int:
        """Compute the flattened dimension after the convolutional blocks."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            dummy = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool2(F.relu(self.bn2(self.conv2(dummy))))
            return int(dummy.numel())
