"""CNN architecture matching the paper summary."""

from __future__ import annotations

import torch
import torch.nn as nn


class PaperEngagementCNN(nn.Module):
    """4-block CNN with the paper's filter sizes and dropout pattern."""

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
