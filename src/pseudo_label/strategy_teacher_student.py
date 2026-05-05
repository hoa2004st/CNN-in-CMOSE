"""Teacher-student pseudo-labeling helpers with EMA updates."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def ema_update(student: torch.nn.Module, teacher: torch.nn.Module, *, momentum: float = 0.999) -> None:
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)


def consistency_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_scores, teacher_scores.detach())
