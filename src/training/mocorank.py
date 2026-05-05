"""MocoRank utilities: momentum update, score queue, and margin loss."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ScorePoolItem:
    label: int
    score: torch.Tensor
    embedding: torch.Tensor


class ScorePool:
    """FIFO score pool used for pairwise MocoRank constraints."""

    def __init__(self, *, max_size: int = 2048) -> None:
        self.max_size = int(max_size)
        self._items: deque[ScorePoolItem] = deque(maxlen=self.max_size)

    def __len__(self) -> int:
        return len(self._items)

    def push_batch(self, labels: torch.Tensor, scores: torch.Tensor, embeddings: torch.Tensor) -> None:
        for label, score, embedding in zip(labels, scores, embeddings):
            self._items.append(
                ScorePoolItem(
                    label=int(label.item()),
                    score=score.detach().clone(),
                    embedding=embedding.detach().clone(),
                )
            )

    def items(self) -> list[ScorePoolItem]:
        return list(self._items)


@torch.no_grad()
def update_momentum_encoder(
    model: torch.nn.Module,
    momentum_model: torch.nn.Module,
    *,
    momentum: float = 0.999,
) -> None:
    for param_model, param_momentum in zip(model.parameters(), momentum_model.parameters()):
        param_momentum.data.mul_(momentum).add_(param_model.data, alpha=1.0 - momentum)


def multi_margin_loss(
    scores_b: torch.Tensor,
    labels_b: torch.Tensor,
    embeddings_b: torch.Tensor,
    pool_items: list[ScorePoolItem],
) -> torch.Tensor:
    if scores_b.numel() == 0 or not pool_items:
        return scores_b.new_zeros(())

    losses: list[torch.Tensor] = []
    for s1, l1, e1 in zip(scores_b, labels_b, embeddings_b):
        for item in pool_items:
            s2 = item.score.to(device=s1.device, dtype=s1.dtype)
            e2 = item.embedding.to(device=e1.device, dtype=e1.dtype)
            l2 = int(item.label)

            cos_sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
            sim_scaled = (cos_sim + 1.0) / 2.0
            diff = abs(int(l1.item()) - l2)

            if diff == 0:
                margin_term = torch.abs(s1 - s2)
            elif diff == 1:
                m = 0.5 * sim_scaled
                margin_term = m - (s1 - s2) if int(l1.item()) > l2 else m - (s2 - s1)
            elif diff == 2:
                m = 0.5 + 0.5 * sim_scaled
                margin_term = m - (s1 - s2) if int(l1.item()) > l2 else m - (s2 - s1)
            else:
                m = 1.0 + 0.5 * sim_scaled
                margin_term = m - (s1 - s2) if int(l1.item()) > l2 else m - (s2 - s1)

            losses.append(torch.clamp(margin_term, min=0.0))

    if not losses:
        return scores_b.new_zeros(())
    return torch.stack(losses).mean()
