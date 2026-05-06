"""MocoRank utilities: momentum update, score queue, and margin loss."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


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
            sim_scaled = ((cos_sim + 1.0) / 2.0).squeeze()
            diff = abs(int(l1.item()) - l2)

            if diff == 0:
                margin_term = torch.abs(s1 - s2).squeeze()
            elif diff == 1:
                m = 0.5 * sim_scaled
                margin_term = m - (s1 - s2) if int(l1.item()) > l2 else m - (s2 - s1)
            elif diff == 2:
                m = 0.5 + 0.5 * sim_scaled
                margin_term = m - (s1 - s2) if int(l1.item()) > l2 else m - (s2 - s1)
            else:
                m = 1.0 + 0.5 * sim_scaled
                margin_term = m - (s1 - s2) if int(l1.item()) > l2 else m - (s2 - s1)

            losses.append(torch.clamp(margin_term.squeeze(), min=0.0))

    if not losses:
        return scores_b.new_zeros(())
    return torch.stack(losses).mean()


def score_to_class_tensor(scores: torch.Tensor) -> torch.Tensor:
    """Map scalar scores in [-1, 1] to the 4 CMOSE ordinal classes."""
    classes = torch.zeros_like(scores, dtype=torch.long)
    classes = torch.where(scores < -0.5, torch.zeros_like(classes), classes)
    classes = torch.where(
        (scores >= -0.5) & (scores < 0.0),
        torch.ones_like(classes),
        classes,
    )
    classes = torch.where(
        (scores >= 0.0) & (scores < 0.5),
        torch.full_like(classes, 2),
        classes,
    )
    classes = torch.where(scores >= 0.5, torch.full_like(classes, 3), classes)
    return classes


def _make_multimodal_loader(
    X_openface: np.ndarray,
    X_i3d: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X_openface).float(),
        torch.from_numpy(X_i3d).float(),
        torch.from_numpy(y).long(),
    )
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": pin_memory,
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def _evaluate_baseline_model(
    model: torch.nn.Module,
    X_openface: np.ndarray,
    X_i3d: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
) -> tuple[float, dict[str, float]]:
    model.eval()
    loader = _make_multimodal_loader(
        X_openface,
        X_i3d,
        y,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    score_batches: list[np.ndarray] = []
    class_batches: list[np.ndarray] = []
    for openface_batch, i3d_batch, _ in loader:
        openface_batch = openface_batch.to(device, non_blocking=pin_memory)
        i3d_batch = i3d_batch.to(device, non_blocking=pin_memory)
        scores = model(openface_batch, i3d_batch)
        preds = score_to_class_tensor(scores)
        score_batches.append(scores.detach().cpu().numpy())
        class_batches.append(preds.detach().cpu().numpy())

    all_scores = np.concatenate(score_batches) if score_batches else np.zeros(0, dtype=np.float32)
    all_preds = np.concatenate(class_batches) if class_batches else np.zeros(0, dtype=np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y, all_preds)) if y.size else 0.0,
        "macro_accuracy": float(balanced_accuracy_score(y, all_preds)) if y.size else 0.0,
        "f1_macro": float(f1_score(y, all_preds, average="macro", zero_division=0)) if y.size else 0.0,
        "f1_weighted": float(f1_score(y, all_preds, average="weighted", zero_division=0)) if y.size else 0.0,
    }
    return float(np.mean(np.abs(all_scores.astype(np.float32) - y.astype(np.float32)))), metrics


def train_cmose_baseline_mocorank(
    model: torch.nn.Module,
    X_train_openface: np.ndarray,
    X_train_i3d: np.ndarray,
    y_train: np.ndarray,
    X_eval_openface: np.ndarray,
    X_eval_i3d: np.ndarray,
    y_eval: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    checkpoint_path: str | Path,
    score_pool_size: int = 2048,
    momentum_update: float = 0.999,
    device: torch.device | None = None,
    num_workers: int = 0,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train the CMOSE baseline with MocoRank loss and momentum queue."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    model.to(device)
    momentum_model = deepcopy(model).to(device)
    momentum_model.eval()
    for param in momentum_model.parameters():
        param.requires_grad_(False)

    train_loader = _make_multimodal_loader(
        X_train_openface,
        X_train_i3d,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    pool = ScorePool(max_size=score_pool_size)
    with torch.no_grad():
        init_openface = torch.from_numpy(X_train_openface).float().to(device)
        init_i3d = torch.from_numpy(X_train_i3d).float().to(device)
        init_labels = torch.from_numpy(y_train).long().to(device)
        init_scores, init_embeddings = momentum_model(
            init_openface, init_i3d, return_embedding=True
        )
        pool.push_batch(init_labels, init_scores, init_embeddings)

    best_eval_mae = float("inf")
    stale_epochs = 0
    history: dict[str, Any] = {
        "train_losses": [],
        "eval_losses": [],
        "eval_accuracies": [],
        "eval_macro_accuracies": [],
        "eval_f1_macros": [],
        "eval_f1_weighteds": [],
        "best_epoch": 0,
        "patience": int(patience),
        "stopped_early": False,
        "loss_name": "mocorank",
    }

    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        for openface_batch, i3d_batch, y_batch in train_loader:
            openface_batch = openface_batch.to(device, non_blocking=pin_memory)
            i3d_batch = i3d_batch.to(device, non_blocking=pin_memory)
            y_batch = y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            scores, embeddings = model(openface_batch, i3d_batch, return_embedding=True)
            rank_loss = multi_margin_loss(scores, y_batch, embeddings, pool.items())
            rank_loss.backward()
            optimizer.step()

            with torch.no_grad():
                update_momentum_encoder(model, momentum_model, momentum=momentum_update)
                mom_scores, mom_embeddings = momentum_model(
                    openface_batch, i3d_batch, return_embedding=True
                )
                pool.push_batch(y_batch, mom_scores, mom_embeddings)

            epoch_loss_sum += float(rank_loss.item()) * len(y_batch)
            epoch_samples += len(y_batch)

        train_loss = epoch_loss_sum / max(epoch_samples, 1)
        eval_loss, eval_metrics = _evaluate_baseline_model(
            model,
            X_eval_openface,
            X_eval_i3d,
            y_eval,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        history["train_losses"].append(train_loss)
        history["eval_losses"].append(eval_loss)
        history["eval_accuracies"].append(eval_metrics["accuracy"])
        history["eval_macro_accuracies"].append(eval_metrics["macro_accuracy"])
        history["eval_f1_macros"].append(eval_metrics["f1_macro"])
        history["eval_f1_weighteds"].append(eval_metrics["f1_weighted"])

        if eval_loss < best_eval_mae:
            best_eval_mae = eval_loss
            stale_epochs = 0
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                history["stopped_early"] = True
                break

        if progress_callback is not None:
            progress_callback(
                f"Baseline epoch {epoch}/{epochs}: train_loss={train_loss:.6f} eval_mae={eval_loss:.6f}"
            )

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


@torch.no_grad()
def predict_cmose_baseline(
    model: torch.nn.Module,
    X_openface: np.ndarray,
    X_i3d: np.ndarray,
    *,
    batch_size: int,
    device: torch.device | None = None,
    num_workers: int = 0,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    model.to(device)
    model.eval()

    loader = _make_multimodal_loader(
        X_openface,
        X_i3d,
        np.zeros(len(X_openface), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    preds: list[np.ndarray] = []
    for openface_batch, i3d_batch, _ in loader:
        openface_batch = openface_batch.to(device, non_blocking=pin_memory)
        i3d_batch = i3d_batch.to(device, non_blocking=pin_memory)
        scores = model(openface_batch, i3d_batch)
        preds.append(score_to_class_tensor(scores).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)
