"""Training helpers for the paper-faithful reproduction pipeline."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if weight is None:
            self.register_buffer("weight", None, persistent=False)
        else:
            self.register_buffer("weight", weight.float(), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency class weights normalized to mean 1."""
    class_counts = np.bincount(y.astype(np.int64))
    if class_counts.size == 0:
        raise ValueError("Cannot compute class weights for an empty target array")

    weights = np.zeros_like(class_counts, dtype=np.float32)
    nonzero = class_counts > 0
    weights[nonzero] = 1.0 / class_counts[nonzero].astype(np.float32)
    if np.any(nonzero):
        weights[nonzero] *= float(nonzero.sum()) / float(weights[nonzero].sum())
    return weights


def build_loss(
    y_train: np.ndarray,
    *,
    loss_name: str,
    focal_gamma: float = 2.0,
    device: torch.device,
) -> tuple[nn.Module, list[float] | None]:
    """Create the requested classification loss and optional class-weight summary."""
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(), None

    class_weights = compute_class_weights(y_train)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=weight_tensor), class_weights.tolist()
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=weight_tensor), class_weights.tolist()
    raise ValueError(f"Unknown loss_name: {loss_name}")


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": pin_memory,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    loss_name: str,
    focal_gamma: float,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    min_delta: float = 0.0,
    epoch_log_interval: int = 1,
    num_workers: int = 0,
    use_amp: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Train using Adam, checkpointing, and configurable early stopping."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_log_interval = max(1, int(epoch_log_interval))
    use_amp = bool(use_amp and device.type == "cuda")
    pin_memory = device.type == "cuda"
    model.to(device)
    criterion, class_weights = build_loss(
        y_train,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader = make_dataloader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = make_dataloader(
        X_eval,
        y_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = math.inf
    patience = max(1, int(patience))
    stale_epochs = 0

    history = {
        "train_losses": [],
        "eval_losses": [],
        "eval_accuracies": [],
        "best_epoch": 0,
        "patience": patience,
        "stopped_early": False,
        "loss_name": loss_name,
        "focal_gamma": focal_gamma if loss_name == "focal" else None,
        "class_weights": class_weights,
    }

    if progress_callback is not None:
        progress_callback(
            "Training setup complete: "
            f"device={device.type}, train_samples={len(X_train)}, eval_samples={len(X_eval)}, "
            f"epochs={epochs}, batch_size={batch_size}, lr={lr}, patience={patience}, "
            f"loss={loss_name}, focal_gamma={focal_gamma}, num_workers={num_workers}, amp={use_amp}"
        )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=pin_memory)
            y_batch = y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(y_batch)
            train_total += len(y_batch)

        eval_loss, eval_acc = _evaluate_loss_accuracy(
            model,
            eval_loader,
            criterion,
            device,
            use_amp=use_amp,
            pin_memory=pin_memory,
        )
        train_loss /= max(train_total, 1)

        history["train_losses"].append(train_loss)
        history["eval_losses"].append(eval_loss)
        history["eval_accuracies"].append(eval_acc)

        if best_loss - eval_loss > min_delta:
            best_loss = eval_loss
            history["best_epoch"] = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
            if progress_callback is not None:
                progress_callback(
                    "Checkpoint updated: "
                    f"epoch={epoch}, eval_loss={eval_loss:.6f}, eval_acc={eval_acc:.4f}"
                )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                history["stopped_early"] = True
                if progress_callback is not None:
                    progress_callback(
                        "Early stopping triggered: "
                        f"epoch={epoch}, best_epoch={history['best_epoch']}, "
                        f"stale_epochs={stale_epochs}"
                    )
                break

        if progress_callback is not None and (
            epoch == 1 or epoch % epoch_log_interval == 0 or epoch == epochs
        ):
            epoch_seconds = time.perf_counter() - epoch_start
            progress_callback(
                "Epoch complete: "
                f"{epoch}/{epochs}, train_loss={train_loss:.6f}, "
                f"eval_loss={eval_loss:.6f}, eval_acc={eval_acc:.4f}, "
                f"stale_epochs={stale_epochs}, epoch_time_s={epoch_seconds:.2f}"
            )

    if checkpoint_path.exists():
        if progress_callback is not None:
            progress_callback(
                f"Loading best checkpoint from epoch {history['best_epoch']} at {checkpoint_path}"
            )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


def predict(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int = 8,
    device: torch.device | None = None,
    num_workers: int = 0,
    use_amp: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(use_amp and device.type == "cuda")
    pin_memory = device.type == "cuda"
    model.to(device)
    model.eval()

    loader = make_dataloader(
        X,
        np.zeros(len(X), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    outputs: list[np.ndarray] = []
    if progress_callback is not None:
        progress_callback(
            "Starting prediction: "
            f"samples={len(X)}, batch_size={batch_size}, device={device.type}, "
            f"num_workers={num_workers}, amp={use_amp}"
        )
    with torch.no_grad():
        total_batches = len(loader)
        for batch_idx, (X_batch, _) in enumerate(loader, start=1):
            X_batch = X_batch.to(device, non_blocking=pin_memory)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X_batch)
            outputs.append(logits.argmax(dim=1).cpu().numpy())
            if progress_callback is not None and (
                batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0
            ):
                progress_callback(f"Prediction progress: batch {batch_idx}/{total_batches}")
    return np.concatenate(outputs)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _evaluate_loss_accuracy(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
    pin_memory: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=pin_memory)
            y_batch = y_batch.to(device, non_blocking=pin_memory)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += len(y_batch)
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)
