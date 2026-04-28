"""Training helpers for the narrowed CMOSE comparison pipeline."""

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

ArrayInput = np.ndarray | tuple[np.ndarray, ...]


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


class OrdinalEMDLoss(nn.Module):
    """Ordinal loss via squared CDF distance between predicted and target classes."""

    def __init__(self, *, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if weight is None:
            self.register_buffer("weight", None, persistent=False)
        else:
            self.register_buffer("weight", weight.float(), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        pred_cdf = torch.cumsum(probs, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        target_cdf = torch.cumsum(target_one_hot, dim=1)
        loss_per_sample = torch.mean((pred_cdf - target_cdf) ** 2, dim=1)
        if self.weight is not None:
            loss_per_sample = loss_per_sample * self.weight[targets]
        return loss_per_sample.mean()


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
    if loss_name == "ordinal":
        return OrdinalEMDLoss(weight=weight_tensor), class_weights.tolist()
    raise ValueError(f"Unknown loss_name: {loss_name}")


def make_dataloader(
    X: ArrayInput,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    feature_arrays = _as_feature_arrays(X)
    sample_count = _num_samples(X)
    if sample_count != len(y):
        raise ValueError(f"Feature/label size mismatch: X={sample_count}, y={len(y)}")

    dataset = TensorDataset(
        *(torch.from_numpy(array).float() for array in feature_arrays),
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
    X_train: ArrayInput,
    y_train: np.ndarray,
    X_eval: ArrayInput,
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
        "eval_macro_accuracies": [],
        "eval_f1_macros": [],
        "eval_f1_weighteds": [],
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
            f"device={device.type}, train_samples={_num_samples(X_train)}, "
            f"eval_samples={_num_samples(X_eval)}, "
            f"epochs={epochs}, batch_size={batch_size}, lr={lr}, patience={patience}, "
            f"loss={loss_name}, focal_gamma={focal_gamma}, num_workers={num_workers}, amp={use_amp}"
        )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_total = 0
        for *feature_batches, y_batch in train_loader:
            feature_batches = [
                tensor.to(device, non_blocking=pin_memory) for tensor in feature_batches
            ]
            y_batch = y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, aux_loss = _forward_model_with_aux(model, feature_batches)
                loss = criterion(logits, y_batch) + aux_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(y_batch)
            train_total += len(y_batch)

        eval_loss, eval_metrics = _evaluate_loss_and_metrics(
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
        history["eval_accuracies"].append(eval_metrics["accuracy"])
        history["eval_macro_accuracies"].append(eval_metrics["macro_accuracy"])
        history["eval_f1_macros"].append(eval_metrics["f1_macro"])
        history["eval_f1_weighteds"].append(eval_metrics["f1_weighted"])

        if best_loss - eval_loss > min_delta:
            best_loss = eval_loss
            history["best_epoch"] = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
            if progress_callback is not None:
                progress_callback(
                    "Checkpoint updated: "
                    f"epoch={epoch}, eval_loss={eval_loss:.6f}, "
                    f"eval_acc={eval_metrics['accuracy']:.4f}, "
                    f"eval_macro_acc={eval_metrics['macro_accuracy']:.4f}, "
                    f"eval_f1_macro={eval_metrics['f1_macro']:.4f}, "
                    f"eval_f1_weighted={eval_metrics['f1_weighted']:.4f}"
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
                f"eval_loss={eval_loss:.6f}, eval_acc={eval_metrics['accuracy']:.4f}, "
                f"eval_macro_acc={eval_metrics['macro_accuracy']:.4f}, "
                f"eval_f1_macro={eval_metrics['f1_macro']:.4f}, "
                f"eval_f1_weighted={eval_metrics['f1_weighted']:.4f}, "
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
    X: ArrayInput,
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
        np.zeros(_num_samples(X), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    outputs: list[np.ndarray] = []
    if progress_callback is not None:
        progress_callback(
            "Starting prediction: "
            f"samples={_num_samples(X)}, batch_size={batch_size}, device={device.type}, "
            f"num_workers={num_workers}, amp={use_amp}"
        )
    with torch.no_grad():
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            *feature_batches, _ = batch
            feature_batches = [
                tensor.to(device, non_blocking=pin_memory) for tensor in feature_batches
            ]
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = _forward_model(model, feature_batches)
            outputs.append(logits.argmax(dim=1).cpu().numpy())
            if progress_callback is not None and (
                batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0
            ):
                progress_callback(f"Prediction progress: batch {batch_idx}/{total_batches}")
    return np.concatenate(outputs)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        **_compute_prediction_metrics(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _evaluate_loss_and_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
    pin_memory: bool = False,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []
    with torch.no_grad():
        for *feature_batches, y_batch in loader:
            feature_batches = [
                tensor.to(device, non_blocking=pin_memory) for tensor in feature_batches
            ]
            y_batch = y_batch.to(device, non_blocking=pin_memory)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, aux_loss = _forward_model_with_aux(model, feature_batches)
                loss = criterion(logits, y_batch) + aux_loss
            total_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)
            y_true_batches.append(y_batch.cpu().numpy())
            y_pred_batches.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(y_true_batches) if y_true_batches else np.zeros(0, dtype=np.int64)
    y_pred = np.concatenate(y_pred_batches) if y_pred_batches else np.zeros(0, dtype=np.int64)
    return total_loss / max(total_samples, 1), _compute_prediction_metrics(y_true, y_pred)


def _compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "macro_accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _as_feature_arrays(X: ArrayInput) -> tuple[np.ndarray, ...]:
    if isinstance(X, tuple):
        return X
    return (X,)


def _num_samples(X: ArrayInput) -> int:
    feature_arrays = _as_feature_arrays(X)
    if not feature_arrays:
        return 0
    sample_count = len(feature_arrays[0])
    for array in feature_arrays[1:]:
        if len(array) != sample_count:
            raise ValueError("All feature arrays must contain the same number of samples")
    return sample_count


def _forward_model(model: nn.Module, feature_batches: list[torch.Tensor]) -> torch.Tensor:
    if len(feature_batches) == 1:
        return model(feature_batches[0])
    return model(*feature_batches)


def _forward_model_with_aux(
    model: nn.Module,
    feature_batches: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_with_aux"):
        if len(feature_batches) == 1:
            logits, aux_loss = model.forward_with_aux(feature_batches[0])
        else:
            logits, aux_loss = model.forward_with_aux(*feature_batches)
        return logits, aux_loss
    logits = _forward_model(model, feature_batches)
    return logits, logits.new_zeros(())
