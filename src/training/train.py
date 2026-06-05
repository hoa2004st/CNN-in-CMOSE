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
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.evaluation.metrics import agreement_metrics_from_confusion, label_distance_metrics_from_confusion

ArrayInput = np.ndarray | tuple[np.ndarray, ...]


def fit_feature_normalizer(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature train-set mean/std statistics on ``n x frames x features`` arrays."""
    if matrices.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {matrices.shape}")

    mean = matrices.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = matrices.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std <= 0.0] = 1.0
    return mean, std


def normalize_dataset_per_feature(
    matrices: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    progress_desc: str | None = None,
    chunk_size: int = 64,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Normalize every sample using train-fit per-feature mean/std statistics."""
    if matrices.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {matrices.shape}")
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError("Expected 1-D mean/std arrays for per-feature normalization")
    if matrices.shape[2] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError(
            "Feature statistics shape mismatch: "
            f"matrices={matrices.shape}, mean={mean.shape}, std={std.shape}"
        )

    total_samples = matrices.shape[0]
    if total_samples == 0:
        return matrices.astype(np.float32, copy=True)

    normalized = np.empty_like(matrices, dtype=np.float32)
    chunk_size = max(1, int(chunk_size))
    desc = progress_desc or "Normalizing samples"
    mean_reshaped = mean.reshape(1, 1, -1)
    std_reshaped = std.reshape(1, 1, -1)

    for start_idx in tqdm(
        range(0, total_samples, chunk_size),
        desc=desc,
        unit="chunk",
        leave=False,
    ):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk = matrices[start_idx:end_idx].astype(np.float32, copy=False)

        chunk_out = normalized[start_idx:end_idx]
        np.subtract(chunk, mean_reshaped, out=chunk_out)
        np.divide(chunk_out, std_reshaped, out=chunk_out)

        if progress_callback is not None:
            progress_callback(end_idx, total_samples)

    return normalized


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
    device: torch.device,
) -> tuple[nn.Module, list[float] | None]:
    """Create the requested classification loss and optional class-weight summary."""
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(), None

    class_weights = compute_class_weights(y_train)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=weight_tensor), class_weights.tolist()
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
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    best_score = math.inf
    patience = max(1, int(patience))
    stale_epochs = 0

    history = {
        "train_losses": [],
        "eval_losses": [],
        "best_epoch": 0,
        "patience": patience,
        "stopped_early": False,
        "loss_name": loss_name,
        "class_weights": class_weights,
        "selection_metric": "eval_loss",
    }

    if progress_callback is not None:
        progress_callback(
            "Training setup complete: "
            f"device={device.type}, train_samples={_num_samples(X_train)}, "
            f"eval_samples={_num_samples(X_eval)}, "
            f"epochs={epochs}, batch_size={batch_size}, lr={lr}, patience={patience}, "
            f"loss={loss_name}, num_workers={num_workers}, amp={use_amp}"
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
                logits, aux_loss = _forward_model_with_aux(model, feature_batches, y_batch=y_batch)
                loss = criterion(logits, y_batch) + aux_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(y_batch)
            train_total += len(y_batch)

        eval_loss = _evaluate_loss(
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

        if math.isfinite(eval_loss) and best_score - eval_loss > min_delta:
            best_score = eval_loss
            history["best_epoch"] = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
            if progress_callback is not None:
                progress_callback(
                    f"Checkpoint updated: epoch={epoch}, eval_loss={eval_loss:.6f}"
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
                f"eval_loss={eval_loss:.6f}, "
                f"stale_epochs={stale_epochs}, epoch_time_s={epoch_seconds:.2f}"
            )

    if checkpoint_path.exists():
        if progress_callback is not None:
            progress_callback(
                f"Loading best checkpoint from epoch {history['best_epoch']} at {checkpoint_path}"
            )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise RuntimeError(
            f"No checkpoint was saved for this run at {checkpoint_path}. "
            "Training did not produce a finite selection score."
        )
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


def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
    pin_memory: bool = False,
) -> float:
    """Return the mean evaluation loss (the early-stopping selection metric).

    Only the loss is computed here; per-epoch classification metrics are
    intentionally not recorded during training.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for *feature_batches, y_batch in loader:
            feature_batches = [
                tensor.to(device, non_blocking=pin_memory) for tensor in feature_batches
            ]
            y_batch = y_batch.to(device, non_blocking=pin_memory)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, aux_loss = _forward_model_with_aux(model, feature_batches, y_batch=y_batch)
                loss = criterion(logits, y_batch) + aux_loss
            total_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)
    return total_loss / max(total_samples, 1)


def _compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "macro_accuracy": 0.0,
            "mae": 0.0,
            "macro_mae": 0.0,
            "cohen_kappa": 0.0,
            "quadratic_weighted_kappa": 0.0,
        }
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        **label_distance_metrics_from_confusion(confusion),
        **agreement_metrics_from_confusion(confusion),
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
    *,
    y_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass and return (main_logits, scalar_aux_loss).

    Handles two aux conventions:
    - Scalar tensor (e.g. reconstruction loss): returned as-is.
    - Dict of aux logits (SemanticGroupFusionModel): CE loss over each group
      is averaged and scaled by model.aux_weight.
    """
    if hasattr(model, "forward_with_aux"):
        if len(feature_batches) == 1:
            logits, aux = model.forward_with_aux(feature_batches[0])
        else:
            logits, aux = model.forward_with_aux(*feature_batches)
        if isinstance(aux, dict):
            if y_batch is None:
                return logits, logits.new_zeros(())
            aux_weight = float(getattr(model, "aux_weight", 0.2))
            aux_loss = aux_weight * torch.stack(
                [F.cross_entropy(a, y_batch) for a in aux.values()]
            ).mean()
            return logits, aux_loss
        return logits, aux
    logits = _forward_model(model, feature_batches)
    return logits, logits.new_zeros(())
