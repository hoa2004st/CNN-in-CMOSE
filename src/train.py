"""Training loop for the CNN engagement detection model.

Supports:
* Class-weighted cross-entropy loss to handle any residual class imbalance in
  the CMOSE dataset.
* Learning-rate scheduling (ReduceLROnPlateau on validation loss).
* Early stopping based on validation loss.
* Saving the best checkpoint.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import EngagementCNN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap numpy arrays in a :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    X:
        Feature tensor of shape ``(n_samples, 1, grid_h, grid_w)``.
    y:
        Integer label array of shape ``(n_samples,)``.
    batch_size:
        Mini-batch size.
    shuffle:
        Whether to shuffle samples before each epoch.

    Returns
    -------
    DataLoader
    """
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(y: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted cross-entropy.

    Parameters
    ----------
    y:
        Integer label array.
    num_classes:
        Total number of classes.

    Returns
    -------
    torch.Tensor of shape ``(num_classes,)`` on the CPU.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.from_numpy(weights).float()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: EngagementCNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_class_weights: bool = True,
    patience: int = 10,
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Train the CNN model.

    Parameters
    ----------
    model:
        The :class:`~src.model.EngagementCNN` instance.
    X_train, y_train:
        Training features ``(n, 1, H, W)`` and integer labels ``(n,)``.
    X_val, y_val:
        Validation features and labels.
    epochs:
        Maximum number of epochs.
    batch_size:
        Mini-batch size.
    lr:
        Initial learning rate for the Adam optimiser.
    weight_decay:
        L2 regularisation coefficient.
    use_class_weights:
        Weight the cross-entropy loss by inverse class frequency.
    patience:
        Number of epochs with no improvement after which training is stopped.
    checkpoint_path:
        If provided, the best model state dict is saved to this file.
    device:
        Torch device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    dict with keys ``train_losses``, ``val_losses``, ``val_accuracies``, and
    ``best_epoch``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    if use_class_weights:
        class_weights = compute_class_weights(y_train, num_classes=model.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    history: dict = {
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "best_epoch": 0,
    }

    best_val_loss = math.inf
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(y_train)

        # ── Validate ─────────────────────────────────────────────────────
        val_loss, val_acc = _evaluate_loss_accuracy(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch, epochs, train_loss, val_loss, val_acc,
        )

        # ── Early stopping / checkpoint ───────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            patience_counter = 0
            if checkpoint_path is not None:
                _save_checkpoint(model, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    return history


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate_loss_accuracy(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / total, correct / total


def _save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Checkpoint saved to %s", path)
