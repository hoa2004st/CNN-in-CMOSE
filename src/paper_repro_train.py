"""Training helpers for the paper-faithful reproduction pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def split_after_smote(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the paper's final 80/20 split after balancing."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    min_delta: float = 0.0,
) -> dict:
    """Train using Adam, checkpointing, and early stopping.

    The paper states patience equals half the number of epochs.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    eval_loader = make_dataloader(X_eval, y_eval, batch_size=batch_size, shuffle=False)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = math.inf
    patience = max(1, epochs // 2)
    stale_epochs = 0

    history = {
        "train_losses": [],
        "eval_losses": [],
        "eval_accuracies": [],
        "best_epoch": 0,
        "patience": patience,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_total += len(y_batch)

        eval_loss, eval_acc = _evaluate_loss_accuracy(model, eval_loader, criterion, device)
        train_loss /= max(train_total, 1)

        history["train_losses"].append(train_loss)
        history["eval_losses"].append(eval_loss)
        history["eval_accuracies"].append(eval_acc)

        if best_loss - eval_loss > min_delta:
            best_loss = eval_loss
            history["best_epoch"] = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


def predict(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = make_dataloader(
        X,
        np.zeros(len(X), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _ in loader:
            logits = model(X_batch.to(device))
            outputs.append(logits.argmax(dim=1).cpu().numpy())
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
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += len(y_batch)
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)
