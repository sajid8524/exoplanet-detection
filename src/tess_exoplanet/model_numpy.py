from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .types import INDEX_TO_LABEL, LABELS


@dataclass
class NumpyMLP:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    labels: list[str]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = self._transform(X)
        hidden = np.maximum(0.0, Xn @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        return softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, out_dir: str | Path) -> Path:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_path = out_path / "model.npz"
        np.savez(
            model_path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            mean=self.mean,
            std=self.std,
            labels=np.asarray(self.labels),
        )
        return model_path

    @classmethod
    def load(cls, model_path: str | Path) -> "NumpyMLP":
        data = np.load(model_path, allow_pickle=False)
        return cls(
            W1=data["W1"],
            b1=data["b1"],
            W2=data["W2"],
            b2=data["b2"],
            mean=data["mean"],
            std=data["std"],
            labels=[str(item) for item in data["labels"].tolist()],
        )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return (X - self.mean) / self.std


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.01,
    hidden_units: int = 96,
    validation_fraction: float = 0.2,
    seed: int = 42,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
) -> tuple[NumpyMLP, list[dict[str, float]]]:
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = stratified_split(y, validation_fraction, rng)

    mean = np.nanmean(X[train_idx], axis=0)
    std = np.nanstd(X[train_idx], axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    Xn = (X - mean) / std

    n_features = Xn.shape[1]
    n_classes = len(LABELS)
    W1 = rng.normal(0.0, np.sqrt(2.0 / max(n_features, 1)), size=(n_features, hidden_units))
    b1 = np.zeros(hidden_units)
    W2 = rng.normal(0.0, np.sqrt(2.0 / max(hidden_units, 1)), size=(hidden_units, n_classes))
    b2 = np.zeros(n_classes)

    class_weights = balanced_class_weights(y[train_idx], n_classes)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[start : start + batch_size]
            xb = Xn[batch_idx]
            yb = y[batch_idx]
            weights = class_weights[yb]

            hidden = np.maximum(0.0, xb @ W1 + b1)
            logits = hidden @ W2 + b2
            probs = softmax(logits)
            target = one_hot(yb, n_classes)

            normalizer = max(float(np.sum(weights)), 1e-8)
            d_logits = (probs - target) * weights[:, None] / normalizer
            dW2 = hidden.T @ d_logits + weight_decay * W2
            db2 = np.sum(d_logits, axis=0)
            d_hidden = d_logits @ W2.T
            d_hidden[hidden <= 0] = 0.0
            dW1 = xb.T @ d_hidden + weight_decay * W1
            db1 = np.sum(d_hidden, axis=0)

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        train_probs = forward(Xn[train_idx], W1, b1, W2, b2)
        val_probs = forward(Xn[val_idx], W1, b1, W2, b2) if len(val_idx) else train_probs
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": cross_entropy(train_probs, y[train_idx]),
                "train_accuracy": accuracy(train_probs, y[train_idx]),
                "val_loss": cross_entropy(val_probs, y[val_idx]) if len(val_idx) else np.nan,
                "val_accuracy": accuracy(val_probs, y[val_idx]) if len(val_idx) else np.nan,
            }
        )

    model = NumpyMLP(W1=W1, b1=b1, W2=W2, b2=b2, mean=mean, std=std, labels=list(LABELS))
    return model, history


def write_history(history: list[dict[str, float]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return softmax(np.maximum(0.0, X @ W1 + b1) @ W2 + b2)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    chosen = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(chosen, 1e-9, 1.0))))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.mean(np.argmax(probs, axis=1) == y))


def balanced_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts > 0, counts, 1.0)
    return len(y) / (n_classes * counts)


def stratified_split(
    y: np.ndarray, validation_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * validation_fraction))
        if len(idx) > 1:
            n_val = min(max(1, n_val), len(idx) - 1)
        else:
            n_val = 0
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts) if train_parts else np.arange(len(y))
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def label_names(indices: np.ndarray) -> list[str]:
    return [INDEX_TO_LABEL[int(idx)] for idx in indices]

