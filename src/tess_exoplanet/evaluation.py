from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import LABELS


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    n = n_classes or len(LABELS)
    matrix = np.zeros((n, n), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, len(LABELS))
    per_class: dict[str, dict[str, float]] = {}
    for idx, label in enumerate(LABELS):
        tp = float(matrix[idx, idx])
        fp = float(np.sum(matrix[:, idx]) - tp)
        fn = float(np.sum(matrix[idx, :]) - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(np.sum(matrix[idx, :])),
        }

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    return {
        "accuracy": accuracy,
        "labels": LABELS,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def write_evaluation(report: dict[str, object], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

