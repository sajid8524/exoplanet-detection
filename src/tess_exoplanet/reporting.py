from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .features import phase_fold
from .io import read_lightcurve_csv, resolve_curve_path
from .preprocessing import preprocess_lightcurve
from .types import Candidate, LABELS


def write_predictions(
    path: str | Path,
    records: list[dict[str, object]],
    probs: np.ndarray,
    labels: Iterable[str] = LABELS,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    labels = list(labels)
    for idx, label in enumerate(labels):
        frame[f"prob_{label}"] = probs[:, idx]
    pred_idx = np.argmax(probs, axis=1)
    frame["predicted_label"] = [labels[int(idx)] for idx in pred_idx]
    frame["confidence"] = np.max(probs, axis=1)
    frame.to_csv(out_path, index=False)
    return out_path


def write_report(
    out_dir: str | Path,
    predictions_path: str | Path,
    evaluation_path: str | Path | None = None,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(predictions_path)
    evaluation = None
    if evaluation_path and Path(evaluation_path).exists():
        evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))

    top = predictions.sort_values("prob_planet", ascending=False).head(10)
    lines: list[str] = []
    lines.append("# TESS Exoplanet Detection Report")
    lines.append("")
    lines.append("## Pipeline Summary")
    lines.append("")
    lines.append("- Preprocessing: quality filtering, median normalization, rolling detrending, sigma clipping.")
    lines.append("- Detection: BLS-style periodic box search for period, epoch, depth, duration, and SNR.")
    lines.append("- Classification: global/local folded view features plus vetting scalars.")
    lines.append("- Vetting: odd-even depth mismatch, secondary eclipse, centroid shift, depth, and SNR flags.")
    lines.append("")

    if evaluation:
        lines.append("## Validation Metrics")
        lines.append("")
        lines.append(f"- Accuracy: {evaluation['accuracy']:.3f}")
        lines.append("- Confusion matrix rows=true, columns=predicted:")
        lines.append("")
        lines.append("| class | " + " | ".join(evaluation["labels"]) + " |")
        lines.append("|---|" + "|".join(["---"] * len(evaluation["labels"])) + "|")
        for label, row in zip(evaluation["labels"], evaluation["confusion_matrix"]):
            lines.append(f"| {label} | " + " | ".join(str(x) for x in row) + " |")
        lines.append("")

    lines.append("## Top Planet Candidates")
    lines.append("")
    columns = [
        "target_id",
        "prob_planet",
        "confidence",
        "period",
        "duration",
        "depth",
        "snr",
        "radius_ratio",
        "flags",
    ]
    present = [col for col in columns if col in top.columns]
    lines.append("| " + " | ".join(present) + " |")
    lines.append("|" + "|".join(["---"] * len(present)) + "|")
    for _, row in top.iterrows():
        values = []
        for col in present:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.5g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Expected Final Submission Artifacts")
    lines.append("")
    lines.append("- Prediction CSV for official test targets.")
    lines.append("- Three-page methodology report.")
    lines.append("- Raw-vs-cleaned and folded-transit plots.")
    lines.append("- Confusion matrix and per-class metrics on validation data.")
    lines.append("- Candidate ranking table with false-positive vetting flags.")

    report_path = out_path / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_demo_plots(metadata_path: str | Path, out_dir: str | Path) -> None:
    metadata = pd.read_csv(metadata_path)
    candidates = metadata[metadata["label"].isin(["planet", "eclipsing_binary", "background_blend"])]
    if candidates.empty:
        return
    row = candidates.iloc[0]
    curve_path = resolve_curve_path(metadata_path, str(row["path"]))
    raw = read_lightcurve_csv(curve_path, target_id=str(row["target_id"]))
    clean, _trend = preprocess_lightcurve(raw)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plot_series_svg(
        out_path / "example_raw_cleaned.svg",
        np.asarray(raw.time),
        [
            ("raw", np.asarray(raw.flux)),
            ("cleaned", _resample_to_x(np.asarray(clean.time), np.asarray(clean.flux), np.asarray(raw.time))),
        ],
        title="Raw vs Cleaned Light Curve",
        x_label="time days",
        y_label="relative flux",
    )

    candidate = Candidate(
        target_id=str(row["target_id"]),
        period=float(row["period"]),
        epoch=float(row["epoch"]),
        duration=float(row["duration"]),
        depth=float(row.get("true_depth", 0.0)),
        snr=0.0,
        score=0.0,
        phase_center=0.0,
        n_transits=0,
    )
    phase = phase_fold(np.asarray(clean.time), candidate.period, candidate.epoch)
    order = np.argsort(phase)
    plot_series_svg(
        out_path / "example_folded.svg",
        phase[order],
        [("folded", np.asarray(clean.flux)[order])],
        title="Phase-Folded Candidate",
        x_label="phase",
        y_label="relative flux",
    )


def plot_series_svg(
    path: str | Path,
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 900,
    height: int = 420,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=float)
    y_values = np.concatenate([np.asarray(values, dtype=float) for _, values in series])
    valid_x = x[np.isfinite(x)]
    valid_y = y_values[np.isfinite(y_values)]
    if valid_x.size == 0 or valid_y.size == 0:
        return out_path

    xmin, xmax = float(np.min(valid_x)), float(np.max(valid_x))
    ymin, ymax = float(np.min(valid_y)), float(np.max(valid_y))
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    ypad = 0.08 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    margin = 58
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    def sx(value: float) -> float:
        return margin + (value - xmin) / (xmax - xmin) * plot_w

    def sy(value: float) -> float:
        return margin + (ymax - value) / (ymax - ymin) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" fill="#111">{_esc(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>',
        f'<text x="{width / 2}" y="{height - 14}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{_esc(x_label)}</text>',
        f'<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="Arial" font-size="12" fill="#333">{_esc(y_label)}</text>',
    ]

    for idx, (name, values) in enumerate(series):
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(x) & np.isfinite(values)
        if np.sum(mask) < 2:
            continue
        points = " ".join(f"{sx(float(xv)):.2f},{sy(float(yv)):.2f}" for xv, yv in zip(x[mask], values[mask]))
        color = colors[idx % len(colors)]
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.4"/>')
        legend_x = margin + 12 + idx * 130
        lines.append(f'<rect x="{legend_x}" y="42" width="18" height="3" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 24}" y="48" font-family="Arial" font-size="12" fill="#222">{_esc(name)}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _resample_to_x(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(source_x) & np.isfinite(source_y)
    if np.sum(mask) < 2:
        return np.full_like(target_x, np.nan, dtype=float)
    return np.interp(target_x, source_x[mask], source_y[mask], left=np.nan, right=np.nan)


def _esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
