from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import write_lightcurve_csv
from .types import LightCurve


KAGGLE_LABEL_MAP = {
    1: "noise",
    2: "planet",
}


def convert_kaggle_kepler(
    train_csv: str | Path | None,
    test_csv: str | Path | None,
    out_dir: str | Path,
    cadence_minutes: float = 29.4,
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
) -> dict[str, Path]:
    """Convert Kaggle Kepler wide-row CSVs into standard light-curve files.

    The Kaggle dataset stores one target per row:

    ``LABEL,FLUX.1,FLUX.2,...``

    This converter writes:

    ``out_dir/train/lightcurves/<target>.csv``
    ``out_dir/train/metadata.csv``

    and the same for the test split when provided.
    """
    out_path = Path(out_dir)
    outputs: dict[str, Path] = {}
    if train_csv:
        outputs["train"] = convert_kaggle_split(
            train_csv,
            out_path / "train",
            split_name="train",
            cadence_minutes=cadence_minutes,
            max_rows=max_train_rows,
        )
    if test_csv:
        outputs["test"] = convert_kaggle_split(
            test_csv,
            out_path / "test",
            split_name="test",
            cadence_minutes=cadence_minutes,
            max_rows=max_test_rows,
        )
    return outputs


def convert_kaggle_split(
    csv_path: str | Path,
    out_dir: str | Path,
    split_name: str,
    cadence_minutes: float = 29.4,
    max_rows: int | None = None,
) -> Path:
    csv_path = Path(csv_path)
    out_path = Path(out_dir)
    curve_dir = out_path / "lightcurves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(csv_path, nrows=max_rows)
    label_col = _find_label_column(frame.columns)
    flux_cols = _flux_columns(frame.columns)
    if label_col is None or not flux_cols:
        raise ValueError(
            f"{csv_path} must contain a LABEL column and FLUX.* columns."
        )

    time = np.arange(len(flux_cols), dtype=float) * cadence_minutes / (24.0 * 60.0)
    metadata_rows: list[dict[str, object]] = []
    for row_idx, row in frame.iterrows():
        raw_label = int(row[label_col])
        label = KAGGLE_LABEL_MAP.get(raw_label)
        if label is None:
            raise ValueError(f"Unknown Kaggle label {raw_label} in {csv_path}.")

        target_id = f"KEPLER_{split_name.upper()}_{row_idx:05d}"
        flux = _kaggle_flux_to_relative(row[flux_cols].to_numpy(dtype=float))
        lightcurve = LightCurve(target_id=target_id, time=time, flux=flux)
        rel_path = Path("lightcurves") / f"{target_id}.csv"
        write_lightcurve_csv(curve_dir / f"{target_id}.csv", lightcurve)
        metadata_rows.append(
            {
                "target_id": target_id,
                "path": rel_path.as_posix(),
                "label": label,
                "source_dataset": "kaggle_kepler_labelled_time_series",
                "source_file": csv_path.name,
                "source_row": int(row_idx),
                "kaggle_label": raw_label,
                "cadence_minutes": cadence_minutes,
            }
        )

    metadata_path = out_path / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    return metadata_path


def describe_kaggle_csv(csv_path: str | Path, nrows: int | None = None) -> dict[str, object]:
    frame = pd.read_csv(csv_path, nrows=nrows)
    label_col = _find_label_column(frame.columns)
    flux_cols = _flux_columns(frame.columns)
    label_counts = {}
    if label_col is not None:
        label_counts = {
            str(KAGGLE_LABEL_MAP.get(int(label), label)): int(count)
            for label, count in frame[label_col].value_counts().sort_index().items()
        }
    return {
        "rows": int(len(frame)),
        "label_column": label_col,
        "flux_columns": int(len(flux_cols)),
        "label_counts": label_counts,
    }


def _find_label_column(columns: Iterable[str]) -> str | None:
    for column in columns:
        if column.strip().upper() == "LABEL":
            return column
    return None


def _flux_columns(columns: Iterable[str]) -> list[str]:
    def flux_index(name: str) -> int:
        try:
            return int(name.split(".", 1)[1])
        except (IndexError, ValueError):
            return 0

    names = [column for column in columns if column.strip().upper().startswith("FLUX")]
    return sorted(names, key=flux_index)


def _kaggle_flux_to_relative(flux: np.ndarray) -> np.ndarray:
    """Map Kaggle Kepler flux deviations onto a relative-flux scale near 1."""
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(flux)
    if not np.any(finite):
        return np.ones_like(flux, dtype=float)

    center = float(np.nanmedian(flux[finite]))
    p05, p95 = np.nanpercentile(flux[finite], [5, 95])
    robust_span = float(p95 - p05)
    scale = max(abs(center), robust_span, float(np.nanstd(flux[finite])), 1.0)
    relative = 1.0 + (flux - center) / scale
    return np.nan_to_num(relative, nan=1.0, posinf=1.0, neginf=1.0)
