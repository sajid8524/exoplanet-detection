from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .types import LightCurve


FLUX_COLUMNS = ["flux", "pdcsap_flux", "sap_flux", "PDCSAP_FLUX", "SAP_FLUX"]


def read_lightcurve_csv(path: str | Path, target_id: str | None = None) -> LightCurve:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)

    time_col = _first_existing(frame, ["time", "TIME", "btjd", "bjd"])
    flux_col = _first_existing(frame, FLUX_COLUMNS)
    if time_col is None or flux_col is None:
        raise ValueError(
            f"{csv_path} must contain a time column and one of {FLUX_COLUMNS}."
        )

    flux_err_col = _first_existing(frame, ["flux_err", "flux_error", "PDCSAP_FLUX_ERR"])
    quality_col = _first_existing(frame, ["quality", "QUALITY"])
    centroid_col = _first_existing(frame, ["centroid_col", "mom_centr1", "MOM_CENTR1"])
    centroid_row = _first_existing(frame, ["centroid_row", "mom_centr2", "MOM_CENTR2"])

    inferred_id = target_id or csv_path.stem
    return LightCurve(
        target_id=str(inferred_id),
        time=frame[time_col].to_numpy(dtype=float),
        flux=frame[flux_col].to_numpy(dtype=float),
        flux_err=_optional_array(frame, flux_err_col),
        quality=_optional_array(frame, quality_col),
        centroid_col=_optional_array(frame, centroid_col),
        centroid_row=_optional_array(frame, centroid_row),
        source_path=str(csv_path),
    )


def write_lightcurve_csv(path: str | Path, lightcurve: LightCurve) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "time": np.asarray(lightcurve.time, dtype=float),
        "flux": np.asarray(lightcurve.flux, dtype=float),
    }
    if lightcurve.flux_err is not None:
        data["flux_err"] = np.asarray(lightcurve.flux_err, dtype=float)
    if lightcurve.quality is not None:
        data["quality"] = np.asarray(lightcurve.quality)
    if lightcurve.centroid_col is not None:
        data["centroid_col"] = np.asarray(lightcurve.centroid_col, dtype=float)
    if lightcurve.centroid_row is not None:
        data["centroid_row"] = np.asarray(lightcurve.centroid_row, dtype=float)
    pd.DataFrame(data).to_csv(csv_path, index=False)


def load_metadata(path: str | Path) -> pd.DataFrame:
    metadata_path = Path(path)
    frame = pd.read_csv(metadata_path)
    if "target_id" not in frame.columns:
        frame["target_id"] = [f"target_{idx:05d}" for idx in range(len(frame))]
    if "path" not in frame.columns:
        raise ValueError("metadata.csv must contain a path column.")
    return frame


def resolve_curve_path(metadata_path: str | Path, row_path: str) -> Path:
    path = Path(row_path)
    if path.is_absolute():
        return path
    return Path(metadata_path).parent / path


def list_lightcurve_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.csv"))


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _optional_array(frame: pd.DataFrame, column: str | None):
    if column is None:
        return None
    return frame[column].to_numpy()

