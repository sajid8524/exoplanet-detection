from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALIASES = {
    "tic_id": ["ticid", "tic_id", "tic", "id", "targetid", "target_id"],
    "ra": ["ra", "raj2000", "ra_orig"],
    "dec": ["dec", "dej2000", "dec_orig"],
    "tess_mag": ["tmag", "tessmag", "tess_mag"],
    "teff": ["teff", "teff_k", "effective_temperature"],
    "stellar_radius": ["rad", "radius", "stellar_radius", "rstar", "r_star"],
    "stellar_mass": ["mass", "stellar_mass", "mstar", "m_star"],
    "logg": ["logg", "log_g"],
    "priority": ["priority", "pri", "ctl_priority"],
    "source_list": ["source", "source_list", "sourceid", "source_id"],
    "rank": ["rank", "ctl_rank"],
}


def prepare_ctl_targets(
    catalog_path: str | Path,
    out_csv: str | Path,
    max_rows: int | None = None,
    chunksize: int = 50_000,
    max_tess_mag: float | None = None,
    min_priority: float | None = None,
    min_teff: float | None = None,
    max_teff: float | None = None,
    max_radius: float | None = None,
) -> dict[str, Any]:
    """Prepare a compact target manifest from the STScI TESS CTL/TIC CSV.

    The manifest is used by the optional light-curve downloader. It does not
    create training labels; CTL/TIC are target catalogs, not classifications.
    """
    catalog = Path(catalog_path)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    total_read = 0
    total_written = 0
    wrote_header = False
    detected_columns: dict[str, str | None] | None = None

    read_kwargs: dict[str, Any] = {"chunksize": chunksize, "low_memory": False}
    if _is_headerless_exoctl(catalog):
        read_kwargs.update(
            {
                "header": None,
                "names": ["tic_id", "priority", "source_list", "rank"],
            }
        )

    for chunk in pd.read_csv(catalog, **read_kwargs):
        total_read += len(chunk)
        column_map = detect_columns(chunk.columns)
        if detected_columns is None:
            detected_columns = column_map
            if column_map["tic_id"] is None:
                raise ValueError(
                    "Could not find TIC ID column. Expected one of "
                    f"{ALIASES['tic_id']}; columns={list(chunk.columns)[:20]}..."
                )

        filtered = filter_ctl_chunk(
            chunk,
            column_map,
            max_tess_mag=max_tess_mag,
            min_priority=min_priority,
            min_teff=min_teff,
            max_teff=max_teff,
            max_radius=max_radius,
        )
        if filtered.empty:
            continue

        manifest = ctl_manifest_frame(filtered, column_map)
        if max_rows is not None:
            remaining = max_rows - total_written
            if remaining <= 0:
                break
            manifest = manifest.head(remaining)

        manifest.to_csv(out_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        total_written += len(manifest)
        if max_rows is not None and total_written >= max_rows:
            break

    return {
        "catalog": str(catalog),
        "out_csv": str(out_path),
        "rows_read": total_read,
        "rows_written": total_written,
        "detected_columns": detected_columns or {},
    }


def detect_columns(columns: Any) -> dict[str, str | None]:
    normalized = {_normalize(name): str(name) for name in columns}
    detected: dict[str, str | None] = {}
    for canonical, aliases in ALIASES.items():
        detected[canonical] = None
        for alias in aliases:
            normalized_alias = _normalize(alias)
            if normalized_alias in normalized:
                detected[canonical] = normalized[normalized_alias]
                break
    return detected


def filter_ctl_chunk(
    chunk: pd.DataFrame,
    column_map: dict[str, str | None],
    max_tess_mag: float | None = None,
    min_priority: float | None = None,
    min_teff: float | None = None,
    max_teff: float | None = None,
    max_radius: float | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=chunk.index)

    if max_tess_mag is not None and column_map.get("tess_mag"):
        mask &= pd.to_numeric(chunk[column_map["tess_mag"]], errors="coerce") <= max_tess_mag
    if min_priority is not None and column_map.get("priority"):
        mask &= pd.to_numeric(chunk[column_map["priority"]], errors="coerce") >= min_priority
    if min_teff is not None and column_map.get("teff"):
        mask &= pd.to_numeric(chunk[column_map["teff"]], errors="coerce") >= min_teff
    if max_teff is not None and column_map.get("teff"):
        mask &= pd.to_numeric(chunk[column_map["teff"]], errors="coerce") <= max_teff
    if max_radius is not None and column_map.get("stellar_radius"):
        mask &= pd.to_numeric(chunk[column_map["stellar_radius"]], errors="coerce") <= max_radius

    return chunk.loc[mask].copy()


def ctl_manifest_frame(chunk: pd.DataFrame, column_map: dict[str, str | None]) -> pd.DataFrame:
    tic_col = column_map["tic_id"]
    if tic_col is None:
        raise ValueError("TIC ID column is required.")

    tic_id = chunk[tic_col].map(_clean_tic_id)
    out = pd.DataFrame(
        {
            "target_id": "TIC" + tic_id,
            "tic_id": tic_id,
            "source_dataset": "stscI_tess_tic_ctl",
        }
    )
    for canonical in [
        "ra",
        "dec",
        "tess_mag",
        "teff",
        "stellar_radius",
        "stellar_mass",
        "logg",
        "priority",
        "source_list",
        "rank",
    ]:
        source = column_map.get(canonical)
        if source is None:
            out[canonical] = np.nan
        elif canonical == "source_list":
            out[canonical] = chunk[source].astype(str)
        else:
            out[canonical] = pd.to_numeric(chunk[source], errors="coerce")
    return out.dropna(subset=["tic_id"]).drop_duplicates(subset=["tic_id"])


def _clean_tic_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower().startswith("tic"):
        text = text[3:].strip()
    if "." in text:
        left, right = text.split(".", 1)
        if set(right) <= {"0"}:
            text = left
    return "".join(ch for ch in text if ch.isdigit())


def _normalize(name: Any) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _is_headerless_exoctl(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline().strip()
    parts = [part.strip().strip('"') for part in first_line.split(",")]
    if len(parts) != 4:
        return False
    if not parts[0].isdigit():
        return False
    try:
        float(parts[1])
        float(parts[3])
    except ValueError:
        return False
    return True
