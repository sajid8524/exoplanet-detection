from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_lightcurve_csv
from .types import LightCurve


def download_tess_lightcurves(
    targets_csv: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    author: str | None = "SPOC",
    cadence: str | None = None,
) -> Path:
    """Download TESS light curves for a CTL/TIC target manifest with Lightkurve."""
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise ImportError(
            "Install the full science stack first: pip install -r requirements-full.txt"
        ) from exc

    targets_path = Path(targets_csv)
    out_path = Path(out_dir)
    curve_dir = out_path / "lightcurves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(targets_path)
    if limit is not None:
        targets = targets.head(limit)

    rows: list[dict[str, object]] = []
    for _, row in targets.iterrows():
        tic_id = str(row.get("tic_id", row.get("target_id", ""))).replace("TIC", "")
        target_id = f"TIC{tic_id}"
        search = lk.search_lightcurve(
            f"TIC {tic_id}", mission="TESS", author=author, cadence=cadence
        )
        if len(search) == 0:
            continue
        collection = search.download_all()
        if collection is None or len(collection) == 0:
            continue

        stitched = collection.stitch().remove_nans()
        time = np.asarray(stitched.time.value, dtype=float)
        flux = np.asarray(stitched.flux.value, dtype=float)
        flux_err = None
        if getattr(stitched, "flux_err", None) is not None:
            flux_err = np.asarray(stitched.flux_err.value, dtype=float)

        lightcurve = LightCurve(target_id=target_id, time=time, flux=flux, flux_err=flux_err)
        rel_path = Path("lightcurves") / f"{target_id}.csv"
        write_lightcurve_csv(curve_dir / f"{target_id}.csv", lightcurve)

        out_row = row.to_dict()
        out_row["target_id"] = target_id
        out_row["path"] = rel_path.as_posix()
        rows.append(out_row)

    metadata_path = out_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return metadata_path

