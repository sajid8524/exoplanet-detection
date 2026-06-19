from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_lightcurve_csv
from .types import LABELS, LightCurve


def generate_dataset(
    out_dir: str | Path,
    n_curves: int = 160,
    seed: int = 42,
    days: float = 27.0,
    cadence_minutes: float = 30.0,
) -> Path:
    """Generate a balanced synthetic dataset for smoke tests and demos."""
    rng = np.random.default_rng(seed)
    out_path = Path(out_dir)
    curve_dir = out_path / "lightcurves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    labels = [LABELS[idx % len(LABELS)] for idx in range(n_curves)]
    rng.shuffle(labels)

    for idx, label in enumerate(labels):
        target_id = f"SYN{idx:05d}"
        lightcurve, truth = generate_lightcurve(
            target_id=target_id,
            label=label,
            rng=rng,
            days=days,
            cadence_minutes=cadence_minutes,
        )
        rel_path = Path("lightcurves") / f"{target_id}.csv"
        write_lightcurve_csv(curve_dir / f"{target_id}.csv", lightcurve)
        rows.append(
            {
                "target_id": target_id,
                "path": rel_path.as_posix(),
                "label": label,
                **truth,
            }
        )

    metadata_path = out_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return metadata_path


def generate_lightcurve(
    target_id: str,
    label: str,
    rng: np.random.Generator,
    days: float = 27.0,
    cadence_minutes: float = 30.0,
) -> tuple[LightCurve, dict[str, float]]:
    time = np.arange(0.0, days, cadence_minutes / (24.0 * 60.0), dtype=float)
    n = time.size

    stellar_amp = rng.uniform(0.0005, 0.004)
    stellar_period = rng.uniform(5.0, 18.0)
    flux = 1.0 + stellar_amp * np.sin(2.0 * np.pi * time / stellar_period)
    flux += 0.5 * stellar_amp * np.sin(2.0 * np.pi * time / rng.uniform(1.5, 4.0))

    red = np.cumsum(rng.normal(0.0, rng.uniform(0.000003, 0.00001), size=n))
    red -= np.nanmedian(red)
    white_noise = rng.normal(0.0, rng.uniform(0.0007, 0.0025), size=n)
    flux = flux + red + white_noise

    centroid_col = 100.0 + rng.normal(0.0, 0.0007, size=n)
    centroid_row = 100.0 + rng.normal(0.0, 0.0007, size=n)
    period = rng.uniform(1.0, 12.0)
    epoch = rng.uniform(0.1, min(period, days - 0.5))
    duration = rng.uniform(0.07, 0.22)
    depth = 0.0

    if label == "planet":
        depth = rng.uniform(0.0015, 0.015)
        flux += transit_profile(time, period, epoch, duration, depth, shape="u")
    elif label == "eclipsing_binary":
        depth = rng.uniform(0.025, 0.14)
        flux += transit_profile(time, period, epoch, duration, depth, shape="v")
        secondary_depth = depth * rng.uniform(0.25, 0.75)
        flux += transit_profile(time, period, epoch + 0.5 * period, duration * 0.9, secondary_depth, shape="v")
    elif label == "background_blend":
        depth = rng.uniform(0.002, 0.025)
        in_event = transit_mask(time, period, epoch, duration)
        flux += transit_profile(time, period, epoch, duration, depth, shape="u")
        shift = rng.uniform(0.004, 0.02)
        centroid_col[in_event] += shift
        centroid_row[in_event] += shift * rng.uniform(-0.8, 0.8)
    elif label == "noise":
        period = np.nan
        epoch = np.nan
        duration = np.nan
        for _ in range(rng.integers(1, 4)):
            center = rng.uniform(0.0, days)
            width = rng.uniform(0.02, 0.12)
            amp = rng.uniform(0.001, 0.008)
            flux -= amp * np.exp(-0.5 * ((time - center) / width) ** 2)
    else:
        raise ValueError(f"Unknown synthetic label: {label}")

    quality = np.zeros(n, dtype=int)
    bad_count = max(1, int(0.005 * n))
    bad_idx = rng.choice(n, size=bad_count, replace=False)
    quality[bad_idx] = 1
    flux[bad_idx] += rng.normal(0.0, 0.03, size=bad_count)

    flux_err = np.full(n, np.nanstd(white_noise), dtype=float)
    lightcurve = LightCurve(
        target_id=target_id,
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        centroid_col=centroid_col,
        centroid_row=centroid_row,
    )
    truth = {
        "period": float(period) if np.isfinite(period) else np.nan,
        "epoch": float(epoch) if np.isfinite(epoch) else np.nan,
        "duration": float(duration) if np.isfinite(duration) else np.nan,
        "true_depth": float(depth),
        "stellar_radius": float(rng.uniform(0.6, 1.4)),
        "stellar_mass": float(rng.uniform(0.6, 1.4)),
        "teff": float(rng.uniform(3800, 6800)),
        "tess_mag": float(rng.uniform(8.0, 14.0)),
    }
    return lightcurve, truth


def transit_profile(
    time: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    depth: float,
    shape: str = "u",
) -> np.ndarray:
    dt = folded_time_distance(time, period, epoch)
    half = duration / 2.0
    profile = np.zeros_like(time, dtype=float)
    inside = np.abs(dt) <= half
    if not np.any(inside):
        return profile

    x = np.abs(dt[inside]) / max(half, 1e-8)
    if shape == "v":
        dip = depth * (1.0 - x)
    else:
        ingress = 0.22
        flat = x <= (1.0 - ingress)
        dip = np.empty_like(x)
        dip[flat] = depth
        dip[~flat] = depth * (1.0 - (x[~flat] - (1.0 - ingress)) / ingress)
        dip = np.clip(dip, 0.0, depth)
    profile[inside] = -dip
    return profile


def transit_mask(time: np.ndarray, period: float, epoch: float, duration: float) -> np.ndarray:
    return np.abs(folded_time_distance(time, period, epoch)) <= duration / 2.0


def folded_time_distance(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return ((time - epoch + 0.5 * period) % period) - 0.5 * period

