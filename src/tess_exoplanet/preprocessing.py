from __future__ import annotations

import numpy as np
import pandas as pd

from .types import LightCurve


def preprocess_lightcurve(
    lightcurve: LightCurve,
    flatten_window: int = 101,
    sigma: float = 5.0,
) -> tuple[LightCurve, np.ndarray]:
    """Clean quality flags, flatten stellar trends, and sigma clip outliers."""
    time = np.asarray(lightcurve.time, dtype=float)
    flux = np.asarray(lightcurve.flux, dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)

    if lightcurve.quality is not None:
        quality = np.asarray(lightcurve.quality)
        mask &= quality == 0

    time = time[mask]
    flux = flux[mask]
    centroid_col = _masked(lightcurve.centroid_col, mask)
    centroid_row = _masked(lightcurve.centroid_row, mask)
    flux_err = _masked(lightcurve.flux_err, mask)

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    centroid_col = _ordered(centroid_col, order)
    centroid_row = _ordered(centroid_row, order)
    flux_err = _ordered(flux_err, order)

    median = np.nanmedian(flux)
    if not np.isfinite(median) or median == 0:
        median = 1.0
    normalized = flux / median

    trend = rolling_median(normalized, flatten_window)
    trend = np.where(np.isfinite(trend) & (np.abs(trend) > 1e-12), trend, 1.0)
    flattened = normalized / trend
    flattened = flattened / np.nanmedian(flattened)

    clip_mask = sigma_clip_mask(flattened, sigma=sigma)
    clean = LightCurve(
        target_id=lightcurve.target_id,
        time=time[clip_mask],
        flux=flattened[clip_mask],
        flux_err=_masked(flux_err, clip_mask),
        quality=None,
        centroid_col=_masked(centroid_col, clip_mask),
        centroid_row=_masked(centroid_row, clip_mask),
        source_path=lightcurve.source_path,
    )
    return clean, trend[clip_mask]


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def sigma_clip_mask(values: np.ndarray, sigma: float = 5.0, max_iter: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    for _ in range(max_iter):
        current = values[mask]
        if current.size < 8:
            break
        center = np.nanmedian(current)
        mad = np.nanmedian(np.abs(current - center))
        robust_sigma = 1.4826 * mad if mad > 0 else np.nanstd(current)
        if not np.isfinite(robust_sigma) or robust_sigma == 0:
            break
        new_mask = np.abs(values - center) <= sigma * robust_sigma
        new_mask &= np.isfinite(values)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return mask


def _masked(values, mask: np.ndarray):
    if values is None:
        return None
    return np.asarray(values)[mask]


def _ordered(values, order: np.ndarray):
    if values is None:
        return None
    return np.asarray(values)[order]

