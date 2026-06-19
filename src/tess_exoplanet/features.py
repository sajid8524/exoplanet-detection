from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bls import search_bls
from .io import load_metadata, read_lightcurve_csv, resolve_curve_path
from .preprocessing import preprocess_lightcurve
from .types import (
    Dataset,
    LABEL_TO_INDEX,
    Candidate,
    LightCurve,
    VettingMetrics,
)


SCALAR_FEATURES = [
    "period",
    "duration",
    "depth",
    "snr",
    "duration_ratio",
    "n_transits",
    "odd_even_depth_delta",
    "secondary_depth",
    "centroid_shift",
    "out_of_transit_scatter",
    "radius_ratio",
]


def prepare_dataset(
    metadata_path: str | Path,
    max_rows: int | None = None,
    use_metadata_candidates: bool = True,
    global_bins: int = 301,
    local_bins: int = 101,
) -> Dataset:
    metadata = load_metadata(metadata_path)
    if max_rows is not None:
        metadata = metadata.head(max_rows)

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    feature_names = _feature_names(global_bins, local_bins)

    for _, row in metadata.iterrows():
        target_id = str(row.get("target_id"))
        curve_path = resolve_curve_path(metadata_path, str(row["path"]))
        lightcurve = read_lightcurve_csv(curve_path, target_id=target_id)
        vector, record = featurize_lightcurve(
            lightcurve,
            row=row,
            use_metadata_candidate=use_metadata_candidates,
            global_bins=global_bins,
            local_bins=local_bins,
        )

        label = str(row.get("label", "")).strip()
        if label not in LABEL_TO_INDEX:
            continue
        vectors.append(vector)
        labels.append(LABEL_TO_INDEX[label])
        record["label"] = label
        records.append(record)

    if not vectors:
        raise ValueError("No labeled rows were loaded. Check metadata labels and paths.")

    return Dataset(
        X=np.vstack(vectors).astype(float),
        y=np.asarray(labels, dtype=int),
        records=records,
        feature_names=feature_names,
    )


def featurize_lightcurve(
    lightcurve: LightCurve,
    row: pd.Series | dict[str, Any] | None = None,
    use_metadata_candidate: bool = True,
    global_bins: int = 301,
    local_bins: int = 101,
    local_width_phase: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any]]:
    clean, _trend = preprocess_lightcurve(lightcurve)
    candidate = _candidate_from_metadata(clean, row) if use_metadata_candidate else None
    if candidate is None:
        candidate = search_bls(clean)

    global_view, local_view = make_global_local_views(
        clean, candidate, global_bins=global_bins, local_bins=local_bins, local_width_phase=local_width_phase
    )
    vetting = compute_vetting_metrics(clean, candidate)
    scalars = candidate_scalars(candidate, vetting)
    vector = np.concatenate([global_view, local_view, scalars])

    record = {
        "target_id": clean.target_id,
        **candidate.to_dict(),
        **vetting.to_dict(),
    }
    return vector, record


def phase_fold(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    return phase.astype(float)


def make_global_local_views(
    lightcurve: LightCurve,
    candidate: Candidate,
    global_bins: int = 301,
    local_bins: int = 101,
    local_width_phase: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(lightcurve.time, dtype=float)
    flux = np.asarray(lightcurve.flux, dtype=float)
    phase = phase_fold(time, candidate.period, candidate.epoch)
    global_flux = bin_by_phase(phase, flux, n_bins=global_bins, low=-0.5, high=0.5)

    duration_phase = candidate.duration / max(candidate.period, 1e-6)
    width = max(local_width_phase, 3.0 * duration_phase)
    width = min(width, 0.25)
    local_flux = bin_by_phase(phase, flux, n_bins=local_bins, low=-width, high=width)

    return _view_signal(global_flux), _view_signal(local_flux)


def bin_by_phase(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    low: float,
    high: float,
) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)
    mask = np.isfinite(phase) & np.isfinite(flux) & (phase >= low) & (phase <= high)
    if np.sum(mask) < 3:
        return np.ones(n_bins, dtype=float)

    phase = phase[mask]
    flux = flux[mask]
    edges = np.linspace(low, high, n_bins + 1)
    index = np.digitize(phase, edges) - 1
    index = np.clip(index, 0, n_bins - 1)

    binned = np.full(n_bins, np.nan)
    for idx in range(n_bins):
        values = flux[index == idx]
        if values.size:
            binned[idx] = np.nanmedian(values)

    valid = np.isfinite(binned)
    if np.sum(valid) == 0:
        return np.ones(n_bins, dtype=float)
    if np.sum(valid) < n_bins:
        x = np.arange(n_bins)
        binned[~valid] = np.interp(x[~valid], x[valid], binned[valid])
    return binned


def compute_vetting_metrics(lightcurve: LightCurve, candidate: Candidate) -> VettingMetrics:
    time = np.asarray(lightcurve.time, dtype=float)
    flux = np.asarray(lightcurve.flux, dtype=float)
    phase = phase_fold(time, candidate.period, candidate.epoch)
    width_phase = min(max(candidate.duration / max(candidate.period, 1e-6), 1e-4), 0.25)
    in_transit = np.abs(phase) <= width_phase / 2.0
    out_transit = ~in_transit
    out_scatter = float(np.nanstd(flux[out_transit])) if np.any(out_transit) else float(np.nanstd(flux))

    transit_numbers = np.floor((time - candidate.epoch) / candidate.period).astype(int)
    odd_depth = _depth_for_mask(flux, in_transit & (transit_numbers % 2 != 0), out_transit)
    even_depth = _depth_for_mask(flux, in_transit & (transit_numbers % 2 == 0), out_transit)
    odd_even_delta = abs(odd_depth - even_depth)

    secondary_distance = np.abs((phase - 0.5 + 0.5) % 1.0 - 0.5)
    secondary_mask = secondary_distance <= width_phase / 2.0
    secondary_depth = _depth_for_mask(flux, secondary_mask, ~secondary_mask)

    centroid_shift = compute_centroid_shift(lightcurve, in_transit, out_transit)
    radius_ratio = float(np.sqrt(max(candidate.depth, 0.0)))
    flags = vetting_flags(candidate, odd_even_delta, secondary_depth, centroid_shift, out_scatter)

    return VettingMetrics(
        odd_even_depth_delta=float(odd_even_delta),
        secondary_depth=float(max(0.0, secondary_depth)),
        transit_width_phase=float(width_phase),
        centroid_shift=float(centroid_shift),
        out_of_transit_scatter=float(out_scatter),
        radius_ratio=radius_ratio,
        flags=flags,
    )


def candidate_scalars(candidate: Candidate, vetting: VettingMetrics) -> np.ndarray:
    duration_ratio = candidate.duration / max(candidate.period, 1e-6)
    return np.asarray(
        [
            candidate.period,
            candidate.duration,
            candidate.depth,
            candidate.snr,
            duration_ratio,
            candidate.n_transits,
            vetting.odd_even_depth_delta,
            vetting.secondary_depth,
            vetting.centroid_shift,
            vetting.out_of_transit_scatter,
            vetting.radius_ratio,
        ],
        dtype=float,
    )


def compute_centroid_shift(
    lightcurve: LightCurve, in_transit: np.ndarray, out_transit: np.ndarray
) -> float:
    if lightcurve.centroid_col is None or lightcurve.centroid_row is None:
        return 0.0
    col = np.asarray(lightcurve.centroid_col, dtype=float)
    row = np.asarray(lightcurve.centroid_row, dtype=float)
    valid_in = in_transit & np.isfinite(col) & np.isfinite(row)
    valid_out = out_transit & np.isfinite(col) & np.isfinite(row)
    if np.sum(valid_in) < 2 or np.sum(valid_out) < 2:
        return 0.0
    d_col = float(np.nanmean(col[valid_in]) - np.nanmean(col[valid_out]))
    d_row = float(np.nanmean(row[valid_in]) - np.nanmean(row[valid_out]))
    return float(np.sqrt(d_col * d_col + d_row * d_row))


def vetting_flags(
    candidate: Candidate,
    odd_even_delta: float,
    secondary_depth: float,
    centroid_shift: float,
    out_scatter: float,
) -> list[str]:
    flags: list[str] = []
    if candidate.snr < 7.0:
        flags.append("low_snr")
    if candidate.depth > 0.08:
        flags.append("very_deep_event")
    if odd_even_delta > max(3.0 * out_scatter, 0.25 * max(candidate.depth, 1e-6)):
        flags.append("odd_even_mismatch")
    if secondary_depth > max(3.0 * out_scatter, 0.20 * max(candidate.depth, 1e-6)):
        flags.append("secondary_eclipse")
    if centroid_shift > 0.002:
        flags.append("centroid_shift")
    return flags


def _depth_for_mask(flux: np.ndarray, in_mask: np.ndarray, out_mask: np.ndarray) -> float:
    if np.sum(in_mask) < 2 or np.sum(out_mask) < 2:
        return 0.0
    return float(np.nanmedian(flux[out_mask]) - np.nanmedian(flux[in_mask]))


def _view_signal(flux_view: np.ndarray) -> np.ndarray:
    signal = 1.0 - np.asarray(flux_view, dtype=float)
    signal -= np.nanmedian(signal)
    scale = np.nanstd(signal)
    if np.isfinite(scale) and scale > 1e-8:
        signal = signal / scale
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


def _candidate_from_metadata(
    lightcurve: LightCurve, row: pd.Series | dict[str, Any] | None
) -> Candidate | None:
    if row is None:
        return None

    def get_float(name: str) -> float | None:
        try:
            value = row[name]  # type: ignore[index]
        except Exception:
            return None
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    period = get_float("period")
    epoch = get_float("epoch")
    duration = get_float("duration")
    if period is None or epoch is None or duration is None or period <= 0 or duration <= 0:
        return None

    time = np.asarray(lightcurve.time, dtype=float)
    flux = np.asarray(lightcurve.flux, dtype=float)
    phase = phase_fold(time, period, epoch)
    width_phase = min(max(duration / period, 1e-4), 0.25)
    in_transit = np.abs(phase) <= width_phase / 2.0
    out_transit = ~in_transit
    depth = _depth_for_mask(flux, in_transit, out_transit)
    scatter = float(np.nanstd(flux[out_transit])) if np.any(out_transit) else float(np.nanstd(flux))
    snr = depth / (scatter + 1e-10) * np.sqrt(max(int(np.sum(in_transit)), 1))
    n_transits = int(max(1, np.floor((np.nanmax(time) - np.nanmin(time)) / period)))
    return Candidate(
        target_id=lightcurve.target_id,
        period=float(period),
        epoch=float(epoch),
        duration=float(duration),
        depth=float(max(depth, 0.0)),
        snr=float(max(snr, 0.0)),
        score=float(max(depth, 0.0) * max(snr, 0.0)),
        phase_center=float(((epoch - np.nanmin(time)) / period) % 1.0),
        n_transits=n_transits,
    )


def _feature_names(global_bins: int, local_bins: int) -> list[str]:
    names = [f"global_{idx:03d}" for idx in range(global_bins)]
    names.extend(f"local_{idx:03d}" for idx in range(local_bins))
    names.extend(SCALAR_FEATURES)
    return names

