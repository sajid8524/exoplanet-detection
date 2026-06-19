from __future__ import annotations

import numpy as np

from .types import Candidate, LightCurve


def search_bls(
    lightcurve: LightCurve,
    min_period: float = 0.5,
    max_period: float = 15.0,
    n_periods: int = 80,
    duration_grid_days: list[float] | None = None,
    n_phase_bins: int = 240,
) -> Candidate:
    """A dependency-light BLS-style search over binned folded light curves."""
    time = np.asarray(lightcurve.time, dtype=float)
    flux = np.asarray(lightcurve.flux, dtype=float)
    if time.size < 20:
        raise ValueError(f"{lightcurve.target_id} has too few valid points.")

    duration_grid = duration_grid_days or [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    periods = np.linspace(min_period, max_period, int(n_periods))
    best: Candidate | None = None
    global_std = float(np.nanstd(flux))
    if not np.isfinite(global_std) or global_std == 0:
        global_std = 1e-6

    start_time = float(np.nanmin(time))
    span = float(np.nanmax(time) - start_time)

    for period in periods:
        phase = ((time - start_time) / period) % 1.0
        bin_index = np.floor(phase * n_phase_bins).astype(int)
        bin_index = np.clip(bin_index, 0, n_phase_bins - 1)

        counts = np.bincount(bin_index, minlength=n_phase_bins).astype(float)
        sums = np.bincount(bin_index, weights=flux, minlength=n_phase_bins).astype(float)
        total_count = float(np.sum(counts))
        total_sum = float(np.sum(sums))
        if total_count <= 10:
            continue

        doubled_counts = np.concatenate([counts, counts])
        doubled_sums = np.concatenate([sums, sums])
        c_counts = np.concatenate([[0.0], np.cumsum(doubled_counts)])
        c_sums = np.concatenate([[0.0], np.cumsum(doubled_sums)])

        for duration in duration_grid:
            width_phase = min(max(duration / period, 1.0 / n_phase_bins), 0.2)
            width_bins = int(max(1, round(width_phase * n_phase_bins)))

            starts = np.arange(n_phase_bins)
            ends = starts + width_bins
            in_count = c_counts[ends] - c_counts[starts]
            in_sum = c_sums[ends] - c_sums[starts]
            out_count = total_count - in_count
            valid = (in_count >= 3) & (out_count >= 3)
            if not np.any(valid):
                continue

            in_mean = np.full(n_phase_bins, np.nan)
            out_mean = np.full(n_phase_bins, np.nan)
            in_mean[valid] = in_sum[valid] / in_count[valid]
            out_mean[valid] = (total_sum - in_sum[valid]) / out_count[valid]
            depth = np.full(n_phase_bins, -np.inf)
            depth[valid] = out_mean[valid] - in_mean[valid]
            snr = depth / (global_std + 1e-10) * np.sqrt(np.maximum(in_count, 1.0))
            snr[~valid] = -np.inf
            snr[depth <= 0] = -np.inf

            idx = int(np.argmax(snr))
            score = float(snr[idx])
            if not np.isfinite(score):
                continue

            center_phase = ((idx + 0.5 * width_bins) / n_phase_bins) % 1.0
            epoch = start_time + center_phase * period
            n_transits = int(max(1, np.floor(span / period)))
            candidate = Candidate(
                target_id=lightcurve.target_id,
                period=float(period),
                epoch=float(epoch),
                duration=float(duration),
                depth=float(depth[idx]),
                snr=score,
                score=score * float(depth[idx]),
                phase_center=float(center_phase),
                n_transits=n_transits,
            )
            if best is None or candidate.snr > best.snr:
                best = candidate

    if best is None:
        return Candidate(
            target_id=lightcurve.target_id,
            period=float(min_period),
            epoch=start_time,
            duration=float(duration_grid[0]),
            depth=0.0,
            snr=0.0,
            score=0.0,
            phase_center=0.0,
            n_transits=0,
        )
    return best
