from __future__ import annotations

from typing import Any

import numpy as np

from .types import LABEL_TO_INDEX


def apply_vetting_priors(
    probs: np.ndarray,
    records: list[dict[str, Any]],
    strength: float = 0.35,
) -> np.ndarray:
    """Blend ML probabilities with simple astrophysical vetting priors.

    This is intentionally conservative. It does not replace the classifier; it
    nudges probabilities when classic false-positive evidence is present.
    """
    adjusted = np.asarray(probs, dtype=float).copy()
    for idx, record in enumerate(records):
        depth = _float(record.get("depth"))
        snr = _float(record.get("snr"))
        scatter = max(_float(record.get("out_of_transit_scatter")), 1e-8)
        secondary = _float(record.get("secondary_depth"))
        odd_even = _float(record.get("odd_even_depth_delta"))
        centroid = _float(record.get("centroid_shift"))
        flags = str(record.get("flags", ""))

        boosts = np.zeros(adjusted.shape[1], dtype=float)
        if centroid > 0.002 or "centroid_shift" in flags:
            boosts[LABEL_TO_INDEX["background_blend"]] += strength
            boosts[LABEL_TO_INDEX["planet"]] -= 0.5 * strength

        secondary_limit = max(3.0 * scatter, 0.20 * max(depth, 1e-8))
        if secondary > secondary_limit or "secondary_eclipse" in flags:
            boosts[LABEL_TO_INDEX["eclipsing_binary"]] += strength
            boosts[LABEL_TO_INDEX["planet"]] -= 0.5 * strength

        odd_even_limit = max(3.0 * scatter, 0.25 * max(depth, 1e-8))
        if odd_even > odd_even_limit or "odd_even_mismatch" in flags:
            boosts[LABEL_TO_INDEX["eclipsing_binary"]] += 0.5 * strength

        if snr < 6.0 or "low_snr" in flags:
            boosts[LABEL_TO_INDEX["noise"]] += 0.5 * strength
            boosts[LABEL_TO_INDEX["planet"]] -= 0.25 * strength

        if not flags and snr >= 7.0 and 0.0005 <= depth <= 0.05:
            boosts[LABEL_TO_INDEX["planet"]] += 0.15 * strength

        adjusted[idx] = adjusted[idx] + boosts

    adjusted = np.clip(adjusted, 1e-6, None)
    adjusted /= np.sum(adjusted, axis=1, keepdims=True)
    return adjusted


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0

