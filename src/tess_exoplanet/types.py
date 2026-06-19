from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


LABELS = ["planet", "eclipsing_binary", "background_blend", "noise"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(LABELS)}
INDEX_TO_LABEL = {idx: name for name, idx in LABEL_TO_INDEX.items()}


@dataclass
class LightCurve:
    target_id: str
    time: Any
    flux: Any
    flux_err: Any | None = None
    quality: Any | None = None
    centroid_col: Any | None = None
    centroid_row: Any | None = None
    source_path: str | None = None


@dataclass
class Candidate:
    target_id: str
    period: float
    epoch: float
    duration: float
    depth: float
    snr: float
    score: float
    phase_center: float
    n_transits: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VettingMetrics:
    odd_even_depth_delta: float
    secondary_depth: float
    transit_width_phase: float
    centroid_shift: float
    out_of_transit_scatter: float
    radius_ratio: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["flags"] = ";".join(self.flags)
        return data


@dataclass
class Dataset:
    X: Any
    y: Any
    records: list[dict[str, Any]]
    feature_names: list[str]

