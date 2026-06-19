from pathlib import Path

import pandas as pd

from tess_exoplanet.tess_ctl import prepare_ctl_targets


def test_prepare_ctl_targets(tmp_path: Path) -> None:
    catalog = tmp_path / "exo_CTL_sample.csv"
    pd.DataFrame(
        {
            "TICID": [123, 456, 789],
            "RA": [10.0, 20.0, 30.0],
            "DEC": [-5.0, 1.0, 2.0],
            "Tmag": [9.5, 13.2, 8.1],
            "Teff": [5400, 6100, 4500],
            "rad": [0.9, 1.5, 0.7],
            "priority": [0.7, 0.1, 0.9],
        }
    ).to_csv(catalog, index=False)

    out = tmp_path / "targets.csv"
    summary = prepare_ctl_targets(
        catalog,
        out,
        max_tess_mag=10.0,
        min_priority=0.5,
        max_radius=1.0,
    )
    targets = pd.read_csv(out)

    assert summary["rows_read"] == 3
    assert summary["rows_written"] == 2
    assert targets["target_id"].tolist() == ["TIC123", "TIC789"]
    assert "stellar_radius" in targets.columns

