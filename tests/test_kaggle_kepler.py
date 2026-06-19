from pathlib import Path

import pandas as pd

from tess_exoplanet.kaggle_kepler import convert_kaggle_split, describe_kaggle_csv


def test_convert_kaggle_split(tmp_path: Path) -> None:
    source = tmp_path / "exoTrain.csv"
    pd.DataFrame(
        {
            "LABEL": [2, 1],
            "FLUX.1": [1.0, 0.99],
            "FLUX.2": [0.98, 1.01],
            "FLUX.3": [1.0, 1.0],
        }
    ).to_csv(source, index=False)

    summary = describe_kaggle_csv(source)
    metadata = convert_kaggle_split(source, tmp_path / "converted", "train")
    converted = pd.read_csv(metadata)

    assert summary["rows"] == 2
    assert summary["flux_columns"] == 3
    assert converted["label"].tolist() == ["planet", "noise"]
    assert (tmp_path / "converted" / converted.loc[0, "path"]).exists()

