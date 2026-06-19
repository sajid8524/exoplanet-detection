from pathlib import Path

from tess_exoplanet.synthetic import generate_dataset
from tess_exoplanet.features import prepare_dataset
from tess_exoplanet.model_numpy import train_model


def test_smoke_pipeline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    metadata = generate_dataset(data_dir, n_curves=24, seed=123)
    dataset = prepare_dataset(metadata, max_rows=24)
    model, history = train_model(dataset.X, dataset.y, epochs=5, seed=123)
    probs = model.predict_proba(dataset.X[:4])

    assert dataset.X.shape[0] == 24
    assert probs.shape == (4, 4)
    assert history

