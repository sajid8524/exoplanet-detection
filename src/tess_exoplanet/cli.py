from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import classification_report, write_evaluation
from .features import featurize_lightcurve, prepare_dataset
from .io import list_lightcurve_files, load_metadata, read_lightcurve_csv, resolve_curve_path
from .kaggle_kepler import convert_kaggle_kepler, describe_kaggle_csv
from .model_numpy import NumpyMLP, label_names, train_model, write_history
from .reporting import write_demo_plots, write_predictions, write_report
from .synthetic import generate_dataset
from .tess_ctl import prepare_ctl_targets
from .tess_download import download_tess_lightcurves
from .types import LABEL_TO_INDEX, LABELS
from .vetting import apply_vetting_priors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TESS exoplanet detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="Generate synthetic light curves")
    simulate.add_argument("--out", required=True)
    simulate.add_argument("--n", type=int, default=160)
    simulate.add_argument("--seed", type=int, default=42)

    kaggle = sub.add_parser("convert-kaggle-kepler", help="Convert Kaggle Kepler wide CSVs")
    kaggle.add_argument("--train", default=None, help="Path to exoTrain.csv")
    kaggle.add_argument("--test", default=None, help="Path to exoTest.csv")
    kaggle.add_argument("--out", required=True)
    kaggle.add_argument("--cadence-minutes", type=float, default=29.4)
    kaggle.add_argument("--max-train-rows", type=int, default=None)
    kaggle.add_argument("--max-test-rows", type=int, default=None)

    ctl = sub.add_parser("prepare-tess-ctl", help="Prepare target manifest from STScI TIC/CTL CSV")
    ctl.add_argument("--catalog", required=True, help="Path to Exoplanet CTL or TIC CSV/CSV.GZ")
    ctl.add_argument("--out", required=True, help="Output target manifest CSV")
    ctl.add_argument("--max-rows", type=int, default=None)
    ctl.add_argument("--chunksize", type=int, default=50_000)
    ctl.add_argument("--max-tess-mag", type=float, default=None)
    ctl.add_argument("--min-priority", type=float, default=None)
    ctl.add_argument("--min-teff", type=float, default=None)
    ctl.add_argument("--max-teff", type=float, default=None)
    ctl.add_argument("--max-radius", type=float, default=None)

    download = sub.add_parser("download-tess-lightcurves", help="Download TESS light curves using Lightkurve")
    download.add_argument("--targets", required=True, help="Target manifest from prepare-tess-ctl")
    download.add_argument("--out", required=True)
    download.add_argument("--limit", type=int, default=None)
    download.add_argument("--author", default="SPOC")
    download.add_argument("--cadence", default=None)

    train = sub.add_parser("train", help="Train the NumPy baseline model")
    train.add_argument("--metadata", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--hidden-units", type=int, default=96)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-rows", type=int, default=None)

    infer = sub.add_parser("infer", help="Predict labels for light curves")
    infer.add_argument("--model", required=True)
    infer.add_argument("--input", required=True, help="CSV file, folder of CSVs, or metadata CSV")
    infer.add_argument("--out", required=True)
    infer.add_argument("--metadata", action="store_true", help="Treat --input as metadata.csv")

    evaluate = sub.add_parser("evaluate", help="Evaluate predictions containing labels")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--out", required=True)

    report = sub.add_parser("report", help="Create markdown report from predictions")
    report.add_argument("--predictions", required=True)
    report.add_argument("--out", required=True)
    report.add_argument("--evaluation", default=None)

    demo = sub.add_parser("run-demo", help="Run synthetic generation, training, evaluation, and report")
    demo.add_argument("--workdir", required=True)
    demo.add_argument("--n", type=int, default=160)
    demo.add_argument("--epochs", type=int, default=80)
    demo.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "convert-kaggle-kepler":
        cmd_convert_kaggle_kepler(args)
    elif args.command == "prepare-tess-ctl":
        cmd_prepare_tess_ctl(args)
    elif args.command == "download-tess-lightcurves":
        cmd_download_tess_lightcurves(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "run-demo":
        cmd_run_demo(args)


def cmd_simulate(args: argparse.Namespace) -> None:
    metadata = generate_dataset(args.out, n_curves=args.n, seed=args.seed)
    print(f"Generated synthetic dataset: {metadata}")


def cmd_convert_kaggle_kepler(args: argparse.Namespace) -> None:
    if not args.train and not args.test:
        raise ValueError("Provide --train exoTrain.csv, --test exoTest.csv, or both.")

    if args.train:
        summary = describe_kaggle_csv(args.train, nrows=args.max_train_rows)
        print(
            "Train input: "
            f"{summary['rows']} rows, {summary['flux_columns']} flux columns, "
            f"labels={summary['label_counts']}"
        )
    if args.test:
        summary = describe_kaggle_csv(args.test, nrows=args.max_test_rows)
        print(
            "Test input: "
            f"{summary['rows']} rows, {summary['flux_columns']} flux columns, "
            f"labels={summary['label_counts']}"
        )

    outputs = convert_kaggle_kepler(
        train_csv=args.train,
        test_csv=args.test,
        out_dir=args.out,
        cadence_minutes=args.cadence_minutes,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )
    for split, metadata_path in outputs.items():
        print(f"Converted {split}: {metadata_path}")


def cmd_prepare_tess_ctl(args: argparse.Namespace) -> None:
    summary = prepare_ctl_targets(
        catalog_path=args.catalog,
        out_csv=args.out,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
        max_tess_mag=args.max_tess_mag,
        min_priority=args.min_priority,
        min_teff=args.min_teff,
        max_teff=args.max_teff,
        max_radius=args.max_radius,
    )
    print(f"Prepared TESS target manifest: {summary['out_csv']}")
    print(f"Rows read: {summary['rows_read']}")
    print(f"Rows written: {summary['rows_written']}")
    print(f"Detected columns: {summary['detected_columns']}")


def cmd_download_tess_lightcurves(args: argparse.Namespace) -> None:
    metadata = download_tess_lightcurves(
        targets_csv=args.targets,
        out_dir=args.out,
        limit=args.limit,
        author=args.author,
        cadence=args.cadence,
    )
    print(f"Downloaded light-curve metadata: {metadata}")


def cmd_train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    dataset = prepare_dataset(args.metadata, max_rows=args.max_rows)
    model, history = train_model(
        dataset.X,
        dataset.y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        seed=args.seed,
    )
    model_path = model.save(out_dir)
    write_history(history, out_dir / "training_history.csv")
    (out_dir / "feature_names.txt").write_text("\n".join(dataset.feature_names), encoding="utf-8")

    records = _records_with_true_labels(dataset.records, dataset.y)
    probs = apply_vetting_priors(model.predict_proba(dataset.X), records)
    pred_path = write_predictions(out_dir / "training_predictions.csv", records, probs, model.labels)
    y_pred = np.argmax(probs, axis=1)
    eval_report = classification_report(dataset.y, y_pred)
    write_evaluation(eval_report, out_dir / "training_evaluation.json")

    print(f"Saved model: {model_path}")
    print(f"Saved training predictions: {pred_path}")
    print(f"Final validation accuracy: {history[-1]['val_accuracy']:.3f}")


def cmd_infer(args: argparse.Namespace) -> None:
    model = NumpyMLP.load(_model_file(args.model))
    if args.metadata:
        X, records = featurize_from_metadata(args.input)
    else:
        X, records = featurize_from_files(args.input)
    probs = apply_vetting_priors(model.predict_proba(X), records)
    pred_path = write_predictions(args.out, records, probs, model.labels)
    print(f"Saved predictions: {pred_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.predictions)
    if "label" not in frame.columns:
        raise ValueError("Predictions must contain a label column for evaluation.")
    if "predicted_label" not in frame.columns:
        raise ValueError("Predictions must contain predicted_label.")
    y_true = np.asarray([LABEL_TO_INDEX[str(item)] for item in frame["label"]], dtype=int)
    y_pred = np.asarray([LABEL_TO_INDEX[str(item)] for item in frame["predicted_label"]], dtype=int)
    report = classification_report(y_true, y_pred)
    write_evaluation(report, args.out)
    print(f"Saved evaluation: {args.out}")
    print(f"Accuracy: {report['accuracy']:.3f}")


def cmd_report(args: argparse.Namespace) -> None:
    report_path = write_report(args.out, args.predictions, args.evaluation)
    print(f"Saved report: {report_path}")


def cmd_run_demo(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir)
    data_dir = workdir / "data"
    model_dir = workdir / "model"
    plots_dir = workdir / "plots"

    metadata = generate_dataset(data_dir, n_curves=args.n, seed=args.seed)
    dataset = prepare_dataset(metadata)
    model, history = train_model(dataset.X, dataset.y, epochs=args.epochs, seed=args.seed)
    model.save(model_dir)
    write_history(history, model_dir / "training_history.csv")

    records = _records_with_true_labels(dataset.records, dataset.y)
    probs = apply_vetting_priors(model.predict_proba(dataset.X), records)
    pred_path = write_predictions(workdir / "predictions.csv", records, probs, model.labels)
    y_pred = np.argmax(probs, axis=1)
    eval_report = classification_report(dataset.y, y_pred)
    eval_path = workdir / "evaluation.json"
    write_evaluation(eval_report, eval_path)
    write_demo_plots(metadata, plots_dir)
    report_path = write_report(workdir, pred_path, eval_path)

    print(f"Demo dataset: {metadata}")
    print(f"Model: {model_dir / 'model.npz'}")
    print(f"Predictions: {pred_path}")
    print(f"Evaluation: {eval_path}")
    print(f"Report: {report_path}")
    print(f"Training final val accuracy: {history[-1]['val_accuracy']:.3f}")
    print(f"Full-dataset demo accuracy: {eval_report['accuracy']:.3f}")


def featurize_from_metadata(metadata_path: str | Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    metadata = load_metadata(metadata_path)
    vectors = []
    records: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        target_id = str(row.get("target_id"))
        curve_path = resolve_curve_path(metadata_path, str(row["path"]))
        lightcurve = read_lightcurve_csv(curve_path, target_id=target_id)
        vector, record = featurize_lightcurve(lightcurve, row=row, use_metadata_candidate=True)
        if "label" in row and str(row["label"]) in LABELS:
            record["label"] = str(row["label"])
        vectors.append(vector)
        records.append(record)
    return np.vstack(vectors), records


def featurize_from_files(input_path: str | Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    vectors = []
    records: list[dict[str, object]] = []
    for path in list_lightcurve_files(input_path):
        lightcurve = read_lightcurve_csv(path)
        vector, record = featurize_lightcurve(lightcurve, use_metadata_candidate=False)
        vectors.append(vector)
        records.append(record)
    if not vectors:
        raise ValueError(f"No CSV light curves found in {input_path}")
    return np.vstack(vectors), records


def _records_with_true_labels(records: list[dict[str, object]], y: np.ndarray) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    names = label_names(y)
    for record, label in zip(records, names):
        copy = dict(record)
        copy["label"] = label
        out.append(copy)
    return out


def _model_file(path: str | Path) -> Path:
    model_path = Path(path)
    if model_path.is_dir():
        model_path = model_path / "model.npz"
    return model_path


if __name__ == "__main__":
    main()
