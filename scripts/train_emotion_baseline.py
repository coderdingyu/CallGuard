from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

from callguard_ml.audio_features import (  # noqa: E402
    DEFAULT_DURATION_SECONDS,
    DEFAULT_SAMPLE_RATE,
    extract_audio_features,
)
from callguard_ml.emotion_model import (  # noqa: E402
    EMOTION_LABELS,
    PRESSURE_LABELS,
    PRESSURE_LEVEL_SCORES,
    PRESSURE_WEIGHTS,
    pressure_level_from_score,
)


LABEL_TO_ID = {label: index for index, label in enumerate(EMOTION_LABELS)}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
PRESSURE_LABEL_TO_ID = {label: index for index, label in enumerate(PRESSURE_LABELS)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stratified_sample(
    records: list[dict[str, Any]],
    max_items: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_items is None or len(records) <= max_items:
        return records

    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(record["emotion"], []).append(record)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    per_label = max_items // max(1, len(by_label))
    remainder = max_items % max(1, len(by_label))
    for index, label in enumerate(sorted(by_label)):
        bucket = by_label[label][:]
        rng.shuffle(bucket)
        take = per_label + (1 if index < remainder else 0)
        sampled.extend(bucket[:take])

    rng.shuffle(sampled)
    return sampled


def build_matrix(
    records: list[dict[str, Any]],
    audio_root: Path,
    sample_rate: int,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    failed_ids: list[str] = []

    for record in tqdm(records, desc="extract emotion features"):
        audio_path = audio_root / record["audio_path"]
        try:
            emotion = str(record["emotion"])
            x_rows.append(
                extract_audio_features(audio_path, sample_rate=sample_rate, duration=duration)
            )
            y_rows.append(LABEL_TO_ID[emotion])
        except Exception as exc:  # noqa: BLE001 - keep batch extraction robust.
            failed_ids.append(f"{record.get('id', record['audio_path'])}: {exc}")

    if not x_rows:
        raise RuntimeError("no usable emotion features were extracted")
    return np.vstack(x_rows), np.asarray(y_rows), failed_ids


def build_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def pressure_levels_from_emotion_ids(ids: np.ndarray) -> list[str]:
    return [pressure_level_from_score(PRESSURE_WEIGHTS[ID_TO_LABEL[int(value)]]) for value in ids]


def pressure_ids_from_emotion_ids(ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            PRESSURE_LABEL_TO_ID[
                pressure_level_from_score(PRESSURE_WEIGHTS[ID_TO_LABEL[int(value)]])
            ]
            for value in ids
        ],
        dtype=np.int64,
    )


def pressure_scores_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    weights = np.asarray(
        [PRESSURE_LEVEL_SCORES[label] for label in PRESSURE_LABELS],
        dtype=np.float32,
    )
    return probabilities @ weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CSEMOTIONS emotion baseline.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "emotion_baseline"))
    parser.add_argument("--export-dir", default=str(PROJECT_ROOT / "ml" / "models" / "emotion_baseline"))
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    processed_dir = data_root / "processed" / "csemotions_emotion"
    audio_root = data_root / "interim" / "csemotions" / "audio"
    output_dir = Path(args.output_dir).resolve()
    export_dir = Path(args.export_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    train_records = stratified_sample(
        load_jsonl(processed_dir / "train.jsonl"),
        max_items=args.max_train,
        seed=args.seed,
    )
    test_records = stratified_sample(
        load_jsonl(processed_dir / "test.jsonl"),
        max_items=args.max_test,
        seed=args.seed,
    )

    x_train, y_train, train_failures = build_matrix(
        train_records,
        audio_root=audio_root,
        sample_rate=args.sample_rate,
        duration=args.duration,
    )
    x_test, y_test, test_failures = build_matrix(
        test_records,
        audio_root=audio_root,
        sample_rate=args.sample_rate,
        duration=args.duration,
    )

    model = build_classifier(args.seed)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    pressure_model = build_classifier(args.seed)
    pressure_y_train = pressure_ids_from_emotion_ids(y_train)
    pressure_y_test = pressure_ids_from_emotion_ids(y_test)
    pressure_model.fit(x_train, pressure_y_train)
    pressure_predictions = pressure_model.predict(x_test)
    pressure_probabilities = pressure_model.predict_proba(x_test)
    pressure_scores = pressure_scores_from_probabilities(pressure_probabilities)
    true_pressure_scores = np.asarray(
        [PRESSURE_WEIGHTS[ID_TO_LABEL[int(value)]] for value in y_test],
        dtype=np.float32,
    )
    true_pressure_levels = pressure_levels_from_emotion_ids(y_test)
    predicted_pressure_levels = [
        PRESSURE_LABELS[int(value)] for value in pressure_predictions
    ]

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(EMOTION_LABELS))),
        target_names=EMOTION_LABELS,
        output_dict=True,
        zero_division=0,
    )
    pressure_report = classification_report(
        true_pressure_levels,
        predicted_pressure_levels,
        labels=PRESSURE_LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": "logistic_regression_audio_features",
        "dataset": "csemotions_emotion",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "sample_rate": args.sample_rate,
        "duration_seconds": args.duration,
        "emotion_labels": EMOTION_LABELS,
        "pressure_mapping": PRESSURE_WEIGHTS,
        "pressure_labels": PRESSURE_LABELS,
        "pressure_level_scores": PRESSURE_LEVEL_SCORES,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "pressure_mae": float(mean_absolute_error(true_pressure_scores, pressure_scores)),
        "pressure_accuracy": float(
            accuracy_score(pressure_y_test, pressure_predictions)
        ),
        "pressure_macro_f1": float(
            f1_score(
                pressure_y_test,
                pressure_predictions,
                labels=list(range(len(PRESSURE_LABELS))),
                average="macro",
            )
        ),
        "classification_report": report,
        "pressure_classification_report": pressure_report,
        "failures": {
            "train": train_failures[:20],
            "test": test_failures[:20],
            "train_failure_count": len(train_failures),
            "test_failure_count": len(test_failures),
        },
    }
    artifact = {
        "model": model,
        "pressure_model": pressure_model,
        "emotion_labels": EMOTION_LABELS,
        "pressure_weights": PRESSURE_WEIGHTS,
        "pressure_labels": PRESSURE_LABELS,
        "pressure_level_scores": PRESSURE_LEVEL_SCORES,
        "sample_rate": args.sample_rate,
        "duration_seconds": args.duration,
        "name": "csemotions_logistic_regression_audio_features",
        "version": "1.0",
        "description": "CSEMOTIONS speech emotion baseline with pressure-score mapping.",
    }

    joblib.dump(artifact, output_dir / "model.joblib")
    joblib.dump(artifact, export_dir / "model.joblib")
    for target_dir in (output_dir, export_dir):
        (target_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({k: v for k, v in metrics.items() if "report" not in k}, indent=2))
    print(f"wrote: {output_dir}")
    print(f"exported: {export_dir}")


if __name__ == "__main__":
    main()
