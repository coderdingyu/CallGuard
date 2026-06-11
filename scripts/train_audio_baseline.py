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
from sklearn.metrics import accuracy_score, classification_report, f1_score
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

LABEL_TO_ID = {"normal": 0, "fraud": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


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
        by_label.setdefault(record["label"], []).append(record)

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

    for record in tqdm(records, desc="extract features"):
        audio_path = audio_root / record["audio_path"]
        try:
            x_rows.append(
                extract_audio_features(audio_path, sample_rate=sample_rate, duration=duration)
            )
            y_rows.append(LABEL_TO_ID[record["label"]])
        except Exception as exc:  # noqa: BLE001 - keep batch extraction robust.
            failed_ids.append(f"{record.get('id', record['audio_path'])}: {exc}")

    if not x_rows:
        raise RuntimeError("no usable audio features were extracted")
    return np.vstack(x_rows), np.asarray(y_rows), failed_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a traditional audio baseline.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "audio_baseline"))
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    processed_dir = data_root / "processed" / "teleantifraud_binary"
    audio_root = data_root / "interim" / "teleantifraud" / "audio"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": "logistic_regression_audio_features",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "sample_rate": args.sample_rate,
        "duration_seconds": args.duration,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "classification_report": report,
        "failures": {
            "train": train_failures[:20],
            "test": test_failures[:20],
            "train_failure_count": len(train_failures),
            "test_failure_count": len(test_failures),
        },
    }

    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
