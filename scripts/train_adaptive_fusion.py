from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

from callguard_ml.adaptive_fusion import adaptive_fusion  # noqa: E402
from callguard_ml.audio_features import extract_audio_features  # noqa: E402
from callguard_ml.risk_rules import score_text_rules  # noqa: E402
from callguard_ml.text_normalization import normalize_chinese_text  # noqa: E402


LABEL_TO_ID = {"normal": 0, "fraud": 1}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def usable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if str(record.get("text", "")).strip() and record.get("label") in LABEL_TO_ID
    ]


def audio_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1500,
                    random_state=seed,
                ),
            ),
        ]
    )


def text_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 4),
                    min_df=1,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
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


def build_audio_matrix(
    records: list[dict[str, Any]],
    audio_root: Path,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)["features"]

    rows = []
    for record in tqdm(records, desc=f"audio features {cache_path.stem}"):
        rows.append(extract_audio_features(audio_root / record["audio_path"]))
    matrix = np.vstack(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=matrix)
    return matrix


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "fraud_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "fraud_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=["normal", "fraud"],
            output_dict=True,
            zero_division=0,
        ),
    }


def choose_parameters(
    y_true: np.ndarray,
    audio_scores: np.ndarray,
    text_scores: np.ndarray,
    rule_scores: np.ndarray,
    audio_reliability: float,
    text_reliability: float,
) -> dict[str, float]:
    best: tuple[float, float, float, float, float] | None = None
    for text_confidence_gate in (0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.1):
        for reliability_power in (1.0, 2.0, 3.0, 4.0, 5.0):
            for floor in np.arange(0.2, 0.81, 0.1):
                for rule_boost in np.arange(0.0, 0.51, 0.05):
                    config = {
                        "audio_reliability": audio_reliability,
                        "text_reliability": text_reliability,
                        "reliability_power": reliability_power,
                        "text_confidence_gate": text_confidence_gate,
                        "confidence_floor": float(floor),
                        "rule_boost": float(rule_boost),
                    }
                    scores = np.asarray(
                        [
                            adaptive_fusion(audio, text, rule, config).risk_score
                            for audio, text, rule in zip(
                                audio_scores,
                                text_scores,
                                rule_scores,
                                strict=False,
                            )
                        ]
                    )
                    for threshold in np.arange(0.35, 0.66, 0.025):
                        predictions = (scores >= threshold).astype(int)
                        macro_f1 = float(f1_score(y_true, predictions, average="macro"))
                        fraud_recall = float(recall_score(y_true, predictions, zero_division=0))
                        accuracy = float(accuracy_score(y_true, predictions))
                        candidate = (
                            macro_f1,
                            fraud_recall,
                            accuracy,
                            -abs(float(threshold) - 0.5),
                            -float(rule_boost),
                        )
                        if best is None or candidate > best:
                            best = candidate
                            best_config = {
                                **config,
                                "decision_threshold": round(float(threshold), 4),
                            }
    return best_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate CallGuard CAEF fusion.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed" / "teleantifraud_asr"))
    parser.add_argument(
        "--binary-data-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "teleantifraud_binary"),
    )
    parser.add_argument("--audio-root", default=str(PROJECT_ROOT / "data" / "interim" / "teleantifraud" / "audio"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "adaptive_fusion"))
    parser.add_argument("--export-dir", default=str(PROJECT_ROOT / "ml" / "models" / "fusion"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    binary_data_dir = Path(args.binary_data_dir).resolve()
    audio_root = Path(args.audio_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    export_dir = Path(args.export_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    train_records = usable_records(load_jsonl(data_dir / "train.jsonl"))
    test_records = usable_records(load_jsonl(data_dir / "test.jsonl"))
    binary_train_records = load_jsonl(binary_data_dir / "train.jsonl")
    asr_audio = build_audio_matrix(
        train_records + test_records,
        audio_root,
        output_dir / "asr_audio_features.npz",
    )
    full_train_audio = build_audio_matrix(
        binary_train_records,
        audio_root,
        output_dir / "full_train_audio_features.npz",
    )
    x_test_audio = asr_audio[len(train_records) :]
    train_texts = [record["text"].strip() for record in train_records]
    test_texts = [record["text"].strip() for record in test_records]
    y_train = np.asarray([LABEL_TO_ID[record["label"]] for record in train_records])
    y_test = np.asarray([LABEL_TO_ID[record["label"]] for record in test_records])
    train_rules = np.asarray(
        [score_text_rules(normalize_chinese_text(text)) for text in train_texts]
    )
    test_rules = np.asarray(
        [score_text_rules(normalize_chinese_text(text)) for text in test_texts]
    )

    fit_indices, validation_indices = train_test_split(
        np.arange(len(train_records)),
        test_size=80,
        stratify=y_train,
        random_state=args.seed,
    )
    validation_paths = {train_records[index]["audio_path"] for index in validation_indices}
    audio_fit_indices = np.asarray(
        [
            index
            for index, record in enumerate(binary_train_records)
            if record["audio_path"] not in validation_paths
        ]
    )
    path_to_full_index = {
        record["audio_path"]: index for index, record in enumerate(binary_train_records)
    }
    validation_audio_indices = np.asarray(
        [path_to_full_index[train_records[index]["audio_path"]] for index in validation_indices]
    )
    y_full_train = np.asarray(
        [LABEL_TO_ID[record["label"]] for record in binary_train_records]
    )

    validation_audio_model = audio_pipeline(args.seed)
    validation_audio_model.fit(
        full_train_audio[audio_fit_indices],
        y_full_train[audio_fit_indices],
    )
    validation_audio_scores = validation_audio_model.predict_proba(
        full_train_audio[validation_audio_indices]
    )[:, 1]

    validation_text_model = text_pipeline(args.seed)
    validation_text_model.fit(
        [train_texts[index] for index in fit_indices],
        y_train[fit_indices],
    )
    validation_text_scores = validation_text_model.predict_proba(
        [train_texts[index] for index in validation_indices]
    )[:, 1]
    validation_rule_scores = train_rules[validation_indices]
    y_validation = y_train[validation_indices]

    audio_reliability = evaluate(y_validation, validation_audio_scores, 0.5)["macro_f1"]
    text_reliability = evaluate(y_validation, validation_text_scores, 0.5)["macro_f1"]
    config = choose_parameters(
        y_validation,
        validation_audio_scores,
        validation_text_scores,
        validation_rule_scores,
        audio_reliability,
        text_reliability,
    )
    config.update(
        {
            "name": "callguard_caef",
            "version": "1.0",
            "description": "Confidence-Aware Explainable Fusion",
            "trained_on": "TeleAntiFraud with a disjoint 160/80 fusion train-validation split",
        }
    )

    full_audio_model = joblib.load(PROJECT_ROOT / "ml" / "models" / "audio_baseline" / "model.joblib")
    full_text_model = joblib.load(PROJECT_ROOT / "ml" / "models" / "text_baseline" / "model.joblib")
    test_audio = full_audio_model.predict_proba(x_test_audio)[:, 1]
    test_text = full_text_model.predict_proba(test_texts)[:, 1]

    fixed_text = 0.8 * test_text + 0.2 * test_rules
    fixed_fusion = 0.65 * test_audio + 0.35 * fixed_text
    adaptive_scores = np.asarray(
        [
            adaptive_fusion(audio, text, rule, config).risk_score
            for audio, text, rule in zip(test_audio, test_text, test_rules, strict=False)
        ]
    )

    metrics = {
        "method": config,
        "dataset": {
            "train_samples": len(train_records),
            "fusion_fit_samples": len(fit_indices),
            "fusion_validation_samples": len(validation_indices),
            "test_samples": len(test_records),
            "audio_fit_samples": len(audio_fit_indices),
        },
        "validation_reliability": {
            "audio_macro_f1": audio_reliability,
            "text_macro_f1": text_reliability,
        },
        "comparisons": {
            "audio_only": evaluate(y_test, test_audio, 0.5),
            "text_only": evaluate(y_test, test_text, 0.5),
            "rules_only": evaluate(y_test, test_rules, 0.5),
            "fixed_65_35": evaluate(y_test, fixed_fusion, 0.5),
            "callguard_caef": evaluate(
                y_test,
                adaptive_scores,
                float(config["decision_threshold"]),
            ),
        },
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (export_dir / "adaptive_fusion.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"exported: {export_dir / 'adaptive_fusion.json'}")


if __name__ == "__main__":
    main()
