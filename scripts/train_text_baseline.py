from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_ID = {"normal": 0, "fraud": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prepare_examples(records: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for record in records:
        text = str(record.get("text", "")).strip()
        label = str(record.get("label", "")).strip()
        if text and label in LABEL_TO_ID:
            texts.append(text)
            labels.append(LABEL_TO_ID[label])
    return texts, labels


def top_terms(model: Pipeline, top_k: int = 20) -> dict[str, list[str]]:
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    fraud_indices = coefficients.argsort()[-top_k:][::-1]
    normal_indices = coefficients.argsort()[:top_k]
    return {
        "fraud": [str(feature_names[index]) for index in fraud_indices],
        "normal": [str(feature_names[index]) for index in normal_indices],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a text-risk baseline from ASR transcripts.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed" / "teleantifraud_asr"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "text_baseline_asr"))
    parser.add_argument("--export-dir", default=str(PROJECT_ROOT / "ml" / "models" / "text_baseline"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    export_dir = Path(args.export_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    train_texts, y_train = prepare_examples(load_jsonl(data_dir / "train.jsonl"))
    test_texts, y_test = prepare_examples(load_jsonl(data_dir / "test.jsonl"))
    if not train_texts or not test_texts:
        raise RuntimeError("ASR text dataset is empty. Run build_asr_text_dataset.py first.")

    model = Pipeline(
        steps=[
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
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(train_texts, y_train)

    predictions = model.predict(test_texts)
    report = classification_report(
        y_test,
        predictions,
        target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": "tfidf_char_ngram_logistic_regression",
        "dataset": "teleantifraud_asr",
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "classification_report": report,
        "top_terms": top_terms(model),
    }

    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(model, export_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (export_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, ensure_ascii=False, indent=2))
    print(f"wrote: {output_dir}")
    print(f"exported: {export_dir}")


if __name__ == "__main__":
    main()
