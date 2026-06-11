from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

from callguard_ml.risk_rules import find_risk_keywords, score_text_rules  # noqa: E402


def to_binary(label: str) -> str:
    return "risk" if label in {"fraud", "medium_risk", "high_risk"} else "normal"


def predict(score: float) -> str:
    return "risk" if score >= 0.25 else "normal"


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "demo_text_risk" / "demo.jsonl"
    output_path = PROJECT_ROOT / "experiments" / "rules_baseline_demo" / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    rows = []
    correct = 0
    for record in records:
        score = score_text_rules(record["text"])
        y_true = to_binary(record["label"])
        y_pred = predict(score)
        correct += int(y_true == y_pred)
        rows.append(
            {
                "id": record["id"],
                "label": record["label"],
                "binary_label": y_true,
                "prediction": y_pred,
                "score": score,
                "keywords": [match.__dict__ for match in find_risk_keywords(record["text"])],
            }
        )

    metrics = {
        "dataset": "callguard_demo",
        "samples": len(records),
        "accuracy": round(correct / len(records), 4) if records else 0,
        "predictions": rows,
    }
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "predictions"}, ensure_ascii=False, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()

