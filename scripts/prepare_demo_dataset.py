from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = PROJECT_ROOT / "data" / "external" / "demo_calls" / "demo_calls.jsonl"
    target = PROJECT_ROOT / "data" / "processed" / "demo_text_risk" / "demo.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records.append(
            {
                "id": record["id"],
                "dataset": "callguard_demo",
                "split": record["split"],
                "text": record["text"],
                "label": record["label"],
                "scenario": record["scenario"],
            }
        )

    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "callguard_demo",
        "records": len(records),
        "labels": count_labels(records),
    }
    (target.parent / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def count_labels(records: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = record["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    main()

