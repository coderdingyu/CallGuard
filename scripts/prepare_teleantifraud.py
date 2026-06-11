from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    if not zip_path.exists():
        print(f"missing: {zip_path}")
        return
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"skip existing extract target: {target_dir}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)
    print(f"extracted: {zip_path} -> {target_dir}")


def iter_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")

    if path.suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path).to_dict(orient="records")

    if path.suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "train", "test", "records"):
                if isinstance(data.get(key), list):
                    return data[key]
    return []


def normalize_record(record: dict[str, Any], split: str) -> dict[str, Any]:
    raw_label = record.get("label", record.get("answer", ""))
    label = str(raw_label).strip().lower()
    normalized_label = "fraud" if label in {"fraud", "true", "1", "yes"} else "normal"
    audio_path, instruction = extract_prompt_fields(record)

    return {
        "id": record.get("id"),
        "dataset": "teleantifraud",
        "split": split,
        "task": record.get("task", "binary_fraud_detection"),
        "audio_path": record.get("audio_path", audio_path),
        "instruction": record.get("instruction", instruction),
        "input": record.get("input", ""),
        "text": record.get("text", record.get("transcript", "")),
        "label": normalized_label,
        "raw_label": raw_label,
    }


def extract_prompt_fields(record: dict[str, Any]) -> tuple[str, str]:
    audio_path = ""
    instruction = ""
    prompt = record.get("prompt", [])
    if not isinstance(prompt, list):
        return audio_path, instruction

    for message in prompt:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            instruction = content
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "audio":
                audio_path = str(item.get("audio_url", ""))
            elif item.get("type") == "text":
                instruction = str(item.get("text", ""))
    return audio_path, instruction


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote: {path} ({len(records)} records)")


def prepare(data_root: Path) -> None:
    raw_dir = data_root / "raw" / "teleantifraud"
    interim_dir = data_root / "interim" / "teleantifraud"
    processed_dir = data_root / "processed" / "teleantifraud_binary"

    extract_zip(raw_dir / "binary_classification.zip", interim_dir / "binary_classification")
    extract_zip(raw_dir / "audio.zip", interim_dir / "audio")

    search_roots = [
        raw_dir,
        interim_dir / "binary_classification",
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("*.json", "*.jsonl", "*.parquet", "*.csv"):
            candidates.extend(root.rglob(pattern))

    if not candidates:
        print(
            "no metadata files found. Put TeleAntiFraud files in "
            f"{raw_dir} and run this script again."
        )
        return

    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for candidate in candidates:
        name = candidate.name.lower()
        split = "test" if "test" in name or "dev" in name or "valid" in name else "train"
        normalized = [normalize_record(record, split) for record in iter_records(candidate)]
        if split == "test":
            test_records.extend(normalized)
        else:
            train_records.extend(normalized)

    write_jsonl(train_records, processed_dir / "train.jsonl")
    write_jsonl(test_records, processed_dir / "test.jsonl")

    summary = {
        "dataset": "teleantifraud",
        "train_records": len(train_records),
        "test_records": len(test_records),
        "labels": {
            "train": count_labels(train_records),
            "test": count_labels(test_records),
        },
    }
    (processed_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def count_labels(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = record["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TeleAntiFraud binary data.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    args = parser.parse_args()
    prepare(Path(args.data_root).expanduser().resolve())


if __name__ == "__main__":
    main()
