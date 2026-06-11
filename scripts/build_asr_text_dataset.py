from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

from callguard_ml.asr import load_whisper_model, transcribe_audio  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = load_jsonl(path)
    return {str(record["audio_path"]): record for record in records}


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def transcribe_split(
    split: str,
    records: list[dict[str, Any]],
    output_path: Path,
    audio_root: Path,
    model,
) -> None:
    existing = read_existing(output_path)
    result_records = list(existing.values())

    for record in tqdm(records, desc=f"ASR {split}"):
        audio_key = str(record["audio_path"])
        if audio_key in existing and existing[audio_key].get("text"):
            continue

        audio_path = audio_root / audio_key
        try:
            result = transcribe_audio(model, audio_path)
            text = result.text.strip()
            status = "ok" if text else "empty"
            error = ""
        except Exception as exc:  # noqa: BLE001 - keep dataset generation resumable.
            text = ""
            status = "failed"
            error = str(exc)

        next_record = {
            "dataset": "teleantifraud_asr",
            "source_dataset": record.get("dataset", "teleantifraud"),
            "split": split,
            "audio_path": audio_key,
            "label": record["label"],
            "text": text,
            "asr_status": status,
            "asr_error": error,
        }
        existing[audio_key] = next_record
        result_records.append(next_record)
        write_jsonl(result_records, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ASR transcript dataset.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed" / "teleantifraud_asr"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "ml" / "models" / "asr" / "faster-whisper-base"))
    parser.add_argument("--max-train", type=int, default=240)
    parser.add_argument("--max-test", type=int, default=80)
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

    model = load_whisper_model(Path(args.model_path).resolve())
    transcribe_split("train", train_records, output_dir / "train.jsonl", audio_root, model)
    transcribe_split("test", test_records, output_dir / "test.jsonl", audio_root, model)

    summary = {
        "dataset": "teleantifraud_asr",
        "model_path": str(Path(args.model_path).resolve()),
        "train_records": len(load_jsonl(output_dir / "train.jsonl")),
        "test_records": len(load_jsonl(output_dir / "test.jsonl")),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
