from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

from callguard_ml.emotion_model import (  # noqa: E402
    EMOTION_LABELS,
    pressure_level_from_score,
    pressure_weight_for_emotion,
)


def normalize_emotion(value: str) -> str:
    emotion = value.strip().lower()
    aliases = {
        "neutral": "neutral",
        "happy": "happy",
        "angry": "angry",
        "sad": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
        "playfulness": "playfulness",
        "playful": "playfulness",
        "fearful": "fearful",
        "fear": "fearful",
    }
    if emotion not in aliases:
        raise ValueError(f"unknown emotion label: {value}")
    return aliases[emotion]


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "unknown"


def iter_csemotions_rows(parquet_dir: Path) -> Iterable[dict[str, Any]]:
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found in {parquet_dir}")

    for parquet_file in files:
        dataset_file = parquet_file.name
        parquet = pq.ParquetFile(parquet_file)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["audio", "text", "emotion", "speaker"])
            for row_index, row in enumerate(table.to_pylist()):
                row["source_file"] = dataset_file
                row["source_row_group"] = row_group
                row["source_row_index"] = row_index
                yield row


def stratified_split(
    records: list[dict[str, Any]],
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(record["emotion"], []).append(record)

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label in sorted(by_label):
        bucket = by_label[label][:]
        rng.shuffle(bucket)
        test_count = max(1, round(len(bucket) * test_size))
        test.extend(bucket[:test_count])
        train.extend(bucket[test_count:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_by_label(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in EMOTION_LABELS}
    for record in records:
        counts[record["emotion"]] = counts.get(record["emotion"], 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CSEMOTIONS for CallGuard.")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw" / "csemotions"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--overwrite-audio", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    parquet_dir = raw_dir / "data"
    data_root = Path(args.data_root).resolve()
    audio_root = data_root / "interim" / "csemotions" / "audio"
    processed_dir = data_root / "processed" / "csemotions_emotion"
    audio_root.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    row_iter = iter_csemotions_rows(parquet_dir)
    total = args.max_items
    for index, row in enumerate(tqdm(row_iter, total=total, desc="prepare CSEMOTIONS")):
        if args.max_items is not None and index >= args.max_items:
            break

        audio = row.get("audio") or {}
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            continue

        emotion = normalize_emotion(str(row["emotion"]))
        speaker = safe_name(str(row.get("speaker", "unknown")))
        sample_id = f"csemotions_{index:05d}"
        audio_relative = Path(emotion) / f"{speaker}_{sample_id}.wav"
        audio_path = audio_root / audio_relative
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite_audio or not audio_path.exists():
            audio_path.write_bytes(audio_bytes)

        pressure_score = pressure_weight_for_emotion(emotion)
        records.append(
            {
                "id": sample_id,
                "audio_path": audio_relative.as_posix(),
                "text": str(row.get("text", "")).strip(),
                "emotion": emotion,
                "speaker": speaker,
                "pressure_score": pressure_score,
                "pressure_level": pressure_level_from_score(pressure_score),
                "source_dataset": "CSEMOTIONS",
                "source_file": row["source_file"],
                "source_row_group": row["source_row_group"],
                "source_row_index": row["source_row_index"],
            }
        )

    if not records:
        raise RuntimeError("no usable CSEMOTIONS records were prepared")

    train, test = stratified_split(records, test_size=args.test_size, seed=args.seed)
    write_jsonl(processed_dir / "train.jsonl", train)
    write_jsonl(processed_dir / "test.jsonl", test)

    summary = {
        "dataset": "CSEMOTIONS",
        "records": len(records),
        "train_samples": len(train),
        "test_samples": len(test),
        "test_size": args.test_size,
        "seed": args.seed,
        "audio_root": str(audio_root.relative_to(PROJECT_ROOT)),
        "processed_dir": str(processed_dir.relative_to(PROJECT_ROOT)),
        "label_counts": count_by_label(records),
        "train_label_counts": count_by_label(train),
        "test_label_counts": count_by_label(test),
        "pressure_mapping": {
            label: pressure_weight_for_emotion(label) for label in EMOTION_LABELS
        },
    }
    (processed_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
