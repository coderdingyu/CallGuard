from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def path_status(path: Path) -> dict[str, object]:
    exists = path.exists()
    file_count = 0
    if exists:
        file_count = 1 if path.is_file() else sum(1 for item in path.rglob("*") if item.is_file())
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": exists,
        "files": file_count,
    }


def main() -> None:
    checks = [
        PROJECT_ROOT / "apps" / "web",
        PROJECT_ROOT / "apps" / "api",
        PROJECT_ROOT / "ml" / "src" / "callguard_ml",
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "ml" / "models" / "audio_baseline" / "model.joblib",
        PROJECT_ROOT / "ml" / "models" / "emotion_baseline" / "model.joblib",
        PROJECT_ROOT / "ml" / "models" / "text_baseline" / "model.joblib",
        PROJECT_ROOT / "ml" / "models" / "fusion" / "adaptive_fusion.json",
        PROJECT_ROOT / "ml" / "models" / "asr" / "faster-whisper-small",
        PROJECT_ROOT / "ml" / "models" / "asr" / "faster-whisper-base",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "reports",
    ]
    print(json.dumps([path_status(path) for path in checks], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
