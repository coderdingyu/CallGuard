from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
}
MODEL_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local faster-whisper models.")
    parser.add_argument("--size", choices=sorted(MODEL_REPOS), default="small")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to ml/models/asr/faster-whisper-<size>.",
    )
    parser.add_argument(
        "--mirror",
        default="https://hf-mirror.com",
        help="Set HF_ENDPOINT before download. Use empty string to disable.",
    )
    parser.add_argument(
        "--method",
        choices=["curl", "huggingface"],
        default="curl",
    )
    args = parser.parse_args()

    if args.mirror:
        os.environ.setdefault("HF_ENDPOINT", args.mirror)

    repo_id = MODEL_REPOS[args.size]
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT / "ml" / "models" / "asr" / f"faster-whisper-{args.size}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "curl":
        download_with_curl(repo_id, output_dir, args.mirror or "https://huggingface.co")
    else:
        snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    print(f"downloaded {repo_id} -> {output_dir}")


def download_with_curl(repo_id: str, output_dir: Path, endpoint: str) -> None:
    base_url = endpoint.rstrip("/")
    for file_name in MODEL_FILES:
        target = output_dir / file_name
        url = f"{base_url}/{repo_id}/resolve/main/{file_name}"
        command = [
            "curl.exe",
            "-L",
            "--fail",
            "--retry",
            "10",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "-C",
            "-",
            "--output",
            str(target),
            url,
        ]
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
