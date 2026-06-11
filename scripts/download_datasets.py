from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
CSEMOTIONS_FILES = [
    "README.md",
    "dataset_infos.json",
    "NOTICE",
    *[f"data/train-{index:05d}-of-00008.parquet" for index in range(8)],
]


def run_command(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run(args, check=True)


def run_curl_download(url: str, target_path: Path, token: str | None) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
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
        str(target_path),
    ]
    if token:
        args.extend(["--header", f"Authorization: Bearer {token}"])
    args.append(url)
    run_command(args)


def dataset_resolve_url(endpoint: str, repo_id: str, filename: str) -> str:
    return f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main/{filename}"


def download_teleantifraud(data_root: Path, priority_only: bool, method: str, endpoint: str) -> None:
    raw_dir = data_root / "raw" / "teleantifraud"
    raw_dir.mkdir(parents=True, exist_ok=True)

    priority_files = [
        "binary_classification.zip",
        "audio.zip",
        "dataset_manifest.json",
    ]

    if method == "curl":
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit(
                "TeleAntiFraud is a gated dataset. Set HF_TOKEN before downloading, "
                "or download the files in your browser after accepting the dataset terms."
            )
        files = priority_files if priority_only else ["*"]
        if files == ["*"]:
            raise SystemExit("curl method supports priority files only. Use --method hf for all files.")
        for filename in files:
            url = dataset_resolve_url(endpoint, "JimmyMa99/TeleAntiFraud", filename)
            run_curl_download(url, raw_dir / filename, token)
        return

    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)

    args = [
        sys.executable,
        "-m",
        "huggingface_hub.commands.huggingface_cli",
        "download",
        "JimmyMa99/TeleAntiFraud",
        "--repo-type",
        "dataset",
        "--local-dir",
        str(raw_dir),
    ]
    if priority_only:
        args.extend(["--include", *priority_files])

    run_command(args)


def download_csemotions(data_root: Path, method: str, endpoint: str) -> None:
    raw_dir = data_root / "raw" / "csemotions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if method == "curl":
        for filename in CSEMOTIONS_FILES:
            url = dataset_resolve_url(endpoint, "AIDC-AI/CSEMOTIONS", filename)
            run_curl_download(url, raw_dir / filename, token=None)
        return

    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)

    args = [
        sys.executable,
        "-m",
        "huggingface_hub.commands.huggingface_cli",
        "download",
        "AIDC-AI/CSEMOTIONS",
        "--repo-type",
        "dataset",
        "--local-dir",
        str(raw_dir),
        "--include",
        "README.md",
        "dataset_infos.json",
        "NOTICE",
        "data/*.parquet",
    ]
    run_command(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CallGuard datasets.")
    parser.add_argument(
        "--dataset",
        choices=["teleantifraud", "csemotions"],
        default="teleantifraud",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CALLGUARD_DATA_ROOT", str(DEFAULT_DATA_ROOT)),
        help="Project data root.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Hugging Face endpoint. Use https://hf-mirror.com in mainland China.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download all files instead of the first-priority subset.",
    )
    parser.add_argument(
        "--method",
        choices=["hf", "curl"],
        default="hf",
        help="Download method. Use curl when the Hugging Face CLI has local SSL/proxy issues.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if args.dataset == "teleantifraud":
        download_teleantifraud(
            data_root,
            priority_only=not args.all_files,
            method=args.method,
            endpoint=args.endpoint,
        )
    elif args.dataset == "csemotions":
        download_csemotions(data_root, method=args.method, endpoint=args.endpoint)


if __name__ == "__main__":
    main()
