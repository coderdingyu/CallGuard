from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION_SECONDS = 5.0


def summarize_feature(values: np.ndarray) -> list[float]:
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    ]


def extract_audio_features(
    audio_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration: float = DEFAULT_DURATION_SECONDS,
    offset: float = 0.0,
) -> np.ndarray:
    path = Path(audio_path)
    y, sr = librosa.load(path, sr=sample_rate, mono=True, offset=offset, duration=duration)
    if y.size == 0:
        raise ValueError(f"empty audio: {path}")

    features: list[float] = []
    features.append(float(librosa.get_duration(y=y, sr=sr)))
    features.extend(summarize_feature(librosa.feature.rms(y=y)[0]))
    features.extend(summarize_feature(librosa.feature.zero_crossing_rate(y=y)[0]))
    features.extend(summarize_feature(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
    features.extend(summarize_feature(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))
    features.extend(summarize_feature(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1).astype(float).tolist())
    features.extend(np.std(mfcc, axis=1).astype(float).tolist())

    return np.asarray(features, dtype=np.float32)


def get_audio_duration_seconds(audio_path: str | Path) -> float:
    path = Path(audio_path)
    return float(librosa.get_duration(path=path))
