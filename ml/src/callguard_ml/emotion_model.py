from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


EMOTION_LABELS = [
    "neutral",
    "happy",
    "angry",
    "sad",
    "surprise",
    "playfulness",
    "fearful",
]

EMOTION_DISPLAY_NAMES = {
    "neutral": "Neutral",
    "happy": "Happy",
    "angry": "Angry",
    "sad": "Sad",
    "surprise": "Surprise",
    "playfulness": "Playfulness",
    "fearful": "Fearful",
}

PRESSURE_WEIGHTS = {
    "neutral": 0.10,
    "happy": 0.18,
    "playfulness": 0.25,
    "sad": 0.55,
    "surprise": 0.70,
    "fearful": 0.90,
    "angry": 0.95,
}

PRESSURE_LABELS = ["normal", "medium", "high"]

PRESSURE_LEVEL_SCORES = {
    "normal": 0.18,
    "medium": 0.62,
    "high": 0.925,
}


@dataclass(frozen=True)
class EmotionPrediction:
    primary_emotion: str
    emotion_confidence: float
    pressure_score: float
    pressure_level: str
    probabilities: dict[str, float]
    pressure_probabilities: dict[str, float]
    pressure_drivers: list[dict[str, float | str]]


def pressure_weight_for_emotion(emotion: str) -> float:
    return float(PRESSURE_WEIGHTS[emotion])


def pressure_level_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.30:
        return "low"
    return "normal"


def load_emotion_model(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {
        "model": artifact,
        "emotion_labels": EMOTION_LABELS,
        "pressure_weights": PRESSURE_WEIGHTS,
    }


def analyze_emotion_model(artifact: dict[str, Any], features: np.ndarray) -> EmotionPrediction:
    model = artifact["model"]
    labels = list(artifact.get("emotion_labels", EMOTION_LABELS))
    pressure_weights = dict(artifact.get("pressure_weights", PRESSURE_WEIGHTS))

    probabilities_array = model.predict_proba(features.reshape(1, -1))[0]
    probabilities = {
        label: float(probabilities_array[index]) for index, label in enumerate(labels)
    }
    primary_emotion = max(probabilities, key=probabilities.get)
    emotion_confidence = probabilities[primary_emotion]
    pressure_model = artifact.get("pressure_model")
    if pressure_model is not None:
        pressure_labels = list(artifact.get("pressure_labels", PRESSURE_LABELS))
        pressure_level_scores = dict(
            artifact.get("pressure_level_scores", PRESSURE_LEVEL_SCORES)
        )
        pressure_array = pressure_model.predict_proba(features.reshape(1, -1))[0]
        pressure_probabilities = {
            label: float(pressure_array[index])
            for index, label in enumerate(pressure_labels)
        }
        pressure_score = sum(
            pressure_probabilities[label] * float(pressure_level_scores[label])
            for label in pressure_labels
        )
        pressure_level = max(pressure_probabilities, key=pressure_probabilities.get)
    else:
        pressure_score = sum(
            probabilities[label] * float(pressure_weights[label]) for label in labels
        )
        pressure_level = pressure_level_from_score(pressure_score)
        pressure_probabilities = {}

    pressure_drivers = sorted(
        [
            {
                "emotion": label,
                "probability": probabilities[label],
                "pressure_weight": float(pressure_weights[label]),
                "contribution": probabilities[label] * float(pressure_weights[label]),
            }
            for label in labels
        ],
        key=lambda item: float(item["contribution"]),
        reverse=True,
    )[:3]

    return EmotionPrediction(
        primary_emotion=primary_emotion,
        emotion_confidence=emotion_confidence,
        pressure_score=pressure_score,
        pressure_level=pressure_level,
        probabilities=probabilities,
        pressure_probabilities=pressure_probabilities,
        pressure_drivers=pressure_drivers,
    )
