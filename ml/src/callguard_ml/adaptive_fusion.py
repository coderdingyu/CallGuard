from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdaptiveFusionResult:
    risk_score: float
    audio_weight: float
    text_weight: float
    audio_confidence: float
    text_confidence: float
    agreement: float
    rule_adjustment: float


def load_fusion_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def confidence(score: float) -> float:
    return min(1.0, max(0.0, 2.0 * abs(float(score) - 0.5)))


def adaptive_fusion(
    audio_score: float,
    text_score: float,
    rule_score: float,
    config: dict[str, Any],
) -> AdaptiveFusionResult:
    audio_score = clamp(audio_score)
    text_score = clamp(text_score)
    rule_score = clamp(rule_score)

    audio_confidence = confidence(audio_score)
    text_confidence = confidence(text_score)
    floor = float(config["confidence_floor"])
    audio_reliability = float(config["audio_reliability"])
    text_reliability = float(config["text_reliability"])
    reliability_power = float(config.get("reliability_power", 1.0))
    text_confidence_gate = float(config.get("text_confidence_gate", 1.1))

    if text_confidence >= text_confidence_gate:
        audio_weight = 0.0
        text_weight = 1.0
    else:
        raw_audio_weight = audio_reliability**reliability_power * (
            floor + (1.0 - floor) * audio_confidence
        )
        raw_text_weight = text_reliability**reliability_power * (
            floor + (1.0 - floor) * text_confidence
        )
        total_weight = raw_audio_weight + raw_text_weight
        audio_weight = raw_audio_weight / total_weight
        text_weight = raw_text_weight / total_weight

    base_score = audio_weight * audio_score + text_weight * text_score
    rule_support = rule_score * text_score
    rule_adjustment = float(config["rule_boost"]) * rule_support * (1.0 - base_score)
    score = clamp(base_score + rule_adjustment)

    return AdaptiveFusionResult(
        risk_score=round(score, 4),
        audio_weight=round(audio_weight, 4),
        text_weight=round(text_weight, 4),
        audio_confidence=round(audio_confidence, 4),
        text_confidence=round(text_confidence, 4),
        agreement=round(1.0 - abs(audio_score - text_score), 4),
        rule_adjustment=round(rule_adjustment, 4),
    )


def clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
