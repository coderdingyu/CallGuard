from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeTextRequest(BaseModel):
    text: str = Field(min_length=1, description="Call transcript or user-provided text.")


class RiskFactor(BaseModel):
    group: str
    keyword: str


class DemoSampleResponse(BaseModel):
    id: str
    title: str
    scenario: str
    label: str
    description: str
    transcript: str
    has_audio: bool
    audio_file_name: str | None = None


class AnalyzeTextResponse(BaseModel):
    risk_score: float
    risk_level: str
    factors: list[RiskFactor]
    suggestion: str
    model: str
    model_score: float | None = None
    model_prediction: str | None = None
    rule_score: float
    evidence_terms: list[str] = Field(default_factory=list)


class AnalyzeAudioResponse(BaseModel):
    file_name: str
    prediction: str
    risk_score: float
    risk_level: str
    probabilities: dict[str, float]
    model: str
    feature_window_seconds: float
    suggestion: str
    notes: list[str]


class EmotionPressureDriver(BaseModel):
    emotion: str
    probability: float
    pressure_weight: float
    contribution: float


class AnalyzeEmotionResponse(BaseModel):
    file_name: str
    primary_emotion: str
    emotion_confidence: float
    pressure_score: float
    pressure_level: str
    probabilities: dict[str, float]
    pressure_probabilities: dict[str, float]
    pressure_drivers: list[EmotionPressureDriver]
    model: str
    feature_window_seconds: float
    suggestion: str
    notes: list[str]


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    text: str
    raw_text: str | None = None
    source: str
    language: str
    language_probability: float
    duration_seconds: float
    model: str
    segments: list[TranscriptionSegment]


class FusionDiagnostics(BaseModel):
    audio_confidence: float
    text_confidence: float
    agreement: float
    rule_adjustment: float


class AnalyzeCallResponse(BaseModel):
    prediction: str
    risk_score: float
    risk_level: str
    transcript: str
    transcript_source: str
    asr: TranscriptionResponse | None
    audio: AnalyzeAudioResponse | None
    emotion: AnalyzeEmotionResponse | None
    text: AnalyzeTextResponse | None
    fusion_weights: dict[str, float]
    fusion_method: str
    decision_threshold: float
    fusion_diagnostics: FusionDiagnostics | None
    suggestion: str
    notes: list[str]
