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


class FeatureStatus(BaseModel):
    id: str
    name: str
    enabled: bool
    status: str
    detail: str


class DeploymentStatusResponse(BaseModel):
    mode: str
    demo_mode: bool
    api_version: str
    database_path: str
    features: list[FeatureStatus]
    limitations: list[str]
    next_steps: list[str]


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


class RiskTimelineSegment(BaseModel):
    index: int
    start: float
    end: float
    risk_score: float
    risk_level: str
    audio_score: float | None
    text_score: float | None
    pressure_score: float | None
    transcript: str
    factors: list[RiskFactor]


class RiskTimelineSummary(BaseModel):
    duration_seconds: float
    window_seconds: float
    hop_seconds: float
    segment_count: int
    medium_or_high_segments: int
    high_risk_segments: int
    peak_risk_score: float
    peak_start: float | None
    peak_end: float | None


class AnalyzeCallResponse(BaseModel):
    record_id: int | None = None
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
    timeline: list[RiskTimelineSegment] = Field(default_factory=list)
    timeline_summary: RiskTimelineSummary | None = None
    suggestion: str
    notes: list[str]


class RuleCreateRequest(BaseModel):
    group: str = Field(min_length=1)
    keyword: str = Field(min_length=1)
    weight: float = Field(default=1.0, ge=0.1, le=5.0)
    enabled: bool = True


class RuleUpdateRequest(BaseModel):
    group: str | None = Field(default=None, min_length=1)
    keyword: str | None = Field(default=None, min_length=1)
    weight: float | None = Field(default=None, ge=0.1, le=5.0)
    enabled: bool | None = None


class RuleResponse(BaseModel):
    id: int
    group: str
    keyword: str
    weight: float
    enabled: bool
    source: str
    created_at: str
    updated_at: str


class DeleteResponse(BaseModel):
    ok: bool
    message: str


class CallRecordSummary(BaseModel):
    id: int
    created_at: str
    input_type: str
    file_name: str | None
    prediction: str
    risk_score: float
    risk_level: str
    pressure_score: float | None
    pressure_level: str | None
    transcript_preview: str
    risk_factors: list[RiskFactor]
    model_summary: dict[str, str | float | int | None]


class CallRecordDetail(CallRecordSummary):
    transcript: str
    analysis_result: dict


class CallListResponse(BaseModel):
    records: list[CallRecordSummary]
    total: int
    limit: int
    offset: int


class KeywordCount(BaseModel):
    keyword: str
    count: int


class AnalyticsSummaryResponse(BaseModel):
    total_calls: int
    high_risk_calls: int
    average_risk_score: float
    risk_level_counts: dict[str, int]
    prediction_counts: dict[str, int]
    top_keywords: list[KeywordCount]
    recent_calls: list[CallRecordSummary]
