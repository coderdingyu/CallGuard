from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

from callguard_ml.asr import load_whisper_model, transcribe_audio  # noqa: E402
from callguard_ml.adaptive_fusion import adaptive_fusion, load_fusion_config  # noqa: E402
from callguard_ml.audio_features import (  # noqa: E402
    DEFAULT_DURATION_SECONDS,
    DEFAULT_SAMPLE_RATE,
    extract_audio_features,
)
from callguard_ml.emotion_model import (  # noqa: E402
    analyze_emotion_model,
    load_emotion_model as load_emotion_model_file,
)
from callguard_ml.risk_rules import DEFAULT_KEYWORD_GROUPS, find_risk_keywords, score_text_rules  # noqa: E402
from callguard_ml.text_model import analyze_text_model, load_text_model as load_text_model_file  # noqa: E402
from callguard_ml.text_normalization import normalize_chinese_text, simplify_chinese_text  # noqa: E402

from .schemas import (  # noqa: E402
    AnalyzeAudioResponse,
    AnalyzeCallResponse,
    AnalyzeEmotionResponse,
    AnalyzeTextRequest,
    AnalyzeTextResponse,
    AnalyticsSummaryResponse,
    CallListResponse,
    CallRecordDetail,
    DemoSampleResponse,
    DeleteResponse,
    EmotionPressureDriver,
    FusionDiagnostics,
    RiskFactor,
    RuleCreateRequest,
    RuleResponse,
    RuleUpdateRequest,
    TranscriptionResponse,
    TranscriptionSegment,
)
from .storage import (  # noqa: E402
    analytics_summary,
    create_call_record,
    create_rule,
    delete_call_record,
    delete_custom_rule,
    get_call_record,
    get_enabled_keyword_groups,
    init_db,
    list_call_records,
    list_rules,
    reset_default_rules,
    toggle_rule,
    update_rule,
)

AUDIO_ROOTS = {
    "teleantifraud": PROJECT_ROOT / "data" / "interim" / "teleantifraud" / "audio",
    "csemotions": PROJECT_ROOT / "data" / "interim" / "csemotions" / "audio",
}

AUDIO_MODEL_PATH = Path(
    os.environ.get(
        "CALLGUARD_AUDIO_MODEL_PATH",
        PROJECT_ROOT / "ml" / "models" / "audio_baseline" / "model.joblib",
    )
)
ASR_SMALL_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "asr" / "faster-whisper-small"
ASR_BASE_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "asr" / "faster-whisper-base"
ASR_MODEL_PATH = Path(os.environ["CALLGUARD_ASR_MODEL_PATH"]) if os.environ.get("CALLGUARD_ASR_MODEL_PATH") else None
ASR_FALLBACK_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "asr" / "faster-whisper-tiny"
TEXT_MODEL_PATH = Path(
    os.environ.get(
        "CALLGUARD_TEXT_MODEL_PATH",
        PROJECT_ROOT / "ml" / "models" / "text_baseline" / "model.joblib",
    )
)
FUSION_CONFIG_PATH = Path(
    os.environ.get(
        "CALLGUARD_FUSION_CONFIG_PATH",
        PROJECT_ROOT / "ml" / "models" / "fusion" / "adaptive_fusion.json",
    )
)
EMOTION_MODEL_PATH = Path(
    os.environ.get(
        "CALLGUARD_EMOTION_MODEL_PATH",
        PROJECT_ROOT / "ml" / "models" / "emotion_baseline" / "model.joblib",
    )
)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


DEMO_SAMPLES = [
    {
        "id": "text_police_fraud",
        "title": "冒充公检法",
        "scenario": "police_impersonation",
        "label": "fraud",
        "description": "高风险文本样例，包含安全账户、保密和转账压力。",
        "transcript": "我是公安机关工作人员，你的银行卡涉嫌洗钱，现在必须马上把资金转到安全账户，不要告诉任何人。",
    },
    {
        "id": "text_refund_fraud",
        "title": "客服退款诈骗",
        "scenario": "customer_service_refund",
        "label": "fraud",
        "description": "高风险文本样例，包含验证码、银行卡和限时处理。",
        "transcript": "这里是客服中心，你的订单出现异常退款，需要你提供银行卡号和验证码，否则今天就无法处理。",
    },
    {
        "id": "text_normal_service",
        "title": "正常预约提醒",
        "scenario": "normal_service",
        "label": "normal",
        "description": "正常服务类文本样例，用来对比风险词和融合分数。",
        "transcript": "您好，这里是医院预约中心，提醒您明天上午九点有体检预约，如需改期可以在公众号操作。",
    },
    {
        "id": "audio_tele_fraud",
        "title": "诈骗音频样例",
        "scenario": "teleantifraud_fraud",
        "label": "fraud",
        "description": "来自 TeleAntiFraud 的风险通话音频，用于完整 ASR + 融合演示。",
        "transcript": "",
        "audio_root": "teleantifraud",
        "audio_path": "audio/NEG-multi-agent-2/tts_fraud_00012/tts_fraud_00012.mp3",
    },
    {
        "id": "audio_tele_normal",
        "title": "正常音频样例",
        "scenario": "teleantifraud_normal",
        "label": "normal",
        "description": "来自 TeleAntiFraud 的正常通话音频，用于对照展示。",
        "transcript": "",
        "audio_root": "teleantifraud",
        "audio_path": "audio/POS-imitate-5/tts_test2094/tts_test2094.mp3",
    },
    {
        "id": "audio_pressure_high",
        "title": "高压力语音",
        "scenario": "csemotions_angry",
        "label": "high_pressure",
        "description": "来自 CSEMOTIONS 的强情绪语音，用于展示压力辅助分支。",
        "transcript": "",
        "audio_root": "csemotions",
        "audio_path": "angry/female001_csemotions_00000.wav",
    },
    {
        "id": "audio_pressure_normal",
        "title": "平稳语音",
        "scenario": "csemotions_neutral",
        "label": "normal_pressure",
        "description": "来自 CSEMOTIONS 的平稳语音，用于压力分支对照。",
        "transcript": "",
        "audio_root": "csemotions",
        "audio_path": "neutral/female001_csemotions_00210.wav",
    },
]


TEXT_SUGGESTIONS = {
    "high": "\u68c0\u6d4b\u5230\u591a\u4e2a\u9ad8\u98ce\u9669\u8bdd\u672f\u4fe1\u53f7\uff0c\u5efa\u8bae\u7acb\u5373\u6682\u505c\u8f6c\u8d26\u3001\u9a8c\u8bc1\u7801\u5171\u4eab\u6216\u654f\u611f\u4fe1\u606f\u63d0\u4ea4\u3002",
    "medium": "\u68c0\u6d4b\u5230\u90e8\u5206\u98ce\u9669\u8bdd\u672f\uff0c\u5efa\u8bae\u5148\u901a\u8fc7\u5b98\u65b9\u6e20\u9053\u6838\u5b9e\u5bf9\u65b9\u8eab\u4efd\u3002",
    "low": "\u68c0\u6d4b\u5230\u5c11\u91cf\u98ce\u9669\u8bcd\uff0c\u8bf7\u4fdd\u6301\u8b66\u60d5\u5e76\u907f\u514d\u900f\u9732\u654f\u611f\u4fe1\u606f\u3002",
    "normal": "\u6682\u672a\u68c0\u6d4b\u5230\u660e\u663e\u98ce\u9669\u8bdd\u672f\u3002",
}

AUDIO_NOTES = [
    "\u5f53\u524d\u7ed3\u679c\u6765\u81ea\u4f20\u7edf\u97f3\u9891\u7279\u5f81 baseline\u3002",
    "\u6a21\u578b\u4ec5\u5206\u6790\u97f3\u9891\u524d 5 \u79d2\uff0c\u9002\u5408\u4f5c\u4e3a\u65e9\u671f\u98ce\u9669\u8f85\u52a9\u4fe1\u53f7\u3002",
]

EMOTION_NOTES = [
    "\u60c5\u7eea\u548c\u538b\u529b\u4ec5\u4f5c\u4e3a\u8bed\u97f3\u72b6\u6001\u8f85\u52a9\u4fe1\u53f7\uff0c\u4e0d\u76f4\u63a5\u4ee3\u8868\u8bc8\u9a97\u6216\u533b\u5b66\u8bca\u65ad\u3002",
    "\u5f53\u524d\u6a21\u578b\u57fa\u4e8e CSEMOTIONS \u666e\u901a\u8bdd\u60c5\u7eea\u8bed\u97f3\u548c\u524d 5 \u79d2\u58f0\u5b66\u7279\u5f81\u8bad\u7ec3\u3002",
]

FUSION_NOTES = [
    "\u878d\u5408\u8bc4\u5206\u4f7f\u7528 CallGuard CAEF\uff0c\u4f1a\u6839\u636e\u6a21\u6001\u53ef\u9760\u6027\u548c\u5f53\u524d\u7f6e\u4fe1\u5ea6\u52a8\u6001\u5206\u914d\u6743\u91cd\u3002",
    "\u672a\u8f93\u5165\u6587\u672c\u65f6\uff0c\u7cfb\u7edf\u4f1a\u81ea\u52a8\u4f7f\u7528\u672c\u5730 Whisper \u6a21\u578b\u751f\u6210 transcript\u3002",
    "\u6587\u672c\u98ce\u9669\u7531 TF-IDF \u6587\u672c\u6a21\u578b\u548c\u5173\u952e\u8bcd\u89c4\u5219\u878d\u5408\u5f97\u5230\u3002",
    "\u5f53\u524d ASR \u4f18\u5148\u4f7f\u7528 faster-whisper-small\uff0c\u672a\u4e0b\u8f7d\u65f6\u81ea\u52a8\u56de\u9000\u5230 base/tiny\u3002",
]


app = FastAPI(
    title="CallGuard API",
    version="0.3.0",
    description="Backend service for call-risk analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db(DEFAULT_KEYWORD_GROUPS)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "callguard-api"}


@app.get("/api/demo/samples", response_model=list[DemoSampleResponse])
def list_demo_samples() -> list[DemoSampleResponse]:
    return [
        DemoSampleResponse(
            id=str(sample["id"]),
            title=str(sample["title"]),
            scenario=str(sample["scenario"]),
            label=str(sample["label"]),
            description=str(sample["description"]),
            transcript=str(sample.get("transcript", "")),
            has_audio=demo_audio_path(sample).exists() if sample.get("audio_path") else False,
            audio_file_name=Path(str(sample["audio_path"])).name if sample.get("audio_path") else None,
        )
        for sample in DEMO_SAMPLES
    ]


@app.get("/api/demo/samples/{sample_id}/audio")
def get_demo_sample_audio(sample_id: str) -> FileResponse:
    sample = find_demo_sample(sample_id)
    path = demo_audio_path(sample)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Demo audio not found: {sample_id}")

    media_type = "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/calls", response_model=CallListResponse)
def get_calls(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> CallListResponse:
    records, total = list_call_records(limit=limit, offset=offset)
    return CallListResponse(records=records, total=total, limit=limit, offset=offset)


@app.get("/api/calls/{record_id}", response_model=CallRecordDetail)
def get_call(record_id: int) -> CallRecordDetail:
    try:
        return CallRecordDetail(**get_call_record(record_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Call record not found: {record_id}") from exc


@app.delete("/api/calls/{record_id}", response_model=DeleteResponse)
def delete_call(record_id: int) -> DeleteResponse:
    deleted = delete_call_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Call record not found: {record_id}")
    return DeleteResponse(ok=True, message="Call record deleted.")


@app.get("/api/analytics/summary", response_model=AnalyticsSummaryResponse)
def get_analytics_summary() -> AnalyticsSummaryResponse:
    return AnalyticsSummaryResponse(**analytics_summary())


@app.get("/api/rules", response_model=list[RuleResponse])
def get_rules() -> list[RuleResponse]:
    return [RuleResponse(**rule) for rule in list_rules()]


@app.post("/api/rules", response_model=RuleResponse)
def post_rule(request: RuleCreateRequest) -> RuleResponse:
    try:
        return RuleResponse(
            **create_rule(
                group=request.group,
                keyword=request.keyword,
                weight=request.weight,
                enabled=request.enabled,
            )
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Rule already exists.") from exc


@app.put("/api/rules/{rule_id}", response_model=RuleResponse)
def put_rule(rule_id: int, request: RuleUpdateRequest) -> RuleResponse:
    try:
        values = request.model_dump(exclude_unset=True)
        return RuleResponse(**update_rule(rule_id, values))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}") from exc
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Rule already exists.") from exc


@app.patch("/api/rules/{rule_id}/toggle", response_model=RuleResponse)
def patch_rule_toggle(rule_id: int) -> RuleResponse:
    try:
        return RuleResponse(**toggle_rule(rule_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}") from exc


@app.delete("/api/rules/{rule_id}", response_model=DeleteResponse)
def delete_rule(rule_id: int) -> DeleteResponse:
    try:
        delete_custom_rule(rule_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DeleteResponse(ok=True, message="Rule deleted.")


@app.post("/api/rules/reset-defaults", response_model=list[RuleResponse])
def post_rules_reset_defaults() -> list[RuleResponse]:
    return [RuleResponse(**rule) for rule in reset_default_rules(DEFAULT_KEYWORD_GROUPS)]


@app.get("/api/model/audio/status")
def audio_model_status() -> dict[str, str | bool]:
    return {
        "model_path": str(AUDIO_MODEL_PATH),
        "available": AUDIO_MODEL_PATH.exists(),
        "model": "logistic_regression_audio_features",
    }


@app.get("/api/model/emotion/status")
def emotion_model_status() -> dict[str, str | bool]:
    return {
        "model_path": str(EMOTION_MODEL_PATH),
        "available": EMOTION_MODEL_PATH.exists(),
        "model": "csemotions_logistic_regression_audio_features",
        "role": "auxiliary_emotion_pressure_signal",
    }


@app.get("/api/model/asr/status")
def asr_model_status() -> dict[str, str | bool]:
    selected_path = get_asr_model_path()
    return {
        "model_path": str(selected_path),
        "available": selected_path.exists(),
        "model": selected_path.name,
        "small_available": ASR_SMALL_MODEL_PATH.exists(),
        "base_available": ASR_BASE_MODEL_PATH.exists(),
        "fallback_available": ASR_FALLBACK_MODEL_PATH.exists(),
    }


@app.get("/api/model/text/status")
def text_model_status() -> dict[str, str | bool]:
    return {
        "model_path": str(TEXT_MODEL_PATH),
        "available": TEXT_MODEL_PATH.exists(),
        "model": "tfidf_char_ngram_logistic_regression",
        "fallback": "keyword_rules",
    }


@app.get("/api/model/fusion/status")
def fusion_model_status() -> dict[str, str | bool | float]:
    if not FUSION_CONFIG_PATH.exists():
        return {
            "model_path": str(FUSION_CONFIG_PATH),
            "available": False,
            "model": "fixed_65_35",
        }
    config = load_adaptive_fusion_config()
    return {
        "model_path": str(FUSION_CONFIG_PATH),
        "available": True,
        "model": str(config["name"]),
        "version": str(config["version"]),
        "decision_threshold": float(config["decision_threshold"]),
    }


@app.post("/api/analyze/text", response_model=AnalyzeTextResponse)
def analyze_text(request: AnalyzeTextRequest) -> AnalyzeTextResponse:
    return analyze_text_content(request.text)


@app.post("/api/analyze/audio", response_model=AnalyzeAudioResponse)
async def analyze_audio(file: UploadFile = File(...)) -> AnalyzeAudioResponse:
    temp_path = await save_upload_to_temp(file)
    try:
        return analyze_audio_path(temp_path, file.filename or temp_path.name)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Audio analysis failed: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)
        await file.close()


@app.post("/api/analyze/emotion", response_model=AnalyzeEmotionResponse)
async def analyze_emotion(file: UploadFile = File(...)) -> AnalyzeEmotionResponse:
    temp_path = await save_upload_to_temp(file)
    try:
        return analyze_emotion_path(temp_path, file.filename or temp_path.name)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Emotion analysis failed: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)
        await file.close()


@app.post("/api/transcribe/audio", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(file: UploadFile = File(...)) -> TranscriptionResponse:
    temp_path = await save_upload_to_temp(file)
    try:
        return transcribe_audio_path(temp_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Audio transcription failed: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)
        await file.close()


@app.post("/api/analyze/call", response_model=AnalyzeCallResponse)
async def analyze_call(
    file: UploadFile | None = File(default=None),
    transcript: str = Form(default=""),
) -> AnalyzeCallResponse:
    manual_transcript = transcript.strip()
    display_transcript = manual_transcript
    model_transcript = manual_transcript
    file_name = file.filename if file is not None else None
    if file is None and not model_transcript:
        raise HTTPException(status_code=400, detail="Audio file or transcript is required.")

    audio_result: AnalyzeAudioResponse | None = None
    emotion_result: AnalyzeEmotionResponse | None = None
    asr_result: TranscriptionResponse | None = None

    if file is not None:
        temp_path = await save_upload_to_temp(file)
        try:
            audio_result = analyze_audio_path(temp_path, file.filename or temp_path.name)
            if EMOTION_MODEL_PATH.exists():
                emotion_result = analyze_emotion_path(temp_path, file.filename or temp_path.name)
            if not model_transcript:
                asr_result = transcribe_audio_path(temp_path)
                display_transcript = asr_result.text
                model_transcript = asr_result.raw_text or asr_result.text
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Call analysis failed: {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)
            await file.close()

    text_result = analyze_text_content(model_transcript) if model_transcript else None
    score, weights, fusion_method, decision_threshold, diagnostics = fuse_scores(
        audio_result,
        text_result,
    )
    level = risk_level_from_score(score)
    prediction = "fraud" if score >= decision_threshold else "normal"

    response = AnalyzeCallResponse(
        prediction=prediction,
        risk_score=round(score, 4),
        risk_level=level,
        transcript=display_transcript,
        transcript_source="manual" if manual_transcript else ("asr" if asr_result else "none"),
        asr=asr_result,
        audio=audio_result,
        emotion=emotion_result,
        text=text_result,
        fusion_weights=weights,
        fusion_method=fusion_method,
        decision_threshold=decision_threshold,
        fusion_diagnostics=diagnostics,
        suggestion=build_fusion_suggestion(level, audio_result, text_result),
        notes=FUSION_NOTES,
    )
    response.record_id = save_call_record_safely(
        response=response,
        input_type=input_type_for_analysis(file is not None, bool(manual_transcript)),
        file_name=file_name,
    )
    if response.record_id is None:
        response.notes = [*response.notes, "历史记录保存失败，但本次分析结果仍然可用。"]
    return response


def analyze_text_content(text: str) -> AnalyzeTextResponse:
    normalized_text = normalize_chinese_text(text)
    keyword_groups = load_keyword_groups_for_analysis()
    matches = find_risk_keywords(normalized_text, keyword_groups)
    factors = [RiskFactor(group=match.group, keyword=match.keyword) for match in matches]
    rule_score = score_text_rules(normalized_text, keyword_groups)
    model_score: float | None = None
    model_prediction: str | None = None
    evidence_terms: list[str] = []
    model_name = "keyword_rules"

    if TEXT_MODEL_PATH.exists():
        model_result = analyze_text_model(load_text_model(), text.strip())
        model_score = model_result.risk_score
        model_prediction = model_result.prediction
        evidence_terms = model_result.evidence_terms
        model_name = "tfidf_char_ngram_logistic_regression"
        model_rule_blend = 0.8 * model_score + 0.2 * rule_score
        score = round(max(model_rule_blend, 0.85 * rule_score), 4)
    else:
        score = rule_score

    level = risk_level_from_score(score)
    return AnalyzeTextResponse(
        risk_score=score,
        risk_level=level,
        factors=factors,
        suggestion=build_text_suggestion(level),
        model=model_name,
        model_score=model_score,
        model_prediction=model_prediction,
        rule_score=rule_score,
        evidence_terms=evidence_terms,
    )


def load_keyword_groups_for_analysis() -> dict[str, list[str]]:
    try:
        groups = get_enabled_keyword_groups()
        return groups or DEFAULT_KEYWORD_GROUPS
    except Exception:  # noqa: BLE001
        return DEFAULT_KEYWORD_GROUPS


def input_type_for_analysis(has_file: bool, has_manual_text: bool) -> str:
    if has_file and has_manual_text:
        return "audio_text"
    if has_file:
        return "audio"
    return "text"


def save_call_record_safely(
    response: AnalyzeCallResponse,
    input_type: str,
    file_name: str | None,
) -> int | None:
    try:
        factors = [
            {"group": factor.group, "keyword": factor.keyword}
            for factor in (response.text.factors if response.text else [])
        ]
        model_summary = {
            "fusion_method": response.fusion_method,
            "audio_model": response.audio.model if response.audio else None,
            "text_model": response.text.model if response.text else None,
            "emotion_model": response.emotion.model if response.emotion else None,
            "asr_model": response.asr.model if response.asr else None,
            "decision_threshold": response.decision_threshold,
        }
        return create_call_record(
            input_type=input_type,
            file_name=file_name,
            transcript=response.transcript,
            prediction=response.prediction,
            risk_score=response.risk_score,
            risk_level=response.risk_level,
            pressure_score=response.emotion.pressure_score if response.emotion else None,
            pressure_level=response.emotion.pressure_level if response.emotion else None,
            risk_factors=factors,
            model_summary=model_summary,
            analysis_result=response.model_dump(mode="json"),
        )
    except Exception:  # noqa: BLE001
        return None


async def save_upload_to_temp(file: UploadFile) -> Path:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    suffix = Path(file.filename).suffix or ".audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                temp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Audio file is too large.")
            temp_file.write(chunk)
    return temp_path


def analyze_audio_path(audio_path: Path, file_name: str) -> AnalyzeAudioResponse:
    model = load_audio_model()
    features = extract_audio_features(
        audio_path,
        sample_rate=DEFAULT_SAMPLE_RATE,
        duration=DEFAULT_DURATION_SECONDS,
    ).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    normal_probability = float(probabilities[0])
    fraud_probability = float(probabilities[1])
    prediction = "fraud" if fraud_probability >= 0.5 else "normal"
    level = risk_level_from_score(fraud_probability)

    return AnalyzeAudioResponse(
        file_name=file_name,
        prediction=prediction,
        risk_score=round(fraud_probability, 4),
        risk_level=level,
        probabilities={
            "normal": round(normal_probability, 4),
            "fraud": round(fraud_probability, 4),
        },
        model="logistic_regression_audio_features",
        feature_window_seconds=DEFAULT_DURATION_SECONDS,
        suggestion=build_audio_suggestion(level, prediction),
        notes=AUDIO_NOTES,
    )


def analyze_emotion_path(audio_path: Path, file_name: str) -> AnalyzeEmotionResponse:
    features = extract_audio_features(
        audio_path,
        sample_rate=DEFAULT_SAMPLE_RATE,
        duration=DEFAULT_DURATION_SECONDS,
    )
    prediction = analyze_emotion_model(load_emotion_model(), features)

    return AnalyzeEmotionResponse(
        file_name=file_name,
        primary_emotion=prediction.primary_emotion,
        emotion_confidence=round(prediction.emotion_confidence, 4),
        pressure_score=round(prediction.pressure_score, 4),
        pressure_level=prediction.pressure_level,
        probabilities={
            label: round(value, 4) for label, value in prediction.probabilities.items()
        },
        pressure_probabilities={
            label: round(value, 4)
            for label, value in prediction.pressure_probabilities.items()
        },
        pressure_drivers=[
            EmotionPressureDriver(
                emotion=str(driver["emotion"]),
                probability=round(float(driver["probability"]), 4),
                pressure_weight=round(float(driver["pressure_weight"]), 4),
                contribution=round(float(driver["contribution"]), 4),
            )
            for driver in prediction.pressure_drivers
        ],
        model="csemotions_logistic_regression_audio_features",
        feature_window_seconds=DEFAULT_DURATION_SECONDS,
        suggestion=build_emotion_suggestion(prediction.pressure_level),
        notes=EMOTION_NOTES,
    )


def transcribe_audio_path(audio_path: Path) -> TranscriptionResponse:
    result = transcribe_audio(load_asr_model(), audio_path)
    display_text = simplify_chinese_text(result.text)
    return TranscriptionResponse(
        text=display_text,
        raw_text=result.text if result.text != display_text else None,
        source="asr",
        language=result.language,
        language_probability=round(result.language_probability, 4),
        duration_seconds=round(result.duration_seconds, 2),
        model=get_asr_model_path().name,
        segments=[
            TranscriptionSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=simplify_chinese_text(str(segment["text"])),
            )
            for segment in result.segments
        ],
    )


@lru_cache(maxsize=1)
def load_audio_model():
    if not AUDIO_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Audio model not found: {AUDIO_MODEL_PATH}")
    return joblib.load(AUDIO_MODEL_PATH)


@lru_cache(maxsize=1)
def load_emotion_model():
    if not EMOTION_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Emotion model not found: {EMOTION_MODEL_PATH}")
    return load_emotion_model_file(EMOTION_MODEL_PATH)


@lru_cache(maxsize=1)
def load_asr_model():
    return load_whisper_model(get_asr_model_path())


@lru_cache(maxsize=1)
def load_text_model():
    if not TEXT_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Text model not found: {TEXT_MODEL_PATH}")
    return load_text_model_file(TEXT_MODEL_PATH)


@lru_cache(maxsize=1)
def load_adaptive_fusion_config():
    return load_fusion_config(FUSION_CONFIG_PATH)


def get_asr_model_path() -> Path:
    if ASR_MODEL_PATH and ASR_MODEL_PATH.exists():
        return ASR_MODEL_PATH
    if ASR_SMALL_MODEL_PATH.exists():
        return ASR_SMALL_MODEL_PATH
    if ASR_BASE_MODEL_PATH.exists():
        return ASR_BASE_MODEL_PATH
    if ASR_FALLBACK_MODEL_PATH.exists():
        return ASR_FALLBACK_MODEL_PATH
    raise HTTPException(
        status_code=503,
        detail="ASR model not found: faster-whisper-small/base/tiny",
    )


def find_demo_sample(sample_id: str) -> dict[str, str]:
    for sample in DEMO_SAMPLES:
        if sample["id"] == sample_id:
            return sample
    raise HTTPException(status_code=404, detail=f"Demo sample not found: {sample_id}")


def demo_audio_path(sample: dict[str, str]) -> Path:
    audio_path = sample.get("audio_path")
    audio_root_key = sample.get("audio_root")
    if not audio_path or not audio_root_key:
        return Path("__missing_demo_audio__")
    root = AUDIO_ROOTS.get(audio_root_key)
    if root is None:
        return Path("__missing_demo_audio__")
    return root / audio_path


def risk_level_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.3:
        return "low"
    return "normal"


def fuse_scores(
    audio_result: AnalyzeAudioResponse | None,
    text_result: AnalyzeTextResponse | None,
) -> tuple[
    float,
    dict[str, float],
    str,
    float,
    FusionDiagnostics | None,
]:
    if audio_result and text_result:
        if FUSION_CONFIG_PATH.exists() and text_result.model_score is not None:
            config = load_adaptive_fusion_config()
            result = adaptive_fusion(
                audio_result.risk_score,
                text_result.model_score,
                text_result.rule_score,
                config,
            )
            return (
                result.risk_score,
                {"audio": result.audio_weight, "text": result.text_weight},
                str(config["name"]),
                float(config["decision_threshold"]),
                FusionDiagnostics(
                    audio_confidence=result.audio_confidence,
                    text_confidence=result.text_confidence,
                    agreement=result.agreement,
                    rule_adjustment=result.rule_adjustment,
                ),
            )
        return (
            0.65 * audio_result.risk_score + 0.35 * text_result.risk_score,
            {"audio": 0.65, "text": 0.35},
            "fixed_65_35",
            0.5,
            None,
        )
    if audio_result:
        return (
            audio_result.risk_score,
            {"audio": 1.0, "text": 0.0},
            "audio_only",
            0.5,
            None,
        )
    if text_result:
        return (
            text_result.risk_score,
            {"audio": 0.0, "text": 1.0},
            "text_only",
            0.5,
            None,
        )
    return 0.0, {"audio": 0.0, "text": 0.0}, "none", 0.5, None


def build_text_suggestion(level: str) -> str:
    return TEXT_SUGGESTIONS[level]


def build_audio_suggestion(level: str, prediction: str) -> str:
    if level == "high":
        return "\u97f3\u9891\u6a21\u578b\u7ed9\u51fa\u9ad8\u98ce\u9669\u9884\u8b66\uff0c\u5efa\u8bae\u6682\u505c\u901a\u8bdd\u4e2d\u7684\u8f6c\u8d26\u3001\u9a8c\u8bc1\u7801\u6216\u8d26\u6237\u64cd\u4f5c\u3002"
    if level == "medium":
        return "\u97f3\u9891\u6a21\u578b\u68c0\u6d4b\u5230\u4e2d\u7b49\u98ce\u9669\u4fe1\u53f7\uff0c\u5efa\u8bae\u7ed3\u5408\u901a\u8bdd\u5185\u5bb9\u7ee7\u7eed\u6838\u9a8c\u3002"
    if level == "low":
        return "\u97f3\u9891\u6a21\u578b\u68c0\u6d4b\u5230\u8f7b\u5fae\u4fe1\u53f7\uff0c\u5f53\u524d\u66f4\u9002\u5408\u4f5c\u4e3a\u8f85\u52a9\u63d0\u9192\u3002"
    if prediction == "fraud":
        return "\u6a21\u578b\u9884\u6d4b\u4e3a\u98ce\u9669\u901a\u8bdd\uff0c\u4f46\u7f6e\u4fe1\u5ea6\u8f83\u4f4e\uff0c\u5efa\u8bae\u7ed3\u5408\u6587\u672c\u8bdd\u672f\u5224\u65ad\u3002"
    return "\u97f3\u9891\u6a21\u578b\u6682\u672a\u68c0\u6d4b\u5230\u660e\u663e\u8bc8\u9a97\u98ce\u9669\u3002"


def build_emotion_suggestion(level: str) -> str:
    if level == "high":
        return "\u8bed\u97f3\u4e2d\u51fa\u73b0\u9ad8\u538b\u6216\u5f3a\u60c5\u7eea\u7ebf\u7d22\uff0c\u5efa\u8bae\u653e\u6162\u64cd\u4f5c\u8282\u594f\uff0c\u7559\u51fa\u989d\u5916\u6838\u9a8c\u65f6\u95f4\u3002"
    if level == "medium":
        return "\u8bed\u97f3\u72b6\u6001\u663e\u793a\u4e2d\u7b49\u538b\u529b\u4fe1\u53f7\uff0c\u5efa\u8bae\u7ed3\u5408\u901a\u8bdd\u8bdd\u672f\u548c\u80cc\u666f\u7ee7\u7eed\u5224\u65ad\u3002"
    if level == "low":
        return "\u8bed\u97f3\u4e2d\u6709\u8f7b\u5fae\u538b\u529b\u7ebf\u7d22\uff0c\u5f53\u524d\u66f4\u9002\u5408\u4f5c\u4e3a\u8f85\u52a9\u53c2\u8003\u3002"
    return "\u8bed\u97f3\u72b6\u6001\u76f8\u5bf9\u5e73\u7a33\uff0c\u672a\u68c0\u6d4b\u5230\u660e\u663e\u9ad8\u538b\u7ebf\u7d22\u3002"


def build_fusion_suggestion(
    level: str,
    audio_result: AnalyzeAudioResponse | None,
    text_result: AnalyzeTextResponse | None,
) -> str:
    has_text_risk = bool(text_result and text_result.factors)
    has_audio_risk = bool(audio_result and audio_result.prediction == "fraud")

    if level == "high":
        return "\u878d\u5408\u6a21\u578b\u7ed9\u51fa\u9ad8\u98ce\u9669\u9884\u8b66\uff0c\u5efa\u8bae\u7acb\u5373\u6682\u505c\u8f6c\u8d26\u3001\u9a8c\u8bc1\u7801\u5171\u4eab\u6216\u8d26\u6237\u64cd\u4f5c\u3002"
    if level == "medium" and has_audio_risk and has_text_risk:
        return "\u97f3\u9891\u548c\u6587\u672c\u5747\u51fa\u73b0\u98ce\u9669\u4fe1\u53f7\uff0c\u5efa\u8bae\u901a\u8fc7\u5b98\u65b9\u6e20\u9053\u6838\u5b9e\u5bf9\u65b9\u8eab\u4efd\u3002"
    if level == "medium":
        return "\u878d\u5408\u6a21\u578b\u68c0\u6d4b\u5230\u4e2d\u7b49\u98ce\u9669\uff0c\u5efa\u8bae\u7ed3\u5408\u901a\u8bdd\u80cc\u666f\u7ee7\u7eed\u6838\u9a8c\u3002"
    if level == "low":
        return "\u68c0\u6d4b\u5230\u8f7b\u5fae\u98ce\u9669\u4fe1\u53f7\uff0c\u8bf7\u4fdd\u6301\u8b66\u60d5\uff0c\u907f\u514d\u900f\u9732\u654f\u611f\u4fe1\u606f\u3002"
    return "\u878d\u5408\u6a21\u578b\u6682\u672a\u68c0\u6d4b\u5230\u660e\u663e\u98ce\u9669\u3002"
