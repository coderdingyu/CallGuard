from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


DEFAULT_ASR_LANGUAGE = "zh"
DEFAULT_ASR_BEAM_SIZE = 5
DEFAULT_ASR_PROMPT = "请使用简体中文完整转写普通话通话内容。保留必要的英文单词和数字。"


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration_seconds: float
    segments: list[dict[str, float | str]]


def load_whisper_model(model_path: str | Path) -> WhisperModel:
    return WhisperModel(str(model_path), device="cpu", compute_type="int8")


def transcribe_audio(
    model: WhisperModel,
    audio_path: str | Path,
    language: str = DEFAULT_ASR_LANGUAGE,
    beam_size: int = DEFAULT_ASR_BEAM_SIZE,
) -> TranscriptionResult:
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        initial_prompt=DEFAULT_ASR_PROMPT,
        condition_on_previous_text=True,
        vad_filter=False,
    )
    segments = [
        {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        }
        for segment in segments_iter
    ]
    text = "".join(segment["text"] for segment in segments).strip()
    return TranscriptionResult(
        text=text,
        language=info.language,
        language_probability=float(info.language_probability),
        duration_seconds=float(info.duration),
        segments=segments,
    )
