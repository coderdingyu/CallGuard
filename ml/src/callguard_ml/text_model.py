from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class TextModelResult:
    prediction: str
    risk_score: float
    evidence_terms: list[str]


def load_text_model(model_path: str | Path) -> Any:
    return joblib.load(Path(model_path))


def analyze_text_model(model: Any, text: str, top_k: int = 8) -> TextModelResult:
    probabilities = model.predict_proba([text])[0]
    fraud_probability = float(probabilities[1])
    prediction = "fraud" if fraud_probability >= 0.5 else "normal"
    return TextModelResult(
        prediction=prediction,
        risk_score=round(fraud_probability, 4),
        evidence_terms=extract_positive_terms(model, text, top_k=top_k),
    )


def extract_positive_terms(model: Any, text: str, top_k: int = 8) -> list[str]:
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    vector = vectorizer.transform([text]).tocoo()

    candidates: list[tuple[float, str]] = []
    for index, value in zip(vector.col, vector.data, strict=False):
        contribution = float(coefficients[index] * value)
        if contribution <= 0:
            continue
        term = str(feature_names[index]).strip()
        if not is_display_term(term):
            continue
        candidates.append((contribution, term))

    candidates.sort(reverse=True, key=lambda item: item[0])
    preferred = unique_terms([term for _, term in candidates if len(term) >= 2], top_k)
    if len(preferred) >= top_k:
        return preferred[:top_k]
    fallback = unique_terms([term for _, term in candidates], top_k)
    return unique_terms(preferred + fallback, top_k)


def unique_terms(terms: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        result.append(term)
        if len(result) >= limit:
            break
    return result


def is_display_term(term: str) -> bool:
    if not term or term.isspace():
        return False
    if term.lower() in {"ap", "pp"}:
        return False
    if term.isascii() and len(term) < 3:
        return False
    return True
