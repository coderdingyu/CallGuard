from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from opencc import OpenCC


@lru_cache(maxsize=1)
def get_converter() -> OpenCC:
    return OpenCC("t2s")


def simplify_chinese_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    simplified = get_converter().convert(normalized)
    return re.sub(r"\s+", " ", simplified).strip()


def normalize_chinese_text(text: str) -> str:
    normalized = simplify_chinese_text(text)
    normalized = normalized.lower()
    return normalized.strip()
