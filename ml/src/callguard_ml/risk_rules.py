from __future__ import annotations

from dataclasses import dataclass


DEFAULT_KEYWORD_GROUPS: dict[str, list[str]] = {
    "money_transfer": [
        "\u8f6c\u8d26",
        "\u8f6c\u94b1",
        "\u6c47\u6b3e",
        "\u4ed8\u6b3e",
        "\u4fdd\u8bc1\u91d1",
        "\u5b89\u5168\u8d26\u6237",
        "\u9000\u6b3e",
        "\u8d44\u91d1",
        "\u62ff\u94b1",
    ],
    "identity_impersonation": [
        "\u516c\u5b89",
        "\u8b66\u5bdf",
        "\u6cd5\u9662",
        "\u68c0\u5bdf\u9662",
        "\u5ba2\u670d",
        "\u94f6\u884c",
        "\u652f\u4ed8\u5b9d",
        "\u5fae\u4fe1",
        "\u793e\u4fdd",
        "\u5feb\u9012",
    ],
    "urgency_pressure": [
        "\u9a6c\u4e0a",
        "\u7acb\u5373",
        "\u73b0\u5728",
        "\u4e0d\u80fd\u544a\u8bc9\u522b\u4eba",
        "\u4fdd\u5bc6",
        "\u903e\u671f",
        "\u6d89\u5acc",
        "\u51bb\u7ed3",
        "\u5904\u7406",
        "\u51fa\u4e8b",
        "\u7d27\u6025",
        "\u7ed1\u67b6",
        "\u63a7\u5236",
        "\u653e\u4eba",
    ],
    "sensitive_info": [
        "\u9a8c\u8bc1\u7801",
        "\u5bc6\u7801",
        "\u8eab\u4efd\u8bc1",
        "\u94f6\u884c\u5361",
        "\u8d26\u53f7",
        "\u5361\u53f7",
        "\u52a8\u6001\u7801",
        "\u4eba\u8138",
    ],
    "family_emergency": [
        "\u4f60\u7684\u5b69\u5b50",
        "\u60a8\u7684\u5b69\u5b50",
        "\u4f60\u7684\u513f\u5b50",
        "\u60a8\u7684\u513f\u5b50",
        "\u4f60\u7684\u5973\u513f",
        "\u60a8\u7684\u5973\u513f",
        "\u6211\u662f\u4ed6\u7684\u670b\u53cb",
        "\u4e0d\u7136\u5c31",
    ],
}


@dataclass(frozen=True)
class RuleMatch:
    group: str
    keyword: str


def find_risk_keywords(
    text: str,
    keyword_groups: dict[str, list[str]] | None = None,
) -> list[RuleMatch]:
    groups = keyword_groups or DEFAULT_KEYWORD_GROUPS
    matches: list[RuleMatch] = []
    normalized_text = text.strip()
    for group, keywords in groups.items():
        for keyword in keywords:
            if keyword in normalized_text:
                matches.append(RuleMatch(group=group, keyword=keyword))
    return matches


def score_text_rules(text: str) -> float:
    matches = find_risk_keywords(text)
    if not matches:
        return 0.0

    matched_groups = {match.group for match in matches}
    group_score = min(len(matched_groups) / len(DEFAULT_KEYWORD_GROUPS), 1.0)
    keyword_score = min(len(matches) / 8.0, 1.0)
    return round(0.65 * group_score + 0.35 * keyword_score, 4)
