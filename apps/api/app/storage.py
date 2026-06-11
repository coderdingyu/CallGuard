from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = Path(
    os.environ.get(
        "CALLGUARD_DB_PATH",
        PROJECT_ROOT / "data" / "local" / "callguard.db",
    )
)


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(default_keyword_groups: dict[str, list[str]]) -> None:
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS call_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_type TEXT NOT NULL,
                file_name TEXT,
                transcript TEXT NOT NULL,
                transcript_preview TEXT NOT NULL,
                prediction TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                pressure_score REAL,
                pressure_level TEXT,
                risk_factors TEXT NOT NULL,
                model_summary TEXT NOT NULL,
                analysis_result TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS text_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_name TEXT NOT NULL,
                keyword TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                enabled INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL DEFAULT 'custom',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(group_name, keyword)
            )
            """
        )
        count = connection.execute("SELECT COUNT(*) FROM text_rules").fetchone()[0]
        if count == 0:
            insert_default_rules(connection, default_keyword_groups)


def insert_default_rules(
    connection: sqlite3.Connection,
    default_keyword_groups: dict[str, list[str]],
) -> None:
    now = utc_now()
    rows = [
        (group_name, keyword, 1.0, 1, "default", now, now)
        for group_name, keywords in default_keyword_groups.items()
        for keyword in keywords
    ]
    connection.executemany(
        """
        INSERT OR IGNORE INTO text_rules
        (group_name, keyword, weight, enabled, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def reset_default_rules(default_keyword_groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    with connect() as connection:
        connection.execute("DELETE FROM text_rules")
        insert_default_rules(connection, default_keyword_groups)
    return list_rules()


def get_enabled_keyword_groups() -> dict[str, list[str]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT group_name, keyword
            FROM text_rules
            WHERE enabled = 1
            ORDER BY group_name, keyword
            """
        ).fetchall()

    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(str(row["group_name"]), []).append(str(row["keyword"]))
    return groups


def list_rules() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, group_name, keyword, weight, enabled, source, created_at, updated_at
            FROM text_rules
            ORDER BY source DESC, group_name, keyword
            """
        ).fetchall()
    return [rule_from_row(row) for row in rows]


def create_rule(group: str, keyword: str, weight: float = 1.0, enabled: bool = True) -> dict[str, Any]:
    now = utc_now()
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO text_rules
            (group_name, keyword, weight, enabled, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'custom', ?, ?)
            """,
            (group.strip(), keyword.strip(), weight, int(enabled), now, now),
        )
        rule_id = int(cursor.lastrowid)
    return get_rule(rule_id)


def get_rule(rule_id: int) -> dict[str, Any]:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT id, group_name, keyword, weight, enabled, source, created_at, updated_at
            FROM text_rules
            WHERE id = ?
            """,
            (rule_id,),
        ).fetchone()
    if row is None:
        raise KeyError(rule_id)
    return rule_from_row(row)


def update_rule(rule_id: int, values: dict[str, Any]) -> dict[str, Any]:
    current = get_rule(rule_id)
    group = str(values.get("group", current["group"]))
    keyword = str(values.get("keyword", current["keyword"]))
    weight = float(values.get("weight", current["weight"]))
    enabled = bool(values.get("enabled", current["enabled"]))
    now = utc_now()
    with connect() as connection:
        connection.execute(
            """
            UPDATE text_rules
            SET group_name = ?, keyword = ?, weight = ?, enabled = ?, updated_at = ?
            WHERE id = ?
            """,
            (group.strip(), keyword.strip(), weight, int(enabled), now, rule_id),
        )
    return get_rule(rule_id)


def toggle_rule(rule_id: int) -> dict[str, Any]:
    current = get_rule(rule_id)
    with connect() as connection:
        connection.execute(
            "UPDATE text_rules SET enabled = ?, updated_at = ? WHERE id = ?",
            (0 if current["enabled"] else 1, utc_now(), rule_id),
        )
    return get_rule(rule_id)


def delete_custom_rule(rule_id: int) -> None:
    current = get_rule(rule_id)
    if current["source"] != "custom":
        raise PermissionError("Default rules can be disabled but not deleted.")
    with connect() as connection:
        connection.execute("DELETE FROM text_rules WHERE id = ?", (rule_id,))


def create_call_record(
    *,
    input_type: str,
    file_name: str | None,
    transcript: str,
    prediction: str,
    risk_score: float,
    risk_level: str,
    pressure_score: float | None,
    pressure_level: str | None,
    risk_factors: list[dict[str, Any]],
    model_summary: dict[str, Any],
    analysis_result: dict[str, Any],
) -> int:
    created_at = utc_now()
    preview = transcript.strip().replace("\n", " ")[:120]
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO call_records
            (
                created_at, input_type, file_name, transcript, transcript_preview,
                prediction, risk_score, risk_level, pressure_score, pressure_level,
                risk_factors, model_summary, analysis_result
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                input_type,
                file_name,
                transcript,
                preview,
                prediction,
                risk_score,
                risk_level,
                pressure_score,
                pressure_level,
                json.dumps(risk_factors, ensure_ascii=False),
                json.dumps(model_summary, ensure_ascii=False),
                json.dumps(analysis_result, ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid)


def list_call_records(limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
    with connect() as connection:
        total = int(connection.execute("SELECT COUNT(*) FROM call_records").fetchone()[0])
        rows = connection.execute(
            """
            SELECT id, created_at, input_type, file_name, prediction, risk_score,
                   risk_level, pressure_score, pressure_level, transcript_preview,
                   risk_factors, model_summary
            FROM call_records
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [call_summary_from_row(row) for row in rows], total


def get_call_record(record_id: int) -> dict[str, Any]:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM call_records
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()
    if row is None:
        raise KeyError(record_id)
    summary = call_summary_from_row(row)
    summary["transcript"] = str(row["transcript"])
    summary["analysis_result"] = json.loads(str(row["analysis_result"]))
    return summary


def delete_call_record(record_id: int) -> bool:
    with connect() as connection:
        cursor = connection.execute("DELETE FROM call_records WHERE id = ?", (record_id,))
        return cursor.rowcount > 0


def analytics_summary() -> dict[str, Any]:
    records, _ = list_call_records(limit=8, offset=0)
    with connect() as connection:
        totals = connection.execute(
            """
            SELECT
              COUNT(*) AS total_calls,
              SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) AS high_risk_calls,
              AVG(risk_score) AS average_risk_score
            FROM call_records
            """
        ).fetchone()
        level_rows = connection.execute(
            """
            SELECT risk_level, COUNT(*) AS count
            FROM call_records
            GROUP BY risk_level
            """
        ).fetchall()
        prediction_rows = connection.execute(
            """
            SELECT prediction, COUNT(*) AS count
            FROM call_records
            GROUP BY prediction
            """
        ).fetchall()
        factor_rows = connection.execute("SELECT risk_factors FROM call_records").fetchall()

    top_keywords: dict[str, int] = {}
    for row in factor_rows:
        for factor in json.loads(str(row["risk_factors"])):
            keyword = str(factor.get("keyword", "")).strip()
            if keyword:
                top_keywords[keyword] = top_keywords.get(keyword, 0) + 1

    return {
        "total_calls": int(totals["total_calls"] or 0),
        "high_risk_calls": int(totals["high_risk_calls"] or 0),
        "average_risk_score": round(float(totals["average_risk_score"] or 0.0), 4),
        "risk_level_counts": {str(row["risk_level"]): int(row["count"]) for row in level_rows},
        "prediction_counts": {str(row["prediction"]): int(row["count"]) for row in prediction_rows},
        "top_keywords": [
            {"keyword": keyword, "count": count}
            for keyword, count in sorted(top_keywords.items(), key=lambda item: item[1], reverse=True)[:10]
        ],
        "recent_calls": records,
    }


def rule_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "group": str(row["group_name"]),
        "keyword": str(row["keyword"]),
        "weight": float(row["weight"]),
        "enabled": bool(row["enabled"]),
        "source": str(row["source"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def call_summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "created_at": str(row["created_at"]),
        "input_type": str(row["input_type"]),
        "file_name": row["file_name"],
        "prediction": str(row["prediction"]),
        "risk_score": float(row["risk_score"]),
        "risk_level": str(row["risk_level"]),
        "pressure_score": None if row["pressure_score"] is None else float(row["pressure_score"]),
        "pressure_level": row["pressure_level"],
        "transcript_preview": str(row["transcript_preview"]),
        "risk_factors": json.loads(str(row["risk_factors"])),
        "model_summary": json.loads(str(row["model_summary"])),
    }
