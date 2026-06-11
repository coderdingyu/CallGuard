from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = Path(tempfile.mkdtemp(prefix="callguard-api-test-"))
os.environ["CALLGUARD_DB_PATH"] = str(TEMP_DIR / "callguard-test.db")
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from app.main import app  # noqa: E402


def assert_ok(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert_ok(health.status_code == 200, "health endpoint failed")

        reset = client.post("/api/rules/reset-defaults")
        assert_ok(reset.status_code == 200, "rule reset failed")
        assert_ok(len(reset.json()) > 0, "default rules were not initialized")

        analysis = client.post(
            "/api/analyze/call",
            data={
                "transcript": "我是公安机关工作人员，现在必须马上把资金转到安全账户。"
            },
        )
        assert_ok(analysis.status_code == 200, f"call analysis failed: {analysis.text}")
        payload = analysis.json()
        assert_ok(payload["record_id"] is not None, "analysis was not saved to history")
        assert_ok(payload["risk_level"] in {"medium", "high"}, "expected risk analysis result")

        calls = client.get("/api/calls")
        assert_ok(calls.status_code == 200, "call list failed")
        assert_ok(calls.json()["total"] >= 1, "history record was not listed")

        summary = client.get("/api/analytics/summary")
        assert_ok(summary.status_code == 200, "analytics summary failed")
        assert_ok(summary.json()["total_calls"] >= 1, "analytics did not include history")

        created = client.post(
            "/api/rules",
            json={"group": "test_group", "keyword": "黑金账户", "weight": 1.0, "enabled": True},
        )
        assert_ok(created.status_code == 200, f"create rule failed: {created.text}")
        rule_id = created.json()["id"]

        text_hit = client.post("/api/analyze/text", json={"text": "请把钱转到黑金账户。"})
        assert_ok(text_hit.status_code == 200, "text analysis with custom rule failed")
        hit_keywords = {factor["keyword"] for factor in text_hit.json()["factors"]}
        assert_ok("黑金账户" in hit_keywords, "custom rule did not affect text analysis")

        toggled = client.patch(f"/api/rules/{rule_id}/toggle")
        assert_ok(toggled.status_code == 200, "toggle rule failed")
        assert_ok(toggled.json()["enabled"] is False, "rule was not disabled")

        text_miss = client.post("/api/analyze/text", json={"text": "请把钱转到黑金账户。"})
        miss_keywords = {factor["keyword"] for factor in text_miss.json()["factors"]}
        assert_ok("黑金账户" not in miss_keywords, "disabled rule still affected analysis")

    print("api smoke tests passed")


if __name__ == "__main__":
    main()
