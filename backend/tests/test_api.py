"""HTTP layer: SSE framing, sessions, rate limiting, fault toggles."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.main import create_app


def parse_sse(body: str) -> list[tuple[str, dict]]:
    out = []
    for frame in body.split("\n\n"):
        ev, data = None, None
        for line in frame.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if ev:
            out.append((ev, data))
    return out


def client(**overrides) -> TestClient:
    s = Settings(_env_file=None, llm_provider="fake", **overrides)
    return TestClient(create_app(s, build_container(s)))


def test_health_config_and_sse_chat_flow():
    with client() as c:
        h = c.get("/api/health").json()
        assert h["status"] == "ok" and h["kb"]["chunks"] > 50
        cfg = c.get("/api/config").json()
        assert len(cfg["demo_accounts"]) == 7 and cfg["voice_enabled"]

        sid = c.post("/api/sessions").json()["session_id"]
        r = c.post("/api/chat", json={"session_id": sid, "message": "how do I fix error E-1042?"})
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.text)
        kinds = [e for e, _ in events]
        assert kinds[0] == "session" and kinds[-1] == "done"
        assert "step" in kinds and "token" in kinds and "trace" in kinds
        msg = next(d for e, d in events if e == "message")
        assert msg["grounding"]["status"] == "grounded"

        sess = c.get(f"/api/sessions/{sid}").json()
        assert [m["role"] for m in sess["messages"]] == ["user", "assistant"]
        assert len(c.get(f"/api/sessions/{sid}/traces").json()["traces"]) == 1


def test_rate_limit_returns_429_with_retry_after():
    with client(rate_limit_per_minute=2, rate_limit_burst=2) as c:
        codes = [c.post("/api/chat", json={"message": "hi"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = c.post("/api/chat", json={"message": "hi"})
        assert int(r.headers["retry-after"]) >= 1 and r.json()["error"] == "rate_limited"


def test_input_validation_and_unknown_session():
    with client() as c:
        assert c.post("/api/chat", json={"message": ""}).status_code == 422
        assert c.post("/api/chat", json={"message": "x" * 2001}).status_code == 422
        assert c.get("/api/sessions/nope").status_code == 404


def test_fault_toggles_and_kill_switch():
    with client() as c:
        sid = c.post("/api/sessions").json()["session_id"]
        f = c.post(f"/api/sessions/{sid}/faults", json={"llm_rate_limit": 1, "crm_outage": True}).json()
        assert f["llm_rate_limit"] == 1 and f["crm_outage"] is True
    with client(enable_fault_injection=False) as c:
        sid = c.post("/api/sessions").json()["session_id"]
        assert c.post(f"/api/sessions/{sid}/faults", json={"crm_outage": True}).status_code == 403


def test_voice_socket_rejects_foreign_origin():
    with client(cors_origins="https://good.example") as c:
        sid = c.post("/api/sessions").json()["session_id"]
        try:
            with c.websocket_connect(f"/ws/voice?session_id={sid}", headers={"origin": "https://evil.example"}) as ws:
                ws.receive()
            raise AssertionError("connection should have been refused")
        except Exception as exc:  # starlette raises WebSocketDisconnect with our close code
            assert "4403" in repr(exc) or "WebSocketDisconnect" in type(exc).__name__
