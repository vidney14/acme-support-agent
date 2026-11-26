"""Voice bridge end-to-end against a fake Gemini Live session (real SDK message types)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from google.genai import types

from app.config import Settings
from app.container import build_container
from app.main import create_app
from app.voice.fake_live import FakeLiveSession, content, make_connect, msg, tone_chunks


def _script():
    tool_call = msg(tool_call=types.LiveServerToolCall(function_calls=[
        types.FunctionCall(id="fc1", name="search_knowledge_base", args={"query": "sync error E-1042"})]))
    return [
        # 1) greeting, triggered by the bridge's greet client_content
        [content(output_transcription=types.Transcription(text="Hi, this is Acme Assist. ")), *tone_chunks(0.2),
         content(turn_complete=True)],
        # 2) user speaks -> tool call -> answer -> customer interrupts -> agent recovers
        [content(input_transcription=types.Transcription(text="My sync shows error E-1042")),
         tool_call, "AWAIT_TOOL",
         *tone_chunks(0.2), content(output_transcription=types.Transcription(text="That error means the file path is ")),
         content(interrupted=True),
         content(input_transcription=types.Transcription(text="Wait, how do I fix it?")),
         *tone_chunks(0.2), content(output_transcription=types.Transcription(text="Shorten the folder names.")),
         msg(usage_metadata=types.UsageMetadata(
             prompt_token_count=900, response_token_count=120,
             prompt_tokens_details=[types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=600),
                                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=300)],
             response_tokens_details=[types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=120)])),
         content(turn_complete=True)],
    ]


def _collect(ws, until: str, limit: int = 200):
    events, audio_frames = [], 0
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes") is not None:
            audio_frames += 1
            continue
        ev = json.loads(m["text"])
        events.append(ev)
        if ev["type"] == until:
            return events, audio_frames
    raise AssertionError(f"never saw {until}: {[e['type'] for e in events]}")


def test_voice_session_tools_interruption_and_shared_memory():
    fake = FakeLiveSession(_script())
    settings = Settings(_env_file=None, llm_provider="fake")
    container = build_container(settings)
    container.live_connect = make_connect(lambda: fake)
    app = create_app(settings, container)

    with TestClient(app) as client:
        sid = client.post("/api/sessions").json()["session_id"]
        with client.websocket_connect(f"/ws/voice?session_id={sid}") as ws:
            evs, _ = _collect(ws, "ready")
            assert evs[0]["type"] == "session"
            evs, frames = _collect(ws, "turn_complete")           # greeting
            assert frames >= 1
            assert any(e["type"] == "transcript" and e["role"] == "assistant" for e in evs)

            ws.send_bytes(b"\x00\x00" * 320)                      # user starts speaking
            evs, frames = _collect(ws, "interrupted")
            steps = [e for e in evs if e["type"] == "step"]
            assert [s["status"] for s in steps] == ["started", "completed"]
            assert steps[0]["tool"] == "search_knowledge_base"
            assert any(e["type"] == "guardrail" and e["decision"] == "answer" for e in evs)
            evs, _ = _collect(ws, "turn_complete")               # interrupted turn closed out
            assert evs[-1]["interrupted"] is True
            evs, _ = _collect(ws, "turn_complete")               # recovery turn
            ws.send_text(json.dumps({"type": "stop"}))

        # Tool result went back to the model as structured data
        assert fake.tool_responses[0].name == "search_knowledge_base"
        assert fake.tool_responses[0].response["ok"] is True
        # Voice config: blocking tools, transcription on, barge-in enabled
        decl = fake.config.tools[0].function_declarations
        assert all(d.behavior == types.Behavior.BLOCKING for d in decl)
        assert fake.config.input_audio_transcription is not None

        # Shared memory: voice turns are in the same session transcript the chat agent reads
        data = client.get(f"/api/sessions/{sid}").json()
        voice_msgs = [m for m in data["messages"] if m["modality"] == "voice"]
        assert any(m["role"] == "assistant" and m["meta"].get("interrupted") for m in voice_msgs)
        assert any("E-1042" in m["text"] for m in voice_msgs if m["role"] == "user")

        traces = client.get(f"/api/sessions/{sid}/traces").json()["traces"]
        last = traces[-1]
        live_span = last["spans"][0]
        assert live_span["kind"] == "voice" and live_span["attrs"]["cost_usd"] > 0
        assert live_span["attrs"]["tokens_by_modality"]["in"]["AUDIO"] == 600

        # ...and the chat agent continues the same conversation
        r = client.post("/api/chat", json={"session_id": sid, "message": "thanks!"})
        assert r.status_code == 200 and "event: done" in r.text
