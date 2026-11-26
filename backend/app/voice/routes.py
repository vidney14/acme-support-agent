"""Voice WebSocket endpoint: admission control, then hand off to VoiceBridge."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from app.api.ratelimit import client_key
from app.voice.bridge import VoiceBridge

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/voice")
async def voice(ws: WebSocket, session_id: str = ""):
    app = ws.app
    c = app.state.container
    s = c.settings

    # CORS doesn't apply to WebSockets, so check Origin ourselves (blocks cross-site hijacking).
    origin = ws.headers.get("origin", "")
    if origin and s.cors_origin_list != ["*"] and origin not in s.cors_origin_list:
        await ws.close(code=4403, reason="origin not allowed")
        return

    await ws.accept()

    async def reject(code: str, message: str) -> None:
        await ws.send_json({"type": "error", "recoverable": code != "voice_disabled", "kind": code, "message": message})
        await ws.close(code=4429 if code == "rate_limited" else 4400)

    if c.genai_client is None and c.live_connect is None:
        return await reject("voice_disabled", "Voice needs a Gemini API key (the server is in offline mode).")
    allowed, retry_after = app.state.voice_limiter.check(f"voice:{client_key(ws)}")
    if not allowed:
        return await reject("rate_limited", f"Too many voice sessions. Try again in {int(retry_after) + 1}s.")
    if app.state.active_voice >= s.max_concurrent_voice_sessions:
        return await reject("rate_limited", "All voice lines are busy right now. Please use chat or retry shortly.")

    session = c.sessions.get_or_create(session_id or None)
    app.state.active_voice += 1
    try:
        await VoiceBridge(c, session, ws).run()
    finally:
        app.state.active_voice -= 1
        try:
            await ws.close()
        except Exception:  # noqa: BLE001 - already closed
            pass
