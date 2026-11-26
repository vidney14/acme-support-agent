"""HTTP API: chat over Server-Sent Events, session/trace inspection, demo controls."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.api.ratelimit import client_key

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

HEARTBEAT_S = 15.0


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    retry: bool = False


class FaultRequest(BaseModel):
    llm_rate_limit: int | None = Field(default=None, ge=0, le=5)
    llm_timeout: int | None = Field(default=None, ge=0, le=5)
    tool_flaky: int | None = Field(default=None, ge=0, le=5)
    crm_outage: bool | None = None


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _container(request: Request):
    return request.app.state.container


@router.get("/health")
async def health(request: Request):
    c = _container(request)
    kb = c.kb.status
    return {"status": "ok" if kb.ready else "degraded", "provider": c.provider.name,
            "kb": {"ready": kb.ready, "chunks": kb.chunks, "documents": kb.documents, "error": kb.error,
                   "reused_index": kb.reused_index},
            "sessions": len(c.sessions)}


@router.get("/config")
async def config(request: Request):
    c = _container(request)
    s = c.settings
    return {
        "provider": c.provider.name,
        "models": {"chat": s.chat_model, "fallback": s.fallback_model, "verifier": s.verifier_model,
                   "live": s.live_model, "embedding": s.embedding_model},
        "guardrails": {"confidence_high": c.kb.gate.high, "confidence_low": c.kb.gate.low},
        "fault_injection": s.enable_fault_injection,
        "demo_accounts": c.crm.demo_accounts(),
        "voice_enabled": c.genai_client is not None or c.live_connect is not None,
        "voice_simulated": c.genai_client is None and c.live_connect is not None,
    }


@router.post("/sessions")
async def create_session(request: Request):
    s = _container(request).sessions.get_or_create(None)
    return s.snapshot()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    s = _container(request).sessions.get(session_id)
    if not s:
        raise HTTPException(404, "session not found or expired")
    return {**s.snapshot(), "messages": [m.to_dict() for m in s.messages]}


@router.get("/sessions/{session_id}/traces")
async def get_traces(session_id: str, request: Request):
    s = _container(request).sessions.get(session_id)
    if not s:
        raise HTTPException(404, "session not found or expired")
    return {"traces": [t.to_dict() for t in s.traces]}


@router.post("/sessions/{session_id}/faults")
async def set_faults(session_id: str, body: FaultRequest, request: Request):
    c = _container(request)
    if not c.settings.enable_fault_injection:
        raise HTTPException(403, "fault injection disabled")
    s = c.sessions.get(session_id)
    if not s:
        raise HTTPException(404, "session not found or expired")
    s.faults.update(body.model_dump(exclude_none=True))
    return s.faults.to_dict()


@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    c = _container(request)
    allowed, retry_after = request.app.state.chat_limiter.check(client_key(request))
    if not allowed:
        return JSONResponse({"error": "rate_limited", "message": "You're sending messages too quickly.",
                             "retry_after_s": retry_after}, status_code=429,
                            headers={"Retry-After": str(int(retry_after) + 1)})

    session = c.sessions.get_or_create(body.session_id)
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event: str, data: dict) -> None:
        await queue.put((event, data))

    async def run() -> None:
        try:
            await c.orchestrator.run_turn(session, body.message, emit, retry=body.retry)
        finally:
            await queue.put(None)

    # The turn runs as its own task: if the browser disconnects mid-turn, the turn still
    # completes server-side, so session state (refunds, tickets, memory) stays consistent.
    task = asyncio.create_task(run())
    request.app.state.background_tasks.add(task)
    task.add_done_callback(request.app.state.background_tasks.discard)

    async def stream():
        yield _sse("session", session.snapshot())
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
            except asyncio.TimeoutError:
                yield ": ping\n\n"  # keeps proxies from closing an idle stream
                continue
            if item is None:
                break
            if await request.is_disconnected():
                log.info("client disconnected; turn continues server-side")
                break
            yield _sse(*item)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})
