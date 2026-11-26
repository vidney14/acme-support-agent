"""Voice bridge: browser <-WebSocket-> FastAPI <-WebSocket-> Gemini Live (gemini-3.8-live).

    Browser mic ──PCM16 16kHz──▶ bridge ──send_realtime_input──▶ Gemini Live
    Browser spk ◀──PCM16 24kHz── bridge ◀──audio parts────────── Gemini Live
                ◀──JSON events── bridge ◀──transcripts / interrupted / tool_call / usage

Why proxy through our backend instead of browser -> Gemini directly?
  1. The API key never reaches the browser.
  2. Voice tool calls run HERE, through the same ToolRegistry, policy engine, turn guards
     and tracer as chat, so voice can't bypass a guardrail that chat enforces.
  3. Transcripts land in the shared session memory, so you can switch to chat mid-call.
Cost: one extra network hop, small next to model latency.

Interruptions (barge-in): Gemini's server-side VAD notices the user speaking over the
model, cancels the generation and sends `interrupted`. We forward that instantly so the
browser flushes its playback queue (the agent stops talking mid-word), and we mark the
assistant transcript as interrupted so later turns know the customer didn't hear the rest.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable

from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types

from app.agent.grounding import check_grounding
from app.agent.prompts import build_system_prompt
from app.agent.turn import TurnState, enforce_turn_guards
from app.llm.base import ToolCall
from app.llm.gemini import live_function_declarations, make_live_client
from app.llm.pricing import live_cost
from app.llm.resilience import classify
from app.observability.tracer import Span, Trace
from app.tools.registry import ToolContext

log = logging.getLogger(__name__)

IN_RATE, OUT_RATE = 16000, 24000
MAX_SESSION_S = 10 * 60          # cost + free-tier protection
IDLE_TIMEOUT_S = 120             # no mic audio for this long -> close


def recent_transcript(session, limit: int = 12) -> str:
    lines = []
    for m in session.messages[-limit:]:
        who = "Customer" if m.role == "user" else "Agent"
        lines.append(f"{who} ({m.modality}): {m.text[:400]}")
    return "\n".join(lines)


def build_live_config(container, session) -> types.LiveConnectConfig:
    s = container.settings
    system = build_system_prompt(session, "voice", session.crm.today)
    history = recent_transcript(session)
    if history:
        system += f"\n\n# Conversation so far (continue it naturally; don't repeat yourself)\n{history}"
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=system,
        tools=[types.Tool(function_declarations=live_function_declarations(container.registry.specs()))],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=s.live_voice))),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,  # snappy barge-in
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,      # don't cut people off
                prefix_padding_ms=40,
                silence_duration_ms=600),
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS),
        context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
    )


class VoiceTurn:
    """Accumulates one voice exchange (user speech -> agent reply) for memory + tracing."""

    def __init__(self, session_id: str):
        self.id = uuid.uuid4().hex[:10]
        self.trace = Trace(session_id=session_id, modality="voice")
        self.state = TurnState(turn_id=self.id, modality="voice")
        self.user_text = ""
        self.agent_text = ""
        self.interrupted = False
        self.last_user_audio_at: float | None = None
        self.first_audio_latency_ms: float | None = None
        self.usage = None
        self.started = time.perf_counter()

    @property
    def empty(self) -> bool:
        return not (self.user_text.strip() or self.agent_text.strip() or self.state.kb_searches
                    or self.state.tickets_created)


class VoiceBridge:
    def __init__(self, container, session, ws: WebSocket,
                 connect: Callable[[types.LiveConnectConfig], Any] | None = None):
        self.c = container
        self.session = session
        self.ws = ws
        self._connect = connect or getattr(container, "live_connect", None) or self._real_connect
        self.turn = VoiceTurn(session.session_id)
        self._send_lock = asyncio.Lock()
        self._last_client_audio = time.monotonic()
        self._closed = False
        self.live = None

    # ---------------------------------------------------------------- plumbing
    async def send_json(self, type_: str, data: dict | None = None) -> None:
        if self._closed:
            return
        async with self._send_lock:
            with contextlib.suppress(Exception):
                await self.ws.send_text(json.dumps({"type": type_, **(data or {})}, default=str))

    async def send_audio(self, pcm: bytes) -> None:
        if self._closed:
            return
        async with self._send_lock:
            with contextlib.suppress(Exception):
                await self.ws.send_bytes(pcm)

    async def emit(self, event: str, data: dict) -> None:
        """Adapter so ToolRegistry/turn guards can stream the SAME events as chat."""
        await self.send_json(event, data)

    def _real_connect(self, cfg: types.LiveConnectConfig):
        live_client = make_live_client(self.c.settings.gemini_api_key)
        return live_client.aio.live.connect(model=self.c.settings.live_model, config=cfg)

    def _ctx(self) -> ToolContext:
        return ToolContext(self.session, self.c.kb, self.c.rules, self.c.settings, self.emit, self.turn.state)

    # ---------------------------------------------------------------- lifecycle
    async def run(self) -> None:
        self.session.voice_connected = True
        await self.send_json("session", self.session.snapshot())
        try:
            async with self._connect(build_live_config(self.c, self.session)) as live:
                self.live = live
                await self.send_json("ready", {"model": self.c.settings.live_model, "input_sample_rate": IN_RATE,
                                               "output_sample_rate": OUT_RATE, "max_session_s": MAX_SESSION_S})
                if self.session.messages:
                    greet = ("[The customer switched from chat to a voice call. Greet them in one short sentence "
                             "and continue where you left off.]")
                else:
                    greet = "[The customer started a voice call. Greet them in one short sentence and ask how you can help.]"
                await live.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=greet)]), turn_complete=True)

                tasks = [asyncio.create_task(self._upstream(live), name="upstream"),
                         asyncio.create_task(self._downstream(live), name="downstream"),
                         asyncio.create_task(self._watchdog(), name="watchdog")]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception() and not isinstance(t.exception(), (WebSocketDisconnect, asyncio.CancelledError)):
                        raise t.exception()
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            err = classify(exc)
            log.warning("voice session error: %s", err)
            await self.send_json("error", {"recoverable": True, "kind": err.kind.value,
                                           "message": self._friendly(err)})
        finally:
            await self._finish_turn(reason="session_end")
            self.session.voice_connected = False
            self._closed = True

    async def _watchdog(self) -> None:
        start = time.monotonic()
        while True:
            await asyncio.sleep(1.0)
            elapsed = time.monotonic() - start
            if elapsed > MAX_SESSION_S - 30 and elapsed < MAX_SESSION_S - 29:
                await self.send_json("notice", {"kind": "voice_limit", "message": "Voice session ends in 30 seconds."})
            if elapsed > MAX_SESSION_S:
                await self.send_json("closed", {"reason": "max_duration"})
                return
            if time.monotonic() - self._last_client_audio > IDLE_TIMEOUT_S:
                await self.send_json("closed", {"reason": "idle"})
                return

    # ---------------------------------------------------------------- browser -> Gemini
    async def _upstream(self, live) -> None:
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                self._last_client_audio = time.monotonic()
                await live.send_realtime_input(audio=types.Blob(data=msg["bytes"], mime_type=f"audio/pcm;rate={IN_RATE}"))
            elif msg.get("text"):
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "stop":
                    return
                if kind == "audio_stream_end":  # mic muted/paused: flush VAD
                    await live.send_realtime_input(audio_stream_end=True)
                elif kind == "text" and data.get("text", "").strip():
                    text = data["text"].strip()[:2000]
                    self.turn.user_text += (" " if self.turn.user_text else "") + text
                    await self.send_json("transcript", {"role": "user", "text": text, "final": True, "typed": True})
                    await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]),
                                                   turn_complete=True)

    # ---------------------------------------------------------------- Gemini -> browser
    async def _downstream(self, live) -> None:
        while True:  # session.receive() yields one model turn, then returns
            async for resp in live.receive():
                await self._handle(live, resp)

    async def _handle(self, live, resp: types.LiveServerMessage) -> None:
        sc = resp.server_content
        if sc:
            if sc.input_transcription and sc.input_transcription.text:
                self.turn.user_text += sc.input_transcription.text
                self.turn.last_user_audio_at = time.perf_counter()
                await self.send_json("transcript", {"role": "user", "text": sc.input_transcription.text})
            if sc.model_turn:
                for part in sc.model_turn.parts or []:
                    if part.inline_data and part.inline_data.data:
                        if self.turn.first_audio_latency_ms is None:
                            ref = self.turn.last_user_audio_at or self.turn.started
                            self.turn.first_audio_latency_ms = (time.perf_counter() - ref) * 1000
                            self.turn.trace.mark_first_token()
                        await self.send_audio(part.inline_data.data)
            if sc.output_transcription and sc.output_transcription.text:
                self.turn.agent_text += sc.output_transcription.text
                await self.send_json("transcript", {"role": "assistant", "text": sc.output_transcription.text})
            if sc.interrupted:
                # Close out the interrupted exchange now; the customer's new utterance starts a fresh turn.
                self.turn.interrupted = True
                await self.send_json("interrupted", {"turn_id": self.turn.id})
                await self._finish_turn(reason="interrupted")
            if sc.turn_complete:
                await self._finish_turn(reason="turn_complete")

        if resp.tool_call and resp.tool_call.function_calls:
            await self._run_tools(live, resp.tool_call.function_calls)
        if resp.tool_call_cancellation:
            await self.send_json("notice", {"kind": "tool_cancelled",
                                            "message": "Customer interrupted; pending tool call cancelled."})
        if resp.usage_metadata:
            self.turn.usage = resp.usage_metadata  # last value per turn
        if resp.go_away:
            await self.send_json("go_away", {"time_left": str(resp.go_away.time_left)})

    async def _run_tools(self, live, calls) -> None:
        await self.send_json("status", {"phase": "act", "label": "Running tools", "tools": [c.name for c in calls]})
        responses = []
        async with self.session.turn_lock:  # don't race a chat turn on the same session
            for fc in calls:
                t0 = self.turn.trace.now_ms()
                call = ToolCall(id=fc.id or f"call_{uuid.uuid4().hex[:8]}", name=fc.name, args=dict(fc.args or {}))
                # tracer ContextVar isn't bound here (long-lived task), so record the span explicitly
                outcome = await self.c.registry.execute(call, self._ctx())
                self.turn.trace.add_span(Span(name=f"tool.{call.name}", kind="tool", start_ms=t0,
                                              duration_ms=self.turn.trace.now_ms() - t0,
                                              status="ok" if outcome.ok else "error",
                                              attrs={"args": call.args if call.name != "verify_identity"
                                                     else {**call.args, "pin": "••••"},
                                                     "summary": outcome.summary, "attempts": outcome.attempts,
                                                     "result": outcome.data if outcome.ok else outcome.error}))
                responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=outcome.for_model()))
        await live.send_tool_response(function_responses=responses)

    async def _finish_turn(self, reason: str) -> None:
        turn = self.turn
        self.turn = VoiceTurn(self.session.session_id)
        if turn.empty:
            return
        ctx = ToolContext(self.session, self.c.kb, self.c.rules, self.c.settings, self.emit, turn.state)
        for t in await enforce_turn_guards(ctx):
            await self.send_json("notice", {"kind": "guardrail",
                                            "message": f"Guardrail auto-escalated: {t['ticket_id']}"})
            if self.live is not None and reason == "turn_complete":
                # Make the agent SAY it, not just show it: inject a system note as client content.
                with contextlib.suppress(Exception):
                    await self.live.send_client_content(turns=types.Content(role="user", parts=[types.Part(
                        text=f"[System notice: escalation ticket {t['ticket_id']} was opened automatically "
                             f"({t['summary'][:120]}). Tell the customer briefly, including the ticket number "
                             f"and that a human will respond within {t['response_target']}.]")]),
                        turn_complete=True)
        # Voice answers can't carry citation tags, so grounding = "did it use retrieved sources?"
        g = check_grounding(turn.agent_text, turn.state)
        if g.status in ("unsupported", "no_claims") and turn.state.retrieved_ids:
            g.status, g.citations = "grounded", sorted(turn.state.retrieved_ids)
            g.detail = f"Spoken answer drew on {len(g.citations)} retrieved passage(s)."
        if turn.user_text.strip():
            self.session.add_message("user", turn.user_text.strip(), "voice", turn.id)
        if turn.agent_text.strip():
            self.session.add_message("assistant", turn.agent_text.strip(), "voice", turn.id,
                                     interrupted=turn.interrupted, grounding=g.to_dict())
        tr = turn.trace
        tr.user_input = turn.user_text.strip()
        attrs: dict[str, Any] = {"model": self.c.settings.live_model, "interrupted": turn.interrupted,
                                 "first_audio_latency_ms": None if turn.first_audio_latency_ms is None
                                 else round(turn.first_audio_latency_ms, 1)}
        if turn.usage:
            mod = {"in": {}, "out": {}}
            for d in turn.usage.prompt_tokens_details or []:
                mod["in"][str(d.modality.name if d.modality else "TEXT")] = d.token_count or 0
            for d in turn.usage.response_tokens_details or []:
                mod["out"][str(d.modality.name if d.modality else "TEXT")] = d.token_count or 0
            attrs.update(input_tokens=turn.usage.prompt_token_count or 0,
                         output_tokens=turn.usage.response_token_count or 0, tokens_by_modality=mod,
                         cost_usd=round(live_cost(self.c.settings.live_model,
                                                  in_text=mod["in"].get("TEXT", 0), in_audio=mod["in"].get("AUDIO", 0),
                                                  out_text=mod["out"].get("TEXT", 0),
                                                  out_audio=mod["out"].get("AUDIO", 0)), 6))
        tr.spans.insert(0, Span(name="live.turn", kind="voice", start_ms=0.0, duration_ms=tr.now_ms(),
                                attrs={**attrs, "agent_transcript": turn.agent_text.strip()[:2000]}))
        tr.ttft_ms = turn.first_audio_latency_ms
        tr.finish("ok" if reason != "error" else "error")
        self.session.add_trace(tr)
        await self.send_json("turn_complete", {"turn_id": turn.id, "interrupted": turn.interrupted,
                                               "latency_ms": attrs["first_audio_latency_ms"],
                                               "grounding": g.to_dict()})
        await self.send_json("trace", tr.to_dict())
        await self.send_json("session", self.session.snapshot())

    @staticmethod
    def _friendly(err) -> str:
        return {
            "rate_limit": "Voice is busy right now (rate limit). Please try again shortly, or keep chatting by text.",
            "client_error": "Voice couldn't start with the current configuration. You can keep chatting by text.",
            "timeout": "The voice connection timed out. Please try again.",
        }.get(err.kind.value, "The voice connection dropped. You can reconnect or continue by text.")
