"""Simulated Gemini Live session (offline voice demo + tests).

Emits real `google.genai.types.LiveServerMessage` objects, so the bridge code under test
is exactly the production code. Used when LLM_PROVIDER=fake: the voice UI (mic capture,
waveforms, transcripts, tool steps, barge-in, playback flush) can be exercised end to end
without an API key. It doesn't understand speech; it plays a fixed scripted call.
"""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from typing import Any

from google.genai import types

AWAIT_TOOL = "AWAIT_TOOL"


def msg(**kw) -> types.LiveServerMessage:
    return types.LiveServerMessage(**kw)


def content(**kw) -> types.LiveServerMessage:
    return msg(server_content=types.LiveServerContent(**kw))


def say(text: str) -> types.LiveServerMessage:
    return content(output_transcription=types.Transcription(text=text))


def heard(text: str) -> types.LiveServerMessage:
    return content(input_transcription=types.Transcription(text=text))


def tone_chunks(seconds: float, freq: float = 190.0, chunk_s: float = 0.2) -> list[types.LiveServerMessage]:
    """Speech-like audio: a tone with a syllable-rate amplitude envelope, 24 kHz PCM16."""
    rate = 24000
    out = []
    n_total = int(seconds * rate)
    step = int(chunk_s * rate)
    for start in range(0, n_total, step):
        n = min(step, n_total - start)
        buf = bytearray()
        for i in range(start, start + n):
            t = i / rate
            env = 0.35 + 0.65 * abs(math.sin(2 * math.pi * 3.2 * t)) * (0.6 + 0.4 * math.sin(2 * math.pi * 0.7 * t))
            v = int(9000 * env * math.sin(2 * math.pi * freq * t))
            buf += v.to_bytes(2, "little", signed=True)
        part = types.Part(inline_data=types.Blob(data=bytes(buf), mime_type="audio/pcm;rate=24000"))
        out.append(content(model_turn=types.Content(role="model", parts=[part])))
    return out


def done() -> types.LiveServerMessage:
    return content(turn_complete=True)


def demo_script() -> list[list[Any]]:
    """Greeting; then a sync-error question answered via the knowledge base, interrupted by
    the customer mid-answer; then the agent recovers with a short reply."""
    return [
        [say("Hi, this is Acme Assist. "), *tone_chunks(1.2), say("How can I help today?"), *tone_chunks(0.8), done()],
        [heard("My desktop app shows error E-1042"), ("sleep", 0.4),
         msg(tool_call=types.LiveServerToolCall(function_calls=[types.FunctionCall(
             id="fake-1", name="search_knowledge_base", args={"query": "sync error E-1042"})])),
         AWAIT_TOOL,
         say("Let me explain. That error means the file path is longer than 260 characters, "),
         *tone_chunks(2.4), say("which Windows doesn't allow by default, so you'll want to "), *tone_chunks(1.0),
         ("sleep", 1.6),
         content(interrupted=True),
         heard(" Wait, how do I actually fix it?"), ("sleep", 0.5),
         say("Shorten the folder or file names, or move the file closer to the drive root."),
         *tone_chunks(2.0),
         msg(usage_metadata=types.UsageMetadata(
             prompt_token_count=1450, response_token_count=310,
             prompt_tokens_details=[types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=900),
                                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=550)],
             response_tokens_details=[types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=310)])),
         done()],
    ]


class FakeLiveSession:
    def __init__(self, script: list[list[Any]], pace_s: float = 0.0):
        self.script = script
        self.pace_s = pace_s
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent_audio = 0
        self.tool_responses: list[types.FunctionResponse] = []
        self.client_contents: list[Any] = []
        self.config: types.LiveConnectConfig | None = None

    async def send_realtime_input(self, audio=None, audio_stream_end=None, text=None, **_):
        if audio is not None:
            self.sent_audio += 1
            if self.sent_audio == 1:
                await self.inbox.put("user_spoke")

    async def send_client_content(self, turns=None, turn_complete=True):
        self.client_contents.append(turns)
        if turn_complete:
            await self.inbox.put("client_content")

    async def send_tool_response(self, function_responses):
        self.tool_responses.extend(function_responses)
        await self.inbox.put("tool_response")

    async def receive(self):
        if not self.script:
            await asyncio.sleep(3600)
        await self.inbox.get()
        for m in self.script.pop(0):
            if m == AWAIT_TOOL:
                while (await self.inbox.get()) != "tool_response":
                    pass
                continue
            if isinstance(m, tuple) and m[0] == "sleep":
                await asyncio.sleep(m[1] if self.pace_s else 0)
                continue
            await asyncio.sleep(self.pace_s)
            yield m
            if m.server_content and m.server_content.turn_complete:
                return


def make_connect(factory):
    """`factory()` returns a fresh FakeLiveSession per call (or a fixed instance for tests)."""
    @asynccontextmanager
    async def connect(cfg):
        session = factory()
        session.config = cfg
        yield session
    return connect


def demo_connect():
    return make_connect(lambda: FakeLiveSession(demo_script(), pace_s=0.12))
