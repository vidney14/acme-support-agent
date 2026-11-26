"""Smoke-test every Gemini integration with your real key (takes about a minute).

    cd backend && python -m scripts.smoke_test

Checks: embeddings, streaming tool call + thought-signature replay, fallback model,
JSON-mode verifier, Live API audio + transcription, Live API tool calling.
Exit code 0 if everything passed.
"""

from __future__ import annotations

import asyncio
import sys
import time
import traceback

from google.genai import types

from app.config import get_settings
from app.llm.base import ToolResult, ToolSpec, UserText
from app.llm.gemini import GeminiProvider, live_function_declarations, make_client, make_live_client
from app.rag.embeddings import GeminiEmbedder

results: list[tuple[str, bool, str]] = []


async def check(name: str, coro):
    t0 = time.perf_counter()
    try:
        detail = await coro
        results.append((name, True, f"{detail} ({(time.perf_counter() - t0):.1f}s)"))
    except Exception as exc:  # noqa: BLE001
        results.append((name, False, f"{type(exc).__name__}: {str(exc)[:300]}"))
        traceback.print_exc(limit=1)


WEATHER = ToolSpec("lookup_order", "Look up an order's status by id.",
                   {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]})


async def embeddings(client, s):
    e = GeminiEmbedder(client, s.embedding_model, s.embedding_dim)
    vecs = await e.embed_documents(["refunds within 14 days", "sync error E-1042"])
    q = await e.embed_query("how long do I have to get my money back?")
    sim = sum(a * b for a, b in zip(q, vecs[0])) / 1.0
    assert len(q) == s.embedding_dim
    return f"dim={len(q)}, sim(query, relevant)={sim:.3f}"


async def tool_roundtrip(provider: GeminiProvider, model: str, thinking: str):
    items = [UserText("What's the status of order A-17? Use the tool.")]
    turn = None
    async for ev in provider.stream(model=model, system="You are a helpful assistant.", items=items,
                                    tools=[WEATHER], thinking_level=thinking, first_byte_timeout_s=30,
                                    idle_timeout_s=30):
        if not hasattr(ev, "text"):
            turn = ev.turn
    assert turn and turn.tool_calls, "model did not call the tool"
    c = turn.tool_calls[0]
    items += [turn, ToolResult(c.id, c.name, {"status": "shipped", "eta": "Friday"}, c.provider_id)]
    final = None
    async for ev in provider.stream(model=model, system="You are a helpful assistant.", items=items,
                                    tools=[WEATHER], thinking_level=thinking, first_byte_timeout_s=30,
                                    idle_timeout_s=30):
        if not hasattr(ev, "text"):
            final = ev
    assert final and final.turn.text, "no final answer after tool result (thought signature issue?)"
    return f"called {c.name}({c.args}); answer: {final.turn.text[:60]!r}; tokens in/out {final.usage.input_tokens}/{final.usage.output_tokens}"


async def verifier(provider: GeminiProvider, s):
    from app.rag.guardrails import VERIFIER_SCHEMA, VERIFIER_SYSTEM
    data, _ = await provider.generate_json(
        model=s.verifier_model, system=VERIFIER_SYSTEM, schema=VERIFIER_SCHEMA, thinking_level=s.fallback_thinking_level,
        prompt="QUESTION:\nDo you offer student discounts?\n\nPASSAGES:\n[p#1] Pro costs $12/month.", timeout_s=20)
    assert data["answerable"] in ("yes", "partial", "no")
    return f"answerable={data['answerable']} ({data['reason'][:60]})"


async def live_audio(client, s):
    cfg = types.LiveConnectConfig(response_modalities=[types.Modality.AUDIO],
                                  output_audio_transcription=types.AudioTranscriptionConfig(),
                                  system_instruction="Answer in one short sentence.")
    audio_bytes, transcript = 0, ""
    async with client.aio.live.connect(model=s.live_model, config=cfg) as live:
        await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(text="Say hello.")]),
                                       turn_complete=True)
        async for m in live.receive():
            sc = m.server_content
            if sc and sc.model_turn:
                audio_bytes += sum(len(p.inline_data.data) for p in sc.model_turn.parts or [] if p.inline_data)
            if sc and sc.output_transcription and sc.output_transcription.text:
                transcript += sc.output_transcription.text
    assert audio_bytes > 0, "no audio received"
    return f"{audio_bytes / 48000:.1f}s of 24kHz audio, transcript {transcript.strip()[:50]!r}"


async def live_tools(client, s):
    cfg = types.LiveConnectConfig(response_modalities=[types.Modality.AUDIO],
                                  tools=[types.Tool(function_declarations=live_function_declarations([WEATHER]))],
                                  output_audio_transcription=types.AudioTranscriptionConfig())
    called = None
    async with client.aio.live.connect(model=s.live_model, config=cfg) as live:
        await live.send_client_content(turns=types.Content(role="user", parts=[types.Part(
            text="Check the status of order A-17 with the tool.")]), turn_complete=True)
        for _ in range(3):
            async for m in live.receive():
                if m.tool_call:
                    fc = m.tool_call.function_calls[0]
                    called = f"{fc.name}({dict(fc.args or {})})"
                    await live.send_tool_response(function_responses=[types.FunctionResponse(
                        id=fc.id, name=fc.name, response={"status": "shipped"})])
            if called:
                break
    assert called, "Live model did not call the tool"
    return f"Live tool call {called} answered"


async def main() -> int:
    s = get_settings()
    if not s.gemini_api_key:
        print("Set GEMINI_API_KEY in backend/.env first.")
        return 2
    client = make_client(s.gemini_api_key)
    provider = GeminiProvider(client)
    print(f"chat={s.chat_model} fallback={s.fallback_model} live={s.live_model} embed={s.embedding_model}\n")
    await check("Embeddings", embeddings(client, s))
    await check(f"Chat tool round-trip ({s.chat_model})", tool_roundtrip(provider, s.chat_model, s.chat_thinking_level))
    await check(f"Fallback tool round-trip ({s.fallback_model})",
                tool_roundtrip(provider, s.fallback_model, s.fallback_thinking_level))
    await check(f"Verifier JSON ({s.verifier_model})", verifier(provider, s))
    live_client = make_live_client(s.gemini_api_key)
    await check(f"Live audio ({s.live_model})", live_audio(live_client, s))
    await check("Live tool calling", live_tools(live_client, s))

    print()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    failed = [r for r in results if not r[1]]
    if failed:
        print("\nHints: 404/NOT_FOUND -> change the model name in .env (see ai.google.dev/gemini-api/docs/models); "
              "429 -> free-tier quota, wait a minute; 400 thinking_level -> set CHAT_THINKING_LEVEL to a supported value.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
