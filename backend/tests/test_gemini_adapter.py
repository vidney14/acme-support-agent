"""Gemini adapter against real SDK types (no network): streaming assembly, thought-signature
preservation, content conversion, Live declarations and error classification."""

from __future__ import annotations

from google import genai
from google.genai import types

from app.llm.base import ModelTurn, StreamDone, TextDelta, ToolCall, ToolResult, ToolSpec, UserText
from app.llm.gemini import GeminiProvider, _tools, live_function_declarations, to_contents
from app.llm.resilience import ErrorKind, classify

SPEC = ToolSpec("lookup", "Look up a thing", {"type": "object", "properties": {"id": {"type": "string"}},
                                              "required": ["id"]})
NOARGS = ToolSpec("ping", "No arguments", {"type": "object", "properties": {}, "required": []})


def chunk(parts=None, finish=None, usage=None) -> types.GenerateContentResponse:
    cand = types.Candidate(content=types.Content(role="model", parts=parts or []),
                           finish_reason=finish)
    return types.GenerateContentResponse(candidates=[cand], usage_metadata=usage)


async def test_stream_assembles_text_tool_calls_and_keeps_signatures(monkeypatch):
    client = genai.Client(api_key="test")
    sig = b"\x01thought-signature"

    async def fake_stream(**kwargs):
        assert kwargs["config"].automatic_function_calling.disable is True
        assert kwargs["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW

        async def gen():
            yield chunk([types.Part(text="Let me check. ")])
            yield chunk([types.Part(function_call=types.FunctionCall(name="lookup", args={"id": "A-1"}),
                                    thought_signature=sig)])
            yield chunk([], finish=types.FinishReason.STOP,
                        usage=types.GenerateContentResponseUsageMetadata(
                            prompt_token_count=120, candidates_token_count=15, thoughts_token_count=30))
        return gen()

    monkeypatch.setattr(client.aio.models, "generate_content_stream", fake_stream)
    events = [e async for e in GeminiProvider(client).stream(
        model="gemini-3.8-flash", system="sys", items=[UserText("hi")], tools=[SPEC], thinking_level="low",
        first_byte_timeout_s=5, idle_timeout_s=5)]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Let me check. "]
    done = events[-1]
    assert isinstance(done, StreamDone) and done.finish_reason == "STOP"
    assert done.turn.tool_calls[0].name == "lookup" and done.turn.tool_calls[0].args == {"id": "A-1"}
    assert done.usage.input_tokens == 120 and done.usage.output_tokens == 45 and done.usage.thinking_tokens == 30
    # The exact parts (with the signature) are kept for replay on the next request
    assert any(p.thought_signature == sig for p in done.turn.native.parts)


def test_to_contents_replays_native_and_groups_parallel_tool_results():
    native = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="a", args={}),
                                                           thought_signature=b"s")])
    items = [UserText("q"), ModelTurn("", [ToolCall("1", "a", {}), ToolCall("2", "b", {})], native=native),
             ToolResult("1", "a", {"ok": True}), ToolResult("2", "b", {"ok": False})]
    contents = to_contents(items)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1] is native                      # verbatim replay
    assert [p.function_response.name for p in contents[2].parts] == ["a", "b"]


def test_tool_declarations_for_generate_content_and_live():
    decl = _tools([SPEC, NOARGS])[0].function_declarations
    assert decl[0].parameters_json_schema["required"] == ["id"]
    assert decl[1].parameters_json_schema is None     # no-arg tools omit the schema
    live = live_function_declarations([SPEC, NOARGS])
    assert all(d.behavior == types.Behavior.BLOCKING for d in live)
    assert live[0].parameters.type == types.Type.OBJECT and live[0].parameters.required == ["id"]
    assert live[1].parameters is None


def test_error_classification():
    class Fake(Exception):
        def __init__(self, code, msg):
            super().__init__(msg)
            self.code = code

    rl = classify(Fake(429, "RESOURCE_EXHAUSTED ... 'retryDelay': '12s'"))
    assert rl.kind is ErrorKind.RATE_LIMIT and rl.retry_after_s == 12.0 and rl.retryable
    assert classify(Fake(503, "unavailable")).retryable
    bad = classify(Fake(400, "invalid thinking level"))
    assert bad.kind is ErrorKind.CLIENT and not bad.retryable
    assert classify(TimeoutError()).kind is ErrorKind.TIMEOUT
