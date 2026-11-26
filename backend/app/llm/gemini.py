"""Gemini adapter (google-genai SDK, generateContent API, manual function calling)."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncIterator

from google import genai
from google.genai import types

from app.llm.base import (AssistantText, Item, ModelTurn, StreamDone, StreamItem, TextDelta, ToolCall,
                          ToolResult, ToolSpec, Usage, UserText)
from app.llm.resilience import ErrorKind, ProviderError

# Finish reasons that mean "the model produced something unusable; trying again may work".
_RETRYABLE_FINISH = {"MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE", "MISSING_THOUGHT_SIGNATURE"}


def make_client(api_key: str) -> genai.Client:
    # Disable the SDK's own retries: retry/fallback policy lives in one place (our code).
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
    )


# The Live (bidi WebSocket) models are only served on v1alpha; the default version returns HTTP 404.
LIVE_API_VERSION = "v1alpha"


def make_live_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version=LIVE_API_VERSION))


def _thinking(level: str | None) -> types.ThinkingConfig | None:
    if not level:
        return None
    return types.ThinkingConfig(thinking_level=types.ThinkingLevel[level.upper()])


def to_contents(items: list[Item]) -> list[types.Content]:
    contents: list[types.Content] = []
    for it in items:
        if isinstance(it, UserText):
            contents.append(types.Content(role="user", parts=[types.Part(text=it.text)]))
        elif isinstance(it, AssistantText):
            contents.append(types.Content(role="model", parts=[types.Part(text=it.text)]))
        elif isinstance(it, ModelTurn):
            if it.native is not None:
                contents.append(it.native)  # verbatim replay keeps thought signatures intact
            else:
                parts = [types.Part(text=it.text)] if it.text else []
                parts += [types.Part(function_call=types.FunctionCall(id=c.provider_id, name=c.name, args=c.args))
                          for c in it.tool_calls]
                contents.append(types.Content(role="model", parts=parts))
        elif isinstance(it, ToolResult):
            part = types.Part(function_response=types.FunctionResponse(
                id=it.provider_id, name=it.name, response=it.result))
            # Parallel calls: all responses go back together in ONE user content.
            if contents and contents[-1].role == "user" and contents[-1].parts and contents[-1].parts[0].function_response:
                contents[-1].parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
    return contents


def _tools(tools: list[ToolSpec]) -> list[types.Tool] | None:
    if not tools:
        return None
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(name=t.name, description=t.description,
                                  parameters_json_schema=t.parameters if t.parameters.get("properties") else None)
        for t in tools])]


def to_schema(js: dict) -> types.Schema:
    """JSON Schema (our simple subset) -> Gemini Schema. Used for Live API declarations,
    which take the OpenAPI-style `parameters` field."""
    kind = js.get("type", "string").upper()
    s = types.Schema(type=types.Type[kind], description=js.get("description"), enum=js.get("enum"))
    if kind == "OBJECT":
        s.properties = {k: to_schema(v) for k, v in js.get("properties", {}).items()}
        s.required = js.get("required") or None
    return s


def live_function_declarations(tools: list[ToolSpec]) -> list[types.FunctionDeclaration]:
    # BLOCKING: gemini-3.8-live defaults to async (NON_BLOCKING) calls, where the model keeps
    # talking while the tool runs. For support flows the answer *depends* on the result
    # ("am I verified?", "was the refund approved?"), so we want the model to wait.
    return [types.FunctionDeclaration(
        name=t.name, description=t.description,
        parameters=to_schema(t.parameters) if t.parameters.get("properties") else None,
        behavior=types.Behavior.BLOCKING) for t in tools]


class GeminiProvider:
    name = "gemini"

    def __init__(self, client: genai.Client):
        self.client = client

    async def stream(self, *, model: str, system: str, items: list[Item], tools: list[ToolSpec],
                     thinking_level: str | None, first_byte_timeout_s: float,
                     idle_timeout_s: float) -> AsyncIterator[StreamItem]:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=_tools(tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=_thinking(thinking_level),
        )
        stream = await asyncio.wait_for(
            self.client.aio.models.generate_content_stream(model=model, contents=to_contents(items), config=config),
            timeout=first_byte_timeout_s,
        )
        it = stream.__aiter__()
        parts: list[types.Part] = []
        text, calls = "", []
        usage_md, finish = None, None
        first = True
        while True:
            try:
                chunk = await asyncio.wait_for(it.__anext__(), timeout=first_byte_timeout_s if first else idle_timeout_s)
            except StopAsyncIteration:
                break
            first = False
            if chunk.usage_metadata:
                usage_md = chunk.usage_metadata
            cand = chunk.candidates[0] if chunk.candidates else None
            if cand is None:
                continue
            if cand.finish_reason:
                finish = cand.finish_reason.name if hasattr(cand.finish_reason, "name") else str(cand.finish_reason)
            for p in (cand.content.parts or []) if cand.content else []:
                parts.append(p)  # keep EVERY part (signatures can ride on any of them)
                if p.text and not p.thought:
                    text += p.text
                    yield TextDelta(p.text)
                if p.function_call:
                    fc = p.function_call
                    calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=fc.name,
                                          args=dict(fc.args or {}), provider_id=fc.id))

        if finish in _RETRYABLE_FINISH and not calls and not text:
            raise ProviderError(ErrorKind.UNKNOWN, f"Model returned {finish}", retryable=True)

        usage = Usage()
        if usage_md:
            thinking = usage_md.thoughts_token_count or 0
            usage = Usage(input_tokens=usage_md.prompt_token_count or 0,
                          output_tokens=(usage_md.candidates_token_count or 0) + thinking,
                          thinking_tokens=thinking)
        native = types.Content(role="model", parts=parts) if parts else None
        yield StreamDone(turn=ModelTurn(text=text, tool_calls=calls, native=native), usage=usage,
                         model=model, finish_reason=finish)

    async def generate_json(self, *, model: str, system: str, prompt: str, schema: dict,
                            thinking_level: str | None, timeout_s: float) -> tuple[dict, Usage]:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=_thinking(thinking_level),
        )
        resp = await asyncio.wait_for(
            self.client.aio.models.generate_content(model=model, contents=prompt, config=config), timeout=timeout_s)
        md = resp.usage_metadata
        usage = Usage(input_tokens=(md.prompt_token_count or 0) if md else 0,
                      output_tokens=((md.candidates_token_count or 0) + (md.thoughts_token_count or 0)) if md else 0)
        try:
            return json.loads(resp.text or "{}"), usage
        except json.JSONDecodeError as exc:
            raise ProviderError(ErrorKind.UNKNOWN, f"Verifier returned invalid JSON: {exc}", retryable=True) from exc
