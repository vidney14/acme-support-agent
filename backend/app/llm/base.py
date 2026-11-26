"""Provider-agnostic LLM types.

The orchestrator only ever speaks in these types. Gemini-specific details (Content/Part
objects, thought signatures) stay inside `gemini.py`, so swapping in OpenAI or Anthropic
means writing one new adapter, not touching the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, Union


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict            # JSON Schema for the arguments


@dataclass
class ToolCall:
    id: str                     # our id (always set), used for UI step tracking
    name: str
    args: dict
    provider_id: str | None = None  # the model's own call id (may be None on Gemini)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0      # includes thinking tokens (they are billed as output)
    thinking_tokens: int = 0


# ---- Conversation items -----------------------------------------------------------

@dataclass
class UserText:
    text: str


@dataclass
class AssistantText:
    """A finished assistant message from an earlier turn (plain text)."""
    text: str


@dataclass
class ModelTurn:
    """One model response inside the *current* turn. `native` keeps the provider's exact
    content so it can be replayed verbatim (Gemini 3 requires thought signatures back)."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    native: Any = None


@dataclass
class ToolResult:
    call_id: str
    name: str
    result: dict
    provider_id: str | None = None


Item = Union[UserText, AssistantText, ModelTurn, ToolResult]


# ---- Streaming output ---------------------------------------------------------------

@dataclass
class TextDelta:
    text: str


@dataclass
class StreamDone:
    turn: ModelTurn
    usage: Usage
    model: str
    finish_reason: str | None = None


StreamItem = Union[TextDelta, StreamDone]


class LLMProvider(Protocol):
    name: str

    def stream(self, *, model: str, system: str, items: list[Item], tools: list[ToolSpec],
               thinking_level: str | None, first_byte_timeout_s: float,
               idle_timeout_s: float) -> AsyncIterator[StreamItem]: ...

    async def generate_json(self, *, model: str, system: str, prompt: str, schema: dict,
                            thinking_level: str | None, timeout_s: float) -> tuple[dict, Usage]: ...
