from __future__ import annotations

import uuid

import pytest

from app.config import Settings
from app.container import build_container
from app.llm.base import ModelTurn, ToolCall


def call(name: str, **args) -> ModelTurn:
    """A model turn that requests one tool call."""
    return ModelTurn(text="", tool_calls=[ToolCall(uuid.uuid4().hex[:6], name, args)])


def say(text: str) -> ModelTurn:
    return ModelTurn(text=text)


class Recorder:
    """Collects emitted events so tests can assert on the exact stream the UI receives."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, event: str, data: dict) -> None:
        self.events.append((event, data))

    def of(self, kind: str) -> list[dict]:
        return [d for e, d in self.events if e == kind]

    @property
    def text(self) -> str:
        out = ""
        for e, d in self.events:
            if e == "token":
                out += d["text"]
            elif e == "draft_reset":
                out = out[: len(out) - d["chars"]]
        return out


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_provider="fake", llm_max_retries=1, tool_max_retries=1)


@pytest.fixture
async def container(settings):
    c = build_container(settings)
    await c.kb.ingest()
    return c


@pytest.fixture
def make_agent(container):
    """Swap the orchestrator's provider for a ScriptedProvider with the given script."""
    from app.agent.orchestrator import AgentOrchestrator
    from app.llm.fake import ScriptedProvider

    def _make(script, json_responses=None, **overrides):
        provider = ScriptedProvider(script, json_responses)
        s = container.settings.model_copy(update=overrides) if overrides else container.settings
        container.kb.gate.provider = provider
        return AgentOrchestrator(provider, container.registry, container.kb, container.rules, s), provider

    return _make
