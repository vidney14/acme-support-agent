"""Tool registry + executor.

Every tool call, whether it comes from the chat agent or the voice agent, goes through
`ToolRegistry.execute`, which applies the same cross-cutting concerns:

  validate args -> check preconditions (identity) -> run with timeout -> retry transient
  failures -> mask PII in logs/traces -> emit live step events -> return a structured
  result the LLM can reason about (never a raw stack trace).

Tool *errors* are returned to the model as data (`{"ok": false, "error": {...}}`) with
an `instruction` telling it what to do next. That's what lets the agent recover in
conversation ("our account system is unavailable right now; I've opened a ticket")
instead of the whole turn crashing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.agent.state import Session
from app.agent.turn import TurnState
from app.config import Settings
from app.faults import InjectedToolFailure
from app.llm.base import ToolCall, ToolSpec
from app.observability.tracer import span
from app.schemas.events import Emit
from app.tools.policy import PolicyRules

log = logging.getLogger(__name__)


class ToolError(Exception):
    """A tool-level failure with guidance for the model."""

    def __init__(self, code: str, message: str, retryable: bool = False, instruction: str = ""):
        super().__init__(message)
        self.code, self.retryable, self.instruction = code, retryable, instruction

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "retryable": self.retryable,
                "instruction": self.instruction}


@dataclass
class ToolContext:
    session: Session
    kb: Any                      # KnowledgeBase (Any to keep this module import-light)
    rules: PolicyRules
    settings: Settings
    emit: Emit
    turn: TurnState


Handler = Callable[[ToolContext, dict], Awaitable[tuple[dict, str]]]


@dataclass
class Tool:
    name: str
    label: str                   # UI status text, e.g. "Verifying identity"
    description: str
    parameters: dict
    handler: Handler
    requires_identity: bool = False
    uses_crm: bool = False       # subject to CRM fault injection / outages
    sensitive_args: tuple[str, ...] = ()

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


@dataclass
class ToolOutcome:
    call_id: str
    name: str
    ok: bool
    data: dict = field(default_factory=dict)
    error: dict | None = None
    summary: str = ""
    attempts: int = 1
    duration_ms: float = 0.0

    def for_model(self) -> dict:
        return {"ok": True, **self.data} if self.ok else {"ok": False, "error": self.error}


def _mask(args: dict, sensitive: tuple[str, ...]) -> dict:
    return {k: ("••••" if k in sensitive else v) for k, v in args.items()}


_JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "object": dict, "array": list}


def coerce_args(schema: dict, args: dict) -> dict:
    """Forgive harmless type slips (e.g. amount "12.00" as a string) before validating."""
    out = dict(args)
    for k, p in schema.get("properties", {}).items():
        v = out.get(k)
        if p.get("type") == "number" and isinstance(v, str):
            try:
                out[k] = float(v.replace("$", "").replace(",", ""))
            except ValueError:
                pass
        elif p.get("type") == "string" and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = str(v)
    return out


def validate_args(schema: dict, args: dict) -> list[str]:
    """Minimal JSON-schema check (required, type, enum). LLMs occasionally send wrong types;
    catching it here gives the model a precise error to self-correct from."""
    problems = []
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if args.get(req) in (None, ""):
            problems.append(f"missing required argument '{req}'")
    for k, v in args.items():
        p = props.get(k)
        if p is None:
            continue
        t = _JSON_TYPES.get(p.get("type", ""))
        if t and not isinstance(v, t) or (p.get("type") == "number" and isinstance(v, bool)):
            problems.append(f"argument '{k}' should be {p.get('type')}")
        if "enum" in p and v not in p["enum"]:
            problems.append(f"argument '{k}' must be one of {p['enum']}")
    return problems


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self.tools.values()]

    def label(self, name: str) -> str:
        t = self.tools.get(name)
        return t.label if t else name

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutcome:
        tool = self.tools.get(call.name)
        t0 = time.perf_counter()
        if tool is None:
            return ToolOutcome(call.id, call.name, False, error=ToolError(
                "UNKNOWN_TOOL", f"No tool named {call.name}", instruction="Use only the provided tools.").to_dict())

        args = coerce_args(tool.parameters, dict(call.args or {}))
        masked = _mask(args, tool.sensitive_args)
        await ctx.emit("step", {"step_id": call.id, "tool": tool.name, "label": tool.label, "status": "started",
                                "args": masked})
        outcome = ToolOutcome(call.id, tool.name, False)

        async with span(f"tool.{tool.name}", "tool", args=masked) as s:
            try:
                problems = validate_args(tool.parameters, args)
                if problems:
                    raise ToolError("INVALID_ARGUMENTS", "; ".join(problems),
                                    instruction="Fix the arguments and call the tool again.")
                if tool.requires_identity and not ctx.session.customer_id:
                    raise ToolError("IDENTITY_REQUIRED", "Customer identity is not verified.",
                                    instruction="Ask the customer for their account email and 4-digit support "
                                                "PIN, then call verify_identity.")
                data, summary = await self._run_with_retries(tool, args, ctx, call.id, outcome)
                outcome.ok, outcome.data, outcome.summary = True, data, summary
            except ToolError as e:
                outcome.error = e.to_dict()
                outcome.summary = e.code
                s.status = "error"
            outcome.duration_ms = (time.perf_counter() - t0) * 1000
            s.set(ok=outcome.ok, attempts=outcome.attempts, summary=outcome.summary,
                  result=outcome.data if outcome.ok else outcome.error)

        await ctx.emit("step", {"step_id": call.id, "tool": tool.name, "label": tool.label,
                                "status": "completed" if outcome.ok else "failed",
                                "summary": outcome.summary, "duration_ms": round(outcome.duration_ms, 1),
                                "attempts": outcome.attempts, "error": outcome.error})
        return outcome

    async def _run_with_retries(self, tool: Tool, args: dict, ctx: ToolContext, step_id: str,
                                outcome: ToolOutcome) -> tuple[dict, str]:
        max_attempts = 1 + ctx.settings.tool_max_retries
        last: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            outcome.attempts = attempt
            try:
                if tool.uses_crm and (ctx.session.faults.crm_outage or ctx.session.faults.take("tool_flaky")):
                    raise InjectedToolFailure()
                return await asyncio.wait_for(tool.handler(ctx, args), timeout=ctx.settings.tool_timeout_s)
            except ToolError as e:
                if not e.retryable:
                    raise
                last = e
            except (asyncio.TimeoutError, InjectedToolFailure, ConnectionError) as e:
                last = e
            if attempt < max_attempts:
                delay = 0.4 * attempt
                await ctx.emit("step", {"step_id": step_id, "tool": tool.name, "label": tool.label,
                                        "status": "retrying", "attempt": attempt + 1,
                                        "error": f"{type(last).__name__}: {last}"[:160]})
                await asyncio.sleep(delay)
        log.warning("tool %s failed after %d attempts: %s", tool.name, max_attempts, last)
        raise ToolError(
            "UPSTREAM_UNAVAILABLE", f"{tool.label} failed: the account system is temporarily unavailable.",
            retryable=False,
            instruction="Apologise, tell the customer the account system is temporarily unavailable, and offer to "
                        "create an escalation ticket so a human follows up. Do not invent account details.")
