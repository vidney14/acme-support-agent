"""The chat agent: a small, explicit state machine around an LLM tool-calling loop.

    ┌──────────┐  tool calls  ┌──────┐
    │  PLAN    │─────────────▶│ ACT  │  (tools run through ToolRegistry: validation,
    │ (LLM)    │◀─────────────│      │   identity check, retries, policy engine)
    └────┬─────┘   results    └──────┘
         │ final text
         ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  GUARD   │────▶│  VERIFY  │────▶│  COMMIT  │
    │ (turn    │     │ (citation│     │ (memory, │
    │  guards) │     │  check)  │     │  trace)  │
    └──────────┘     └──────────┘     └──────────┘

    step budget exhausted ──▶ ESCALATE (ticket + honest message)

The LLM decides *what* to do (which tools, in what order, what to say). The code decides
what is *allowed* (preconditions, policy, guards) and guarantees the outcome. Why not
LangGraph? The graph here has five nodes; a hand-written loop keeps every transition,
retry and event explicit and debuggable, with no framework magic to explain away.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from app.agent.grounding import check_grounding
from app.agent.history import transcript_items
from app.agent.prompts import build_system_prompt
from app.agent.state import Session
from app.agent.turn import TurnState, enforce_turn_guards
from app.config import Settings
from app.faults import InjectedRateLimit, InjectedTimeout
from app.llm.base import Item, LLMProvider, ModelTurn, StreamDone, TextDelta, ToolResult, UserText
from app.llm.pricing import text_cost
from app.llm.resilience import ErrorKind, ProviderError, RetryPolicy, classify
from app.observability.tracer import Trace, bind_trace, span, unbind_trace
from app.schemas.events import Emit
from app.tools.policy import PolicyRules
from app.tools.registry import ToolContext, ToolRegistry
from app.tools.support_tools import open_ticket

log = logging.getLogger(__name__)


class Phase(str, Enum):
    PLAN = "plan"
    ACT = "act"
    GUARD = "guard"
    VERIFY = "verify"
    COMMIT = "commit"
    ESCALATE = "escalate"


PHASE_LABELS = {
    Phase.PLAN: "Thinking",
    Phase.ACT: "Running tools",
    Phase.GUARD: "Checking guardrails",
    Phase.VERIFY: "Verifying answer grounding",
    Phase.ESCALATE: "Escalating to a human",
}


class AgentFailure(Exception):
    def __init__(self, err: ProviderError):
        super().__init__(str(err))
        self.err = err


@dataclass
class TurnResult:
    turn_id: str
    text: str = ""
    ok: bool = True
    grounding: dict = field(default_factory=dict)
    tickets: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    error: dict | None = None


def render_items(items: list[Item]) -> str:
    """Human-readable prompt snapshot for the observability drawer."""
    out = []
    for it in items:
        name = type(it).__name__
        if isinstance(it, ModelTurn):
            calls = "; ".join(f"{c.name}({json.dumps(c.args)[:200]})" for c in it.tool_calls)
            out.append(f"[model] {it.text}" + (f"\n  → tool calls: {calls}" if calls else ""))
        elif isinstance(it, ToolResult):
            out.append(f"[tool:{it.name}] {json.dumps(it.result)[:600]}")
        else:
            out.append(f"[{'user' if name == 'UserText' else 'assistant'}] {it.text}")
    return "\n".join(out)


class AgentOrchestrator:
    def __init__(self, provider: LLMProvider, registry: ToolRegistry, kb, rules: PolicyRules, settings: Settings):
        self.provider = provider
        self.registry = registry
        self.kb = kb
        self.rules = rules
        self.settings = settings
        self.retry_policy = RetryPolicy(max_retries=settings.llm_max_retries)

    # ------------------------------------------------------------------ public
    async def run_turn(self, session: Session, user_text: str, emit: Emit, retry: bool = False) -> TurnResult:
        async with session.turn_lock:  # one turn at a time per session (chat + voice share it)
            return await self._run_turn(session, user_text.strip(), emit, retry)

    # ------------------------------------------------------------------ internals
    async def _run_turn(self, session: Session, user_text: str, emit: Emit, retry: bool) -> TurnResult:
        turn_id = uuid.uuid4().hex[:10]
        trace = Trace(session_id=session.session_id, modality="text", user_input=user_text)
        token = bind_trace(trace)
        turn = TurnState(turn_id=turn_id, modality="text", user_text=user_text)
        ctx = ToolContext(session, self.kb, self.rules, self.settings, emit, turn)
        result = TurnResult(turn_id=turn_id)

        last = session.messages[-1] if session.messages else None
        is_retry = retry and last is not None and last.role == "user" and last.text == user_text
        history = transcript_items(session.messages[:-1] if is_retry else session.messages)
        if not is_retry:
            session.add_message("user", user_text, "text", turn_id)
        items: list[Item] = [*history, UserText(user_text)]

        await emit("turn_start", {"turn_id": turn_id, "modality": "text", "trace_id": trace.trace_id})
        try:
            async with span("agent.turn", "agent", turn_id=turn_id) as root:
                root.set(system_prompt=build_system_prompt(session, "text", session.crm.today))
                answer = await self._loop(ctx, items, emit, trace)

                await self._phase(emit, Phase.GUARD)
                for t in await enforce_turn_guards(ctx):
                    note = (f"\n\nI've opened ticket **{t['ticket_id']}** so a human can follow up "
                            f"(expected response: {t['response_target']}).")
                    answer += note
                    await emit("token", {"text": note})
                    await emit("notice", {"kind": "guardrail", "message":
                                          f"Guardrail auto-escalated: {t['ticket_id']} ({t['summary'][:80]})"})

                await self._phase(emit, Phase.VERIFY)
                grounding = check_grounding(answer, turn)
                await emit("guardrail", {"kind": "grounding", **grounding.to_dict()})
                root.set(grounding=grounding.to_dict(), guard_actions=turn.guard_actions)

            sources = {cid: self._source_label(cid) for cid in grounding.citations}
            session.add_message("assistant", answer, "text", turn_id, grounding=grounding.to_dict(),
                                sources=sources, tickets=turn.tickets_created)
            result.text, result.grounding = answer, grounding.to_dict()
            result.tickets = [t for t in session.tickets if t["ticket_id"] in turn.tickets_created]
            trace.finish("ok")
            await emit("message", {"turn_id": turn_id, "role": "assistant", "text": answer,
                                   "grounding": grounding.to_dict(), "sources": sources})
        except AgentFailure as f:
            trace.finish("error")
            result.ok, result.error = False, f.err.to_dict()
            await emit("error", {"turn_id": turn_id, "recoverable": True, **f.err.to_dict(),
                                 "message": self._friendly_error(f.err)})
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the client
            log.exception("turn crashed")
            trace.finish("error")
            result.ok = False
            result.error = {"kind": "internal", "message": str(exc)[:200]}
            await emit("error", {"turn_id": turn_id, "recoverable": True, "kind": "internal",
                                 "message": "Something went wrong on our side. Please try again."})
        finally:
            unbind_trace(token)
            session.add_trace(trace)
            result.trace = trace.to_dict()
            await emit("trace", result.trace)
            await emit("session", session.snapshot())
            await emit("done", {"turn_id": turn_id, "ok": result.ok})
        return result

    async def _loop(self, ctx: ToolContext, items: list[Item], emit: Emit, trace: Trace) -> str:
        answer_parts: list[str] = []
        for step in range(self.settings.max_agent_steps):
            await self._phase(emit, Phase.PLAN, step=step)
            system = build_system_prompt(ctx.session, "text", ctx.session.crm.today)  # facts change as tools run
            sep = "\n\n" if answer_parts and answer_parts[-1].strip() else ""
            model_turn = await self._call_model(ctx, system, items, emit, trace, prefix=sep)
            items.append(model_turn)
            if model_turn.text.strip():
                answer_parts.append(sep + model_turn.text)

            if not model_turn.tool_calls:
                answer = "".join(answer_parts).strip()
                if not answer:
                    answer = "Sorry, I couldn't put together a reply. Could you rephrase your question?"
                    await emit("token", {"text": answer})
                return answer

            await self._phase(emit, Phase.ACT, tools=[c.name for c in model_turn.tool_calls])
            # Sequential on purpose: calls in one batch can depend on each other
            # (verify_identity must finish before get_account_details).
            for call in model_turn.tool_calls:
                outcome = await self.registry.execute(call, ctx)
                items.append(ToolResult(call.id, call.name, outcome.for_model(), call.provider_id))

        # Step budget exhausted: the model is looping. Stop, be honest, hand off.
        await self._phase(emit, Phase.ESCALATE)
        t = await open_ticket(ctx, summary=f"Agent could not resolve within {self.settings.max_agent_steps} steps: "
                                           f"{ctx.turn.user_text[:200]}", category="other", priority="P3",
                              attempted_steps="Multiple tool calls without convergence.", source="step_budget")
        msg = (f"I'm sorry, I wasn't able to fully resolve this. I've opened ticket {t['ticket_id']} and a "
               f"human will follow up (expected response: {t['response_target']}).")
        await emit("token", {"text": ("\n\n" if answer_parts else "") + msg})
        return ("".join(answer_parts) + "\n\n" + msg).strip()

    async def _call_model(self, ctx: ToolContext, system: str, items: list[Item], emit: Emit, trace: Trace,
                          prefix: str = "") -> ModelTurn:
        """One model call with retry -> fallback. Streams text deltas to the UI as they arrive."""
        s = self.settings
        plan = [(s.chat_model, s.chat_thinking_level)]
        if s.fallback_model and s.fallback_model != s.chat_model:
            plan.append((s.fallback_model, s.fallback_thinking_level))
        faults = ctx.session.faults
        last_err: ProviderError | None = None

        for idx, (model, thinking) in enumerate(plan):
            attempt = 0
            while True:
                attempt += 1
                emitted = ""
                try:
                    async with span("llm.generate", "llm", model=model, attempt=attempt,
                                    role="primary" if idx == 0 else "fallback") as sp:
                        if idx == 0 and faults.take("llm_rate_limit"):
                            raise InjectedRateLimit()
                        if faults.take("llm_timeout"):
                            raise InjectedTimeout()
                        t0 = time.perf_counter()
                        done: StreamDone | None = None
                        async for ev in self.provider.stream(
                                model=model, system=system, items=items, tools=self.registry.specs(),
                                thinking_level=thinking, first_byte_timeout_s=s.llm_timeout_s,
                                idle_timeout_s=s.llm_timeout_s):
                            if isinstance(ev, TextDelta):
                                if not emitted:
                                    sp.set(ttft_ms=round((time.perf_counter() - t0) * 1000, 1))
                                    trace.mark_first_token()
                                    if prefix:
                                        await emit("token", {"text": prefix})
                                        emitted += prefix
                                emitted += ev.text
                                await emit("token", {"text": ev.text})
                            else:
                                done = ev
                        if done is None:
                            raise ProviderError(ErrorKind.SERVER, "Model stream ended early", True)
                        u = done.usage
                        sp.set(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                               thinking_tokens=u.thinking_tokens, finish_reason=done.finish_reason,
                               cost_usd=round(text_cost(model, u.input_tokens, u.output_tokens), 6),
                               prompt=render_items(items[-4:]),
                               output=render_items([done.turn]))
                        return done.turn
                except Exception as exc:  # noqa: BLE001
                    err = classify(exc)
                    last_err = err
                    if emitted:  # partial text already on screen: tell the UI to discard it
                        await emit("draft_reset", {"chars": len(emitted), "reason": err.kind.value})
                    delay = self.retry_policy.delay_for(attempt, err)
                    if delay is None:
                        break
                    await emit("notice", {"kind": "retry", "message":
                                          f"{model}: {err.kind.value.replace('_', ' ')}. Retrying in {delay:.1f}s "
                                          f"(attempt {attempt + 1})"})
                    await asyncio.sleep(delay)
            if idx + 1 < len(plan):
                await emit("notice", {"kind": "fallback", "message":
                                      f"{model} unavailable ({last_err.kind.value.replace('_', ' ')}). "
                                      f"Switched to fallback model {plan[idx + 1][0]}."})
        raise AgentFailure(last_err or ProviderError(ErrorKind.UNAVAILABLE, "No model available", True))

    async def _phase(self, emit: Emit, phase: Phase, **extra) -> None:
        await emit("status", {"phase": phase.value, "label": PHASE_LABELS.get(phase, phase.value), **extra})

    def _source_label(self, chunk_id: str) -> str:
        if self.kb and self.kb.retriever:
            c = self.kb.retriever._by_id.get(chunk_id)
            if c:
                return c.breadcrumb
        return chunk_id

    @staticmethod
    def _friendly_error(err: ProviderError) -> str:
        return {
            "rate_limit": "Our AI service is getting more requests than it can handle right now. Please retry in a moment.",
            "timeout": "The AI service took too long to respond. Please retry.",
            "server_error": "The AI service had a temporary problem. Please retry.",
            "unavailable": "We couldn't reach the AI service. Please check your connection and retry.",
        }.get(err.kind.value, "Something went wrong generating a reply. Please retry.")
