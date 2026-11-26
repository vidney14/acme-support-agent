"""Lightweight tracing: one Trace per user turn, made of nested Spans.

Shape is deliberately close to OpenTelemetry (trace -> spans with parent ids, kinds,
attributes) so it could be exported to Langfuse/OTel later without changing callers.

A ContextVar holds the "current trace", so any layer (retriever, tool, verifier) can
record a span without threading a tracer object through every function signature.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_current_trace: ContextVar["Trace | None"] = ContextVar("current_trace", default=None)
_current_span: ContextVar["Span | None"] = ContextVar("current_span", default=None)

MAX_ATTR_CHARS = 4000  # keep prompt snapshots bounded


def _clip(v: Any) -> Any:
    if isinstance(v, str) and len(v) > MAX_ATTR_CHARS:
        return v[:MAX_ATTR_CHARS] + f"… [+{len(v) - MAX_ATTR_CHARS} chars]"
    return v


@dataclass
class Span:
    name: str
    kind: str                      # llm | tool | retrieval | guardrail | embedding | voice | agent
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    start_ms: float = 0.0          # relative to trace start
    duration_ms: float | None = None
    status: str = "ok"
    attrs: dict[str, Any] = field(default_factory=dict)

    def set(self, **attrs: Any) -> None:
        self.attrs.update({k: _clip(v) for k, v in attrs.items()})

    def to_dict(self) -> dict:
        return {"span_id": self.span_id, "parent_id": self.parent_id, "name": self.name, "kind": self.kind,
                "start_ms": round(self.start_ms, 1),
                "duration_ms": None if self.duration_ms is None else round(self.duration_ms, 1),
                "status": self.status, "attrs": self.attrs}


@dataclass
class Trace:
    session_id: str
    modality: str                  # text | voice
    user_input: str = ""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started_at: float = field(default_factory=time.time)
    spans: list[Span] = field(default_factory=list)
    ttft_ms: float | None = None   # time to first streamed token / first audio byte
    total_ms: float | None = None
    status: str = "ok"
    _t0: float = field(default_factory=time.perf_counter)

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def mark_first_token(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = self.now_ms()

    def finish(self, status: str = "ok") -> None:
        self.total_ms = self.now_ms()
        self.status = status

    def add_span(self, span: Span) -> Span:
        self.spans.append(span)
        return span

    @property
    def totals(self) -> dict:
        tin = sum(int(s.attrs.get("input_tokens", 0) or 0) for s in self.spans)
        tout = sum(int(s.attrs.get("output_tokens", 0) or 0) for s in self.spans)
        cost = sum(float(s.attrs.get("cost_usd", 0) or 0) for s in self.spans)
        llm_calls = sum(1 for s in self.spans if s.kind in ("llm", "voice"))
        tool_calls = sum(1 for s in self.spans if s.kind == "tool")
        return {"input_tokens": tin, "output_tokens": tout, "cost_usd": round(cost, 6),
                "llm_calls": llm_calls, "tool_calls": tool_calls}

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "session_id": self.session_id, "modality": self.modality,
                "user_input": _clip(self.user_input), "started_at": self.started_at,
                "ttft_ms": None if self.ttft_ms is None else round(self.ttft_ms, 1),
                "total_ms": None if self.total_ms is None else round(self.total_ms, 1),
                "status": self.status, "totals": self.totals, "spans": [s.to_dict() for s in self.spans]}


def current_trace() -> Trace | None:
    return _current_trace.get()


def bind_trace(trace: Trace | None):
    return _current_trace.set(trace)


def unbind_trace(token) -> None:
    _current_trace.reset(token)


@asynccontextmanager
async def span(name: str, kind: str, **attrs: Any):
    """Record a span on the current trace (no-op if there is none)."""
    trace = _current_trace.get()
    if trace is None:
        yield Span(name=name, kind=kind)  # detached span, discarded
        return
    parent = _current_span.get()
    s = trace.add_span(Span(name=name, kind=kind, parent_id=parent.span_id if parent else None,
                            start_ms=trace.now_ms()))
    s.set(**attrs)
    token = _current_span.set(s)
    try:
        yield s
    except BaseException as exc:
        s.status = "error"
        s.set(error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    finally:
        s.duration_ms = trace.now_ms() - s.start_ms
        _current_span.reset(token)
