"""Events streamed to the UI. The SAME event vocabulary is used by the SSE chat stream
and the voice WebSocket, so the frontend renders tool steps, guardrails and traces
identically no matter which modality produced them.

| type          | when                                                        |
|---------------|-------------------------------------------------------------|
| turn_start    | a user turn begins (turn_id, modality)                      |
| status        | orchestrator phase changed (plan / act / guard / verify)    |
| step          | a tool starts / retries / completes / fails                 |
| token         | assistant text delta (chat)                                 |
| draft_reset   | discard the streamed draft (we are retrying the model call) |
| notice        | non-fatal info: model fallback, retry, guard intervention   |
| guardrail     | confidence verdict or grounding check result                |
| escalation    | a ticket was created (session is now in escalated state)    |
| message       | a finished message (assistant text + citations + grounding) |
| trace         | the finished trace for the observability drawer             |
| session       | session state snapshot (verified customer, status, ...)     |
| error         | turn failed; `recoverable` tells the UI to offer Retry      |
| done          | turn finished                                               |
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]


async def noop_emit(_type: str, _data: dict[str, Any]) -> None:
    return None
