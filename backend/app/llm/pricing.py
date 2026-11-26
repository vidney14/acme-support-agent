"""Token pricing used for the cost estimate in the observability drawer.

Prices are USD per 1M tokens, paid tier, from ai.google.dev/gemini-api/docs/pricing
(checked Nov 2026; Flash prices rise on 2027-01-01). On the free tier the real cost is
$0; we still show what the traffic *would* cost on the paid tier, which is the number
that matters when you ask "can we afford to run this in production?".
"""

from __future__ import annotations

PRICES_PER_M: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.8-live": {"input_text": 0.75, "input_audio": 3.00, "output_text": 4.50, "output_audio": 12.00},
    "gemini-embedding-001": {"input": 0.15},
}
_DEFAULT = {"input": 0.75, "output": 3.75}


def text_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES_PER_M.get(model, _DEFAULT)
    return (input_tokens * p.get("input", 0) + output_tokens * p.get("output", 0)) / 1_000_000


def live_cost(model: str, *, in_text: int = 0, in_audio: int = 0, out_text: int = 0, out_audio: int = 0) -> float:
    p = PRICES_PER_M.get(model, PRICES_PER_M["gemini-3.8-live"])
    return (in_text * p["input_text"] + in_audio * p["input_audio"]
            + out_text * p["output_text"] + out_audio * p["output_audio"]) / 1_000_000
