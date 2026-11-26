"""Offline providers.

- `ScriptedProvider`: replays a fixed list of model turns (or raises given exceptions).
  Used by tests to drive the orchestrator through exact scenarios.
- `RuleBasedProvider`: a tiny heuristic "model" that plans the same tool chains a real
  LLM would for the demo scenarios. Set LLM_PROVIDER=fake to run the whole stack (UI,
  SSE, tools, RAG, guardrails, tracing) with no API key or network access, e.g. in CI.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import AsyncIterator

from app.llm.base import (AssistantText, Item, ModelTurn, StreamDone, StreamItem, TextDelta, ToolCall, ToolResult,
                          ToolSpec, Usage, UserText)


def _words(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text)


async def _stream_turn(turn: ModelTurn, model: str, items: list[Item], delay: float = 0.0) -> AsyncIterator[StreamItem]:
    for w in _words(turn.text):
        if delay:
            await asyncio.sleep(delay)
        yield TextDelta(w)
    in_tok = sum(len(getattr(i, "text", "") or str(getattr(i, "result", ""))) for i in items) // 4
    yield StreamDone(turn=turn, usage=Usage(input_tokens=in_tok, output_tokens=max(1, len(turn.text) // 4)),
                     model=model, finish_reason="STOP")


class ScriptedProvider:
    name = "scripted"

    def __init__(self, script: list, json_responses: list | None = None):
        """script items: ModelTurn | Exception | callable(items) -> ModelTurn"""
        self.script = list(script)
        self.json_responses = list(json_responses or [])
        self.calls: list[dict] = []

    async def stream(self, *, model: str, system: str, items: list[Item], tools: list[ToolSpec],
                     thinking_level: str | None, first_byte_timeout_s: float,
                     idle_timeout_s: float) -> AsyncIterator[StreamItem]:
        self.calls.append({"model": model, "system": system, "items": list(items)})
        if not self.script:
            raise AssertionError("ScriptedProvider ran out of script")
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, tuple) and nxt[0] == "partial":  # ("partial", text, exc): stream some text, then fail
            for w in _words(nxt[1]):
                yield TextDelta(w)
            raise nxt[2]
        turn = nxt(items) if callable(nxt) else nxt
        turn = ModelTurn(text=turn.text, tool_calls=[ToolCall(c.id or uuid.uuid4().hex[:6], c.name, c.args)
                                                     for c in turn.tool_calls])
        async for ev in _stream_turn(turn, model, items):
            yield ev

    async def generate_json(self, *, model, system, prompt, schema, thinking_level, timeout_s):
        if not self.json_responses:
            return {"answerable": "yes", "reason": "scripted"}, Usage(10, 5)
        nxt = self.json_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt, Usage(10, 5)


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PIN = re.compile(r"\b(\d{4})\b")
FIELD_WORDS = {"phone number": "phone", "phone": "phone", "contact channel": "preferred_contact_channel",
               "preferred contact channel": "preferred_contact_channel", "timezone": "timezone",
               "time zone": "timezone", "display name": "display_name", "name": "display_name",
               "marketing emails": "marketing_emails"}
UPDATE = re.compile(r"(change|switch|update|set)\s+my\s+(preferred contact channel|contact channel|phone number|"
                    r"phone|time zone|timezone|display name|name|marketing emails)\s+(?:to\s+)?(.+)$")
REFUND_REQUEST = re.compile(
    r"(charged twice|double[- ]charged|duplicate charge|(get|want|need|request|issue|give me) (a |my )?refund|"
    r"refund (me|it|this|that|my|the|inv-)|can i (get|have) (a |my )?(money|refund))")


def _call(name: str, **args) -> ModelTurn:
    return ModelTurn(text="", tool_calls=[ToolCall(uuid.uuid4().hex[:8], name, args)])


class RuleBasedProvider:
    """Deterministic stand-in for an LLM. Not smart, just enough to demo every flow offline."""

    name = "fake"

    def __init__(self, token_delay_s: float = 0.015):
        self.delay = token_delay_s

    async def stream(self, *, model: str, system: str, items: list[Item], tools: list[ToolSpec],
                     thinking_level: str | None, first_byte_timeout_s: float,
                     idle_timeout_s: float) -> AsyncIterator[StreamItem]:
        await asyncio.sleep(0.25)  # pretend to think
        turn = self._decide(system, items)
        async for ev in _stream_turn(turn, model, items, self.delay):
            yield ev

    async def generate_json(self, *, model, system, prompt, schema, thinking_level, timeout_s):
        await asyncio.sleep(0.1)
        q = prompt.split("PASSAGES:")[0].lower()
        p = prompt.split("PASSAGES:")[-1].lower()
        overlap = len({w for w in re.findall(r"[a-z]{5,}", q)} & {w for w in re.findall(r"[a-z]{5,}", p)})
        verdict = "yes" if overlap >= 3 else "partial" if overlap >= 1 else "no"
        return {"answerable": verdict, "reason": f"{overlap} key terms overlap (offline heuristic)"}, Usage(200, 20)

    # ------------------------------------------------------------------ heuristics
    def _decide(self, system: str, items: list[Item]) -> ModelTurn:
        last_user_idx = max(i for i, it in enumerate(items) if isinstance(it, UserText))
        msg = items[last_user_idx].text
        low = msg.lower()
        done = [it for it in items[last_user_idx + 1:] if isinstance(it, ToolResult)]
        results = {r.name: r.result for r in done}
        earlier = " ".join(it.text.lower() for it in items[:last_user_idx] if isinstance(it, UserText))
        verified = "Identity VERIFIED" in system
        is_question = bool(re.match(r"^(how|what|when|why|do|does|is|are|can refunds|which)\b", low))
        refund_request = REFUND_REQUEST.search(low) is not None
        wants_refund = (refund_request and not is_question) or (
            REFUND_REQUEST.search(earlier) is not None and EMAIL.search(msg) is not None)
        wants_human = any(k in low for k in ("human", "real person", "agent please", "speak to someone"))

        if "create_escalation_ticket" in results:
            r = results["create_escalation_ticket"]
            return ModelTurn(f"I've opened ticket {r.get('ticket_id')} with our team. A human will follow up "
                             f"(expected response: {r.get('human_response_target')}).")
        if wants_human and not done:
            return _call("create_escalation_ticket", summary=f"Customer asked for a human: {msg[:150]}",
                         category="other", priority="P3", attempted_steps="None yet")

        # identity
        if "verify_identity" in results:
            r = results["verify_identity"]
            if not r.get("ok", True) or not r.get("verified"):
                if r.get("locked"):
                    return _call("create_escalation_ticket", summary="Identity verification locked",
                                 category="account", priority="P2", attempted_steps="3 failed PIN attempts")
                return ModelTurn("That email and PIN didn't match our records. Could you double-check them? "
                                 f"({r.get('attempts_remaining', '?')} attempts left)")
            if not wants_refund and "search_knowledge_base" not in results:
                follow = self._follow_up(msg, results)
                if follow is not None:
                    return follow
                return ModelTurn(f"Thanks {r['name']}, you're verified. How can I help with your "
                                 f"{r['plan'].title()} account today?")
        elif EMAIL.search(msg) and PIN.search(msg) and not verified:
            return _call("verify_identity", email=EMAIL.search(msg).group(0), pin=PIN.search(msg).group(1))

        if verified and not wants_refund and "verify_identity" not in results:
            follow = self._follow_up(msg, results)
            if follow is not None and not (follow.tool_calls and follow.tool_calls[0].name == "search_knowledge_base"):
                return follow

        if wants_refund:
            if not verified and "verify_identity" not in results:
                return ModelTurn("I can help with that refund. First I need to verify your identity: what's the "
                                 "email on your AcmeCloud account and your 4-digit support PIN?")
            if "get_account_details" not in results:
                return _call("get_account_details")
            acct = results["get_account_details"]
            if not acct.get("ok", True):
                return ModelTurn("I'm sorry, our account system is temporarily unavailable, so I can't look up "
                                 "your invoices right now. Would you like me to open a ticket?")
            if "search_knowledge_base" not in results:
                return _call("search_knowledge_base", query=f"refund policy {acct['plan']} {acct['billing_cycle']}")
            if "issue_refund" not in results:
                inv = self._pick_invoice(acct, msg)
                if inv is None:
                    return ModelTurn("I couldn't find a refundable invoice on your account. Which invoice do you mean?")
                return _call("issue_refund", invoice_id=inv["invoice_id"], amount=inv["amount_usd"],
                             reason="duplicate charge" if inv.get("duplicate_of") else "customer request")
            r = results["issue_refund"]
            cites = " ".join(f"[{p['cite']}]" for p in results["search_knowledge_base"].get("passages", [])[:1])
            if r.get("status") == "approved":
                return ModelTurn(f"Done! I've refunded ${r['amount_usd']:.2f} for {r['invoice_id']} "
                                 f"(reference {r['refund_id']}). It goes back to your original payment method "
                                 f"within 5-10 business days {cites}.")
            reasons = "; ".join(x["message"] for x in r["policy"]["rules"])
            if r.get("status") == "needs_human_approval":
                return _call("create_escalation_ticket", summary=f"Refund needs approval: {reasons}",
                             category="billing", priority="P2", attempted_steps="Policy engine evaluated refund")
            return ModelTurn(f"I'm sorry, I can't issue that refund: {reasons} {cites}")

        if "search_knowledge_base" in results:
            r = results["search_knowledge_base"]
            if not r.get("ok", True):
                return ModelTurn("I can't reach our knowledge base right now. Want me to open a ticket?")
            if r["decision"] == "escalate":
                return _call("create_escalation_ticket", summary=f"Not covered by KB: {msg[:150]}",
                             category="other", priority="P3", attempted_steps="Searched knowledge base")
            p = r["passages"][0]
            first = re.split(r"(?<=[.!?])\s", p["text"].replace("\n", " "))[0:2]
            hedge = "Here's what I can confirm: " if r["decision"] == "answer_with_caution" else ""
            tail = " I couldn't confirm the rest, so I can connect you with a human if you'd like." if hedge else ""
            return ModelTurn(f"{hedge}{' '.join(first)} [{p['cite']}]{tail}")

        if len(low.split()) <= 3 and any(g in low for g in ("hi", "hello", "hey", "thanks", "thank you")):
            return ModelTurn("Hi! I'm Acme Assist. I can help with billing, refunds, sync problems, sign-in and "
                             "account changes. What can I do for you?")
        return _call("search_knowledge_base", query=msg[:200])

    @staticmethod
    def _follow_up(msg: str, results: dict) -> ModelTurn | None:
        """After verification: handle the rest of the request (field update or a KB question)."""
        low = msg.lower()
        if "update_account_field" in results:
            r = results["update_account_field"]
            if not r.get("ok", True):
                return ModelTurn(f"I couldn't make that change: {r['error']['message']}")
            return ModelTurn(f"Done: your {r['field'].replace('_', ' ')} is now {r['new_value']}. "
                             "You'll get a confirmation email.")
        m = UPDATE.search(low)
        if m:
            field = FIELD_WORDS[m.group(2)]
            value = m.group(3).strip(" .")
            if field == "marketing_emails":
                value = "false" if "out" in value or "stop" in value else "true"
            return _call("update_account_field", field=field, value=value)
        if "search_knowledge_base" in results:
            return None
        rest = EMAIL.sub("", PIN.sub("", msg))
        rest = re.sub(r"\b(my|email|is|and|pin|support|it's|hi|hello)\b[,.:]?", " ", rest, flags=re.I)
        if len(re.findall(r"[a-zA-Z]{3,}", rest)) >= 4:
            return _call("search_knowledge_base", query=rest.strip(" ,.-")[:200])
        return None

    @staticmethod
    def _pick_invoice(acct: dict, msg: str) -> dict | None:
        invs = [i for i in acct["invoices"] if i["status"] in ("paid", "partially_refunded")]
        named = re.search(r"INV-\d+", msg.upper())
        if named:
            return next((i for i in acct["invoices"] if i["invoice_id"] == named.group(0)), None)
        dup = next((i for i in invs if i.get("duplicate_of")), None)
        if dup:
            return dup
        if "annual" in msg.lower():
            return next((i for i in invs if i.get("billing_cycle") == "annual"), invs[0] if invs else None)
        return min(invs, key=lambda i: i["days_since_charge"]) if invs else None
