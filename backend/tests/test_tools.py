"""Policy engine + tool executor: the deterministic layer the LLM can't talk its way past."""

from __future__ import annotations

import pytest

from app.agent.turn import TurnState, enforce_turn_guards
from app.llm.base import ToolCall
from app.tools.registry import ToolContext
from tests.conftest import Recorder


def ctx_for(container, session, emit=None):
    return ToolContext(session, container.kb, container.rules, container.settings, emit or Recorder(),
                       TurnState("t", "text"))


async def run(container, session, name, emit=None, **args):
    ctx = ctx_for(container, session, emit)
    out = await container.registry.execute(ToolCall("c1", name, args), ctx)
    return out, ctx


async def verified(container, email, pin):
    s = container.sessions.get_or_create(None)
    out, _ = await run(container, s, "verify_identity", email=email, pin=pin)
    assert out.ok and out.data["verified"]
    return s


@pytest.mark.parametrize("email,pin,invoice,amount,status,codes", [
    ("priya@example.com", "4321", "INV-20932", 12, "approved", []),                         # duplicate charge
    ("priya@example.com", "4321", "INV-20118", 12, "denied", ["OUTSIDE_REFUND_WINDOW"]),     # 33 days > 14
    ("marcus@example.com", "8812", "INV-30588", 40, "approved", []),                         # under $200
    ("marcus@example.com", "8812", "INV-30410", 2400, "needs_human_approval", ["ABOVE_AUTOMATED_LIMIT"]),
    ("elena@example.com", "5590", "INV-40021", 18000, "needs_human_approval", ["ENTERPRISE_MSA"]),
    ("tom@example.com", "2468", "INV-20877", 12, "denied", ["PAST_DUE_BALANCE"]),
    ("wei@example.com", "7070", "INV-20750", 120, "needs_human_approval", ["ABOVE_AUTOMATED_LIMIT", "FREQUENCY_LIMIT"]),
    ("priya@example.com", "4321", "INV-20932", 50, "denied", ["AMOUNT_EXCEEDS_INVOICE"]),
])
async def test_refund_policy_matrix(container, email, pin, invoice, amount, status, codes):
    s = await verified(container, email, pin)
    out, _ = await run(container, s, "issue_refund", invoice_id=invoice, amount=amount, reason="test")
    assert out.ok
    assert out.data["status"] == status
    assert [r["code"] for r in out.data["policy"]["rules"]] == codes


async def test_refund_requires_identity_and_cannot_repeat(container):
    s = container.sessions.get_or_create(None)
    out, _ = await run(container, s, "issue_refund", invoice_id="INV-20932", amount=12, reason="x")
    assert not out.ok and out.error["code"] == "IDENTITY_REQUIRED"
    s = await verified(container, "priya@example.com", "4321")
    first, _ = await run(container, s, "issue_refund", invoice_id="INV-20932", amount=12, reason="x")
    again, _ = await run(container, s, "issue_refund", invoice_id="INV-20932", amount=12, reason="x")
    assert first.data["status"] == "approved" and again.data["status"] == "denied"
    assert "ALREADY_REFUNDED" in [r["code"] for r in again.data["policy"]["rules"]]


async def test_sessions_have_isolated_crm_copies(container):
    a = await verified(container, "priya@example.com", "4321")
    b = await verified(container, "priya@example.com", "4321")
    await run(container, a, "issue_refund", invoice_id="INV-20932", amount=12, reason="x")
    out, _ = await run(container, b, "issue_refund", invoice_id="INV-20932", amount=12, reason="x")
    assert out.data["status"] == "approved"


async def test_identity_lockout_and_no_account_enumeration(container):
    s = container.sessions.get_or_create(None)
    unknown, _ = await run(container, s, "verify_identity", email="nobody@example.com", pin="1111")
    wrong_pin, _ = await run(container, s, "verify_identity", email="priya@example.com", pin="1111")
    assert unknown.data.keys() == wrong_pin.data.keys()  # same shape: doesn't reveal which accounts exist
    locked, ctx = await run(container, s, "verify_identity", email="priya@example.com", pin="1111")
    assert locked.data["locked"] is True and s.identity_locked
    after, _ = await run(container, s, "verify_identity", email="priya@example.com", pin="4321")
    assert after.error["code"] == "IDENTITY_LOCKED"
    ctx.turn.identity_locked_now = True
    tickets = await enforce_turn_guards(ctx)
    assert tickets and tickets[0]["category"] == "account"


async def test_pin_is_masked_in_events_and_traces(container):
    rec = Recorder()
    s = container.sessions.get_or_create(None)
    await run(container, s, "verify_identity", emit=rec, email="priya@example.com", pin="4321")
    started = rec.of("step")[0]
    assert started["args"]["pin"] == "••••"


async def test_flaky_crm_is_retried_and_outage_degrades_gracefully(container):
    rec = Recorder()
    s = await verified(container, "priya@example.com", "4321")
    s.faults.tool_flaky = 1
    out, _ = await run(container, s, "get_account_details", emit=rec)
    assert out.ok and out.attempts == 2
    assert [e["status"] for e in rec.of("step")] == ["started", "retrying", "completed"]
    s.faults.crm_outage = True
    out, _ = await run(container, s, "get_account_details")
    assert not out.ok and out.error["code"] == "UPSTREAM_UNAVAILABLE"
    assert "offer to create an escalation ticket" in out.error["instruction"]


@pytest.mark.parametrize("field,value,ok", [
    ("timezone", "Europe/Berlin", True),
    ("timezone", "Mars/Olympus", False),
    ("preferred_contact_channel", "Phone", True),
    ("preferred_contact_channel", "fax", False),
    ("marketing_emails", "opt out", True),
    ("phone", "call me maybe", False),
])
async def test_account_field_validation(container, field, value, ok):
    s = await verified(container, "priya@example.com", "4321")
    out, _ = await run(container, s, "update_account_field", field=field, value=value)
    assert out.ok is ok


async def test_non_whitelisted_field_is_rejected_by_schema(container):
    s = await verified(container, "priya@example.com", "4321")
    out, _ = await run(container, s, "update_account_field", field="email", value="evil@example.com")
    assert not out.ok and out.error["code"] == "INVALID_ARGUMENTS"


async def test_tickets_are_idempotent_per_category(container):
    s = container.sessions.get_or_create(None)
    a, _ = await run(container, s, "create_escalation_ticket", summary="x", category="billing", priority="P2")
    b, _ = await run(container, s, "create_escalation_ticket", summary="y", category="billing", priority="P2")
    assert a.data["ticket_id"] == b.data["ticket_id"] and b.data["already_open"]
    assert s.status == "escalated"


async def test_kb_tier_comes_from_session_not_from_model(container):
    """Even if the model tried to pass a tier, search uses the verified customer's plan."""
    s = await verified(container, "priya@example.com", "4321")  # pro
    rec = Recorder()
    out, _ = await run(container, s, "search_knowledge_base", emit=rec, query="SLA service credits", tier="enterprise")
    g = rec.of("guardrail")[0]
    assert g["tier_filter"] == "pro"
    assert not any(p["cite"].startswith("enterprise-sla") for p in out.data["passages"])
