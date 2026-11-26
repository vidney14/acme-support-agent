"""The six support tools.

Typical chain for "I was charged twice, please refund it":
  verify_identity -> get_account_details -> search_knowledge_base -> issue_refund
  (-> create_escalation_ticket if the policy engine says a human must approve)
"""

from __future__ import annotations

import random
import re
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.tools.registry import Tool, ToolContext, ToolError, ToolRegistry

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE = re.compile(r"^\+?[0-9][0-9\-\s().]{6,19}$")


# --------------------------------------------------------------------------- identity
async def verify_identity(ctx: ToolContext, args: dict) -> tuple[dict, str]:
    s, rules = ctx.session, ctx.rules
    email, pin = str(args["email"]).strip().lower(), str(args["pin"]).strip()
    if s.identity_locked:
        raise ToolError("IDENTITY_LOCKED", "Verification is locked after too many failed attempts.",
                        instruction="Do not ask for the PIN again. Create an escalation ticket (category account).")
    if not EMAIL.match(email) or not re.fullmatch(r"\d{4}", pin):
        raise ToolError("INVALID_FORMAT", "Email must be a valid address and the PIN exactly 4 digits.",
                        instruction="Ask the customer to re-check the email address and 4-digit PIN.")
    if s.customer and s.customer["email"] == email:
        c = s.customer
        return ({"verified": True, "already_verified": True, "customer_id": c["customer_id"],
                 "name": c["profile"]["display_name"], "plan": c["tier"]}, f"Already verified ({c['tier'].title()})")

    c = s.crm.find_by_email(email)
    if c is None or c["support_pin"] != pin:
        s.verify_failures += 1
        remaining = rules.max_failed_attempts - s.verify_failures
        if remaining <= 0:
            s.identity_locked = True
            ctx.turn.identity_locked_now = True
            return ({"verified": False, "locked": True,
                     "instruction": "Verification is now locked. Tell the customer a human will verify them and "
                                    "create an escalation ticket (category account, priority P2)."},
                    "Didn't match, verification locked")
        # Same message for unknown email and wrong PIN: don't leak which accounts exist.
        return ({"verified": False, "attempts_remaining": remaining,
                 "instruction": "Tell the customer the email or PIN didn't match and ask them to try again."},
                f"Didn't match, {remaining} attempts left")

    s.customer_id = c["customer_id"]
    s.verify_failures = 0
    return ({"verified": True, "customer_id": c["customer_id"], "name": c["profile"]["display_name"],
             "plan": c["tier"], "contract_status": c["contract_status"]},
            f"Verified {c['profile']['display_name']} ({c['tier'].title()})")


# --------------------------------------------------------------------------- account
async def get_account_details(ctx: ToolContext, args: dict) -> tuple[dict, str]:
    c, crm = ctx.session.customer, ctx.session.crm
    window = ctx.rules.frequency_days
    recent_auto = [r for r in c["refunds"] if r.get("approved_by") == "automated" and crm.days_since(r["date"]) <= window]
    data = {
        "customer_id": c["customer_id"],
        "name": c["profile"]["display_name"],
        "plan": c["tier"],
        "billing_cycle": c.get("billing_cycle"),
        "seats": c.get("seats"),
        "contract_status": c["contract_status"],
        "outstanding_balance_usd": c.get("outstanding_balance", 0),
        "storage_used_gb": c.get("storage_used_gb"),
        "invoices": [{"invoice_id": i["invoice_id"], "date": i["date"], "days_since_charge": crm.days_since(i["date"]),
                      "amount_usd": i["amount"], "status": i["status"], "billing_cycle": i.get("cycle"),
                      "description": i.get("description", ""), "duplicate_of": i.get("duplicate_of"),
                      "refunded_usd": i.get("refunded_amount", 0.0)} for i in c["invoices"]],
        "automated_refunds_last_90_days": len(recent_auto),
        "profile": c["profile"],
    }
    if c.get("contract"):
        data["contract"] = c["contract"]
    if c.get("expired_on"):
        data["expired_on"] = c["expired_on"]
    ctx.turn.used_account_data = True
    return data, f"{c['tier'].title()} plan, {c['contract_status'].replace('_', ' ')}, {len(c['invoices'])} invoices"


# --------------------------------------------------------------------------- knowledge
INSTRUCTIONS = {
    "answer": "Answer using ONLY these passages. Cite each fact with its tag, e.g. [refund-policy#4].",
    "answer_with_caution": "The passages only partly cover this. Answer only what they support (with citations), "
                           "say clearly what you could not confirm, and offer to connect a human.",
    "escalate": "The knowledge base does NOT cover this question. Do NOT answer from general knowledge. Tell the "
                "customer you can't confirm this and call create_escalation_ticket so a human follows up.",
}


async def search_knowledge_base(ctx: ToolContext, args: dict) -> tuple[dict, str]:
    if not ctx.kb or not ctx.kb.status.ready:
        raise ToolError("KB_UNAVAILABLE", "The knowledge base is temporarily unavailable.", retryable=True,
                        instruction="Tell the customer you can't look up policy right now and offer a human.")
    customer = ctx.session.customer
    tier = customer["tier"] if customer else None  # tier comes from verified state, never from the model
    query = str(args["query"])[:500]
    result, verdict = await ctx.kb.search(query, tier)
    decision = verdict.decision.value
    passages = [] if decision == "escalate" else [
        {"cite": c.chunk_id, "source": c.breadcrumb, "text": c.text} for c in result.chunks]
    ctx.turn.kb_searches.append({"query": query, "decision": decision, "score": verdict.score,
                                 "citations": [p["cite"] for p in passages]})
    ctx.turn.retrieved_ids.update(p["cite"] for p in passages)
    await ctx.emit("guardrail", {"kind": "retrieval_confidence", "query": query, **verdict.to_dict(),
                                 "tier_filter": tier or "all plans",
                                 "results": [c.to_dict() for c in result.chunks]})
    data = {"decision": decision, "confidence": round(verdict.score, 3), "band": verdict.band.value,
            "filtered_to_plan": tier or "all plans (customer not verified; plan-specific rules may differ)",
            "instruction": INSTRUCTIONS[decision], "passages": passages}
    return data, f"{len(passages)} passages, {verdict.band.value} confidence ({verdict.score:.2f}): {decision.replace('_', ' ')}"


# --------------------------------------------------------------------------- actions
async def issue_refund(ctx: ToolContext, args: dict) -> tuple[dict, str]:
    s, crm = ctx.session, ctx.session.crm
    c = s.customer
    inv = crm.invoice(c["customer_id"], str(args["invoice_id"]))
    if inv is None:
        raise ToolError("INVOICE_NOT_FOUND", f"No invoice {args['invoice_id']} on this account.",
                        instruction="Check the invoice id with get_account_details or ask the customer.")
    amount = round(float(args["amount"]), 2)
    window = ctx.rules.frequency_days
    recent = sum(1 for r in c["refunds"] if r.get("approved_by") == "automated" and crm.days_since(r["date"]) <= window)
    decision = ctx.rules.evaluate_refund(c, inv, amount, crm.days_since(inv["date"]), recent)
    record = {"invoice_id": inv["invoice_id"], "amount": amount, **decision.to_dict()}
    ctx.turn.refund_outcomes.append(record)

    if decision.outcome == "approve":
        refund = crm.record_refund(c["customer_id"], inv["invoice_id"], amount, str(args.get("reason", "")))
        s.actions.append({"type": "refund", "at": time.time(), "refund_id": refund["refund_id"],
                          "summary": f"Refunded ${amount:.2f} on {inv['invoice_id']} ({refund['refund_id']})"})
        return ({"status": "approved", "refund_id": refund["refund_id"], "amount_usd": amount,
                 "invoice_id": inv["invoice_id"], "payout": "original payment method, 5-10 business days",
                 "policy": decision.to_dict()},
                f"Approved ${amount:.2f}, reference {refund['refund_id']}")
    if decision.outcome == "needs_human":
        return ({"status": "needs_human_approval", "policy": decision.to_dict(),
                 "instruction": "Explain that this refund needs approval from the billing team, then call "
                                "create_escalation_ticket (category billing, priority P2) with the invoice, amount "
                                "and the rule codes."},
                "Needs approval: " + ", ".join(h.code for h in decision.hits))
    return ({"status": "denied", "policy": decision.to_dict(),
             "instruction": "Explain the reason(s) kindly and specifically. Offer an alternative (e.g. cancel "
                            "auto-renewal, pay the balance first) or a human review."},
            "Denied: " + ", ".join(h.code for h in decision.hits if h.kind == "deny"))


def _validate_field(spec: dict, raw: object):
    t = spec["type"]
    v = str(raw).strip()
    if t == "string":
        if not v or len(v) > spec.get("max_len", 80):
            raise ValueError(f"must be 1-{spec.get('max_len', 80)} characters")
        return v
    if t == "enum":
        if v.lower() not in spec["values"]:
            raise ValueError(f"must be one of {spec['values']}")
        return v.lower()
    if t == "phone":
        if not PHONE.match(v):
            raise ValueError("must be a valid phone number, e.g. +1-617-555-0100")
        return v
    if t == "timezone":
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("must be an IANA time zone, e.g. America/New_York") from None
        return v
    if t == "bool":
        if v.lower() in ("true", "yes", "on", "opt in", "opt-in", "1"):
            return True
        if v.lower() in ("false", "no", "off", "opt out", "opt-out", "0"):
            return False
        raise ValueError("must be true or false")
    raise ValueError("unsupported field type")


async def update_account_field(ctx: ToolContext, args: dict) -> tuple[dict, str]:
    field, specs = str(args["field"]), ctx.rules.updatable_fields
    if field not in specs:
        raise ToolError("FIELD_NOT_UPDATABLE", f"'{field}' cannot be changed by the assistant.",
                        instruction="Explain how the customer can change it themselves (see account-management "
                                    "policy) or offer a human.")
    try:
        value = _validate_field(specs[field], args["value"])
    except ValueError as e:
        raise ToolError("INVALID_VALUE", f"{field} {e}", instruction="Ask the customer for a valid value.") from None
    old, new = ctx.session.crm.update_profile(ctx.session.customer_id, field, value)
    ctx.session.actions.append({"type": "account_update", "at": time.time(),
                                "summary": f"Updated {field}: {old!r} → {new!r}"})
    return {"status": "updated", "field": field, "old_value": old, "new_value": new,
            "note": "Customer receives a confirmation email."}, f"{field} updated"


async def open_ticket(ctx: ToolContext, *, summary: str, category: str, priority: str,
                      attempted_steps: str = "", source: str = "agent") -> dict:
    s = ctx.session
    for t in s.tickets:  # idempotent: one open ticket per category per session
        if t["category"] == category:
            return {**t, "already_open": True}
    c = s.customer
    tier = c["tier"] if c else "free"
    targets = ctx.rules.response_targets.get(tier, ctx.rules.response_targets["free"])
    ticket = {
        "ticket_id": f"ESC-{random.randint(1000, 9999)}",
        "category": category, "priority": priority, "summary": summary[:400],
        "attempted_steps": attempted_steps[:400],
        "customer_id": c["customer_id"] if c else None, "plan": c["tier"] if c else "unverified",
        "response_target": targets["p1"] if priority == "P1" else targets["standard"],
        "created_at": time.time(), "created_by": source,
        "transcript_excerpt": [f"{m.role}: {m.text[:200]}" for m in s.messages[-6:]],
    }
    s.tickets.append(ticket)
    s.status = "escalated"
    ctx.turn.tickets_created.append(ticket["ticket_id"])
    s.actions.append({"type": "escalation", "at": time.time(),
                      "summary": f"Opened {ticket['ticket_id']} ({category}/{priority})"})
    await ctx.emit("escalation", ticket)
    return ticket


async def create_escalation_ticket(ctx: ToolContext, args: dict) -> tuple[dict, str]:
    t = await open_ticket(ctx, summary=str(args["summary"]), category=str(args.get("category", "other")),
                          priority=str(args.get("priority", "P3")),
                          attempted_steps=str(args.get("attempted_steps", "")))
    data = {"ticket_id": t["ticket_id"], "priority": t["priority"], "category": t["category"],
            "human_response_target": t["response_target"], "already_open": t.get("already_open", False),
            "instruction": "Give the customer the ticket number and the response target."}
    return data, f"{t['ticket_id']} ({t['priority']}), human reply within {t['response_target']}"


# --------------------------------------------------------------------------- registry
def build_registry() -> ToolRegistry:
    obj = lambda props, req=(): {"type": "object", "properties": props, "required": list(req)}  # noqa: E731
    return ToolRegistry([
        Tool("verify_identity", "Verifying identity",
             "Verify the customer's identity with the email on their AcmeCloud account and their 4-digit support "
             "PIN. Required before discussing account details or billing, or making any change. If you don't have "
             "both values, ask the customer; never guess them.",
             obj({"email": {"type": "string", "description": "Account email address"},
                  "pin": {"type": "string", "description": "4-digit support PIN"}}, ["email", "pin"]),
             verify_identity, uses_crm=True, sensitive_args=("pin",)),
        Tool("get_account_details", "Checking account tier & contract",
             "Get the verified customer's plan tier, billing cycle, contract status, outstanding balance, recent "
             "invoices (with ids, dates, amounts, duplicates) and recent refunds. Call after verify_identity and "
             "before any billing action or account-specific answer.",
             obj({}), get_account_details, requires_identity=True, uses_crm=True),
        Tool("search_knowledge_base", "Searching knowledge base",
             "Search AcmeCloud's support policies and troubleshooting guides. Results are automatically filtered to "
             "the verified customer's plan. Call this before answering ANY question about policies, pricing, plans, "
             "refund rules, contracts, troubleshooting, security or privacy. The result contains a confidence "
             "`decision` and an `instruction` that you must follow.",
             obj({"query": {"type": "string",
                            "description": "Focused search query, rewritten from the customer's question"}}, ["query"]),
             search_knowledge_base),
        Tool("issue_refund", "Processing refund",
             "Request a refund for one invoice of the verified customer. A deterministic policy engine checks the "
             "refund window, automated approval limit, account status and frequency rules, and returns approved, "
             "needs_human_approval or denied. Only call when the customer asked for a refund; if several invoices "
             "could match, ask which one.",
             obj({"invoice_id": {"type": "string"},
                  "amount": {"type": "number", "description": "Amount in USD (full invoice amount unless partial)"},
                  "reason": {"type": "string", "description": "Short reason, e.g. 'duplicate charge'"}},
                 ["invoice_id", "amount", "reason"]),
             issue_refund, requires_identity=True, uses_crm=True),
        Tool("update_account_field", "Updating account",
             "Update one profile field for the verified customer. Allowed fields: display_name, phone, "
             "preferred_contact_channel (email|phone|chat), timezone (IANA, e.g. Europe/Berlin), marketing_emails "
             "(true|false). Login email, payment method and plan cannot be changed here.",
             obj({"field": {"type": "string", "enum": ["display_name", "phone", "preferred_contact_channel",
                                                        "timezone", "marketing_emails"]},
                  "value": {"type": "string"}}, ["field", "value"]),
             update_account_field, requires_identity=True, uses_crm=True),
        Tool("create_escalation_ticket", "Creating escalation ticket",
             "Hand the case to a human support agent. Use when a refund needs human approval, the knowledge base "
             "decision is 'escalate', identity can't be verified, for security/privacy requests, for unresolved "
             "technical problems, or when the customer asks for a human. Include what you already tried.",
             obj({"summary": {"type": "string", "description": "What the customer needs, in 1-3 sentences"},
                  "category": {"type": "string", "enum": ["billing", "technical", "security", "privacy", "account",
                                                           "other"]},
                  "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
                  "attempted_steps": {"type": "string"}}, ["summary", "category", "priority"]),
             create_escalation_ticket),
    ])
