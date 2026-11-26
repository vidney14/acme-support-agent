// Pure state transitions: events from the SSE chat stream or the voice WebSocket are
// folded into a list of Turns. Keeping this pure (no React, no I/O) makes it easy to
// reason about and test.

import type {
  Confidence,
  Grounding,
  Notice,
  SessionSnapshot,
  Step,
  StoredMessage,
  Ticket,
  Trace,
  Turn,
} from "./types";

export interface AgentState {
  turns: Turn[];
  traces: Trace[];
  snapshot: SessionSnapshot | null;
  openVoiceTurnId: string | null; // voice turn currently being spoken/heard
}

export const initialState: AgentState = { turns: [], traces: [], snapshot: null, openVoiceTurnId: null };

let counter = 0;
export const newId = () => `t${Date.now().toString(36)}${(counter++).toString(36)}`;

export function blankTurn(id: string, user: string, modality: Turn["modality"]): Turn {
  return {
    id,
    modality,
    user,
    agent: "",
    steps: [],
    confidences: [],
    sources: {},
    tickets: [],
    notices: [],
    status: "streaming",
    startedAt: Date.now(),
  };
}

export type Action =
  | { type: "reset" }
  | { type: "restore"; messages: StoredMessage[]; traces: Trace[]; snapshot: SessionSnapshot }
  | { type: "turn_start"; id: string; user: string; modality: Turn["modality"] }
  | { type: "turn_retry"; id: string }
  | { type: "event"; id: string; event: string; data: Record<string, unknown> }
  | { type: "voice_event"; event: string; data: Record<string, unknown> }
  | { type: "turn_error"; id: string; error: Turn["error"] }
  | { type: "recovered"; id: string; text: string; grounding?: Grounding; sources?: Record<string, string> }
  | { type: "snapshot"; snapshot: SessionSnapshot }
  | { type: "traces"; traces: Trace[] };

function patch(state: AgentState, id: string, fn: (t: Turn) => Turn): AgentState {
  return { ...state, turns: state.turns.map((t) => (t.id === id ? fn(t) : t)) };
}

function upsertTrace(traces: Trace[], tr: Trace): Trace[] {
  const rest = traces.filter((t) => t.trace_id !== tr.trace_id);
  return [...rest, tr].slice(-60);
}

/** Apply one agent event to one turn (shared by chat and voice). */
function applyToTurn(t: Turn, event: string, d: Record<string, unknown>): Turn {
  switch (event) {
    case "turn_start":
      return { ...t, serverTurnId: d.turn_id as string, traceId: d.trace_id as string };
    case "status":
      return { ...t, phase: { phase: d.phase as string, label: d.label as string } };
    case "step": {
      const s = d as unknown as Step;
      const exists = t.steps.some((x) => x.step_id === s.step_id);
      const steps = exists
        ? t.steps.map((x) => (x.step_id === s.step_id ? { ...x, ...s, args: x.args ?? s.args } : x))
        : [...t.steps, s];
      return { ...t, steps };
    }
    case "token":
      return { ...t, agent: t.agent + (d.text as string) };
    case "draft_reset": {
      const n = Number(d.chars) || 0;
      return { ...t, agent: t.agent.slice(0, Math.max(0, t.agent.length - n)) };
    }
    case "notice":
      return { ...t, notices: [...t.notices, d as unknown as Notice] };
    case "guardrail":
      if (d.kind === "retrieval_confidence") return { ...t, confidences: [...t.confidences, d as unknown as Confidence] };
      if (d.kind === "grounding") return { ...t, grounding: d as unknown as Grounding };
      return t;
    case "escalation": {
      const tk = d as unknown as Ticket;
      if (t.tickets.some((x) => x.ticket_id === tk.ticket_id)) return t;
      return { ...t, tickets: [...t.tickets, tk] };
    }
    case "message":
      return {
        ...t,
        agent: (d.text as string) ?? t.agent,
        grounding: (d.grounding as Grounding) ?? t.grounding,
        sources: (d.sources as Record<string, string>) ?? t.sources,
      };
    case "trace":
      return { ...t, traceId: (d as unknown as Trace).trace_id };
    case "error":
      return {
        ...t,
        status: "error",
        phase: undefined,
        error: { message: d.message as string, recoverable: Boolean(d.recoverable), kind: d.kind as string },
        finishedAt: Date.now(),
      };
    case "done":
      return { ...t, status: t.status === "error" ? "error" : "done", phase: undefined, finishedAt: Date.now() };
    default:
      return t;
  }
}

export function reducer(state: AgentState, a: Action): AgentState {
  switch (a.type) {
    case "reset":
      return initialState;

    case "restore": {
      // Rebuild turns from the stored transcript (grouped by server turn id).
      const byTurn = new Map<string, Turn>();
      for (const m of a.messages) {
        let t = byTurn.get(m.turn_id);
        if (!t) {
          t = { ...blankTurn(newId(), "", m.modality), status: "done", restored: true, serverTurnId: m.turn_id };
          byTurn.set(m.turn_id, t);
        }
        if (m.role === "user") t.user = m.text;
        else {
          t.agent = m.text;
          t.grounding = m.meta.grounding;
          t.sources = m.meta.sources ?? {};
          t.interrupted = m.meta.interrupted;
        }
      }
      return { ...state, turns: [...byTurn.values()], traces: a.traces, snapshot: a.snapshot, openVoiceTurnId: null };
    }

    case "turn_start":
      return { ...state, turns: [...state.turns, blankTurn(a.id, a.user, a.modality)] };

    case "turn_retry":
      return patch(state, a.id, (t) => ({
        ...blankTurn(t.id, t.user, t.modality),
        notices: [{ kind: "retry", message: "Retrying this message" }],
      }));

    case "event": {
      let next = patch(state, a.id, (t) => applyToTurn(t, a.event, a.data));
      if (a.event === "session") next = { ...next, snapshot: a.data as unknown as SessionSnapshot };
      if (a.event === "trace") next = { ...next, traces: upsertTrace(next.traces, a.data as unknown as Trace) };
      return next;
    }

    case "voice_event":
      return voiceReducer(state, a.event, a.data);

    case "turn_error":
      return patch(state, a.id, (t) => ({ ...t, status: "error", phase: undefined, error: a.error, finishedAt: Date.now() }));

    case "recovered":
      return patch(state, a.id, (t) => ({
        ...t,
        agent: a.text,
        grounding: a.grounding,
        sources: a.sources ?? {},
        status: "done",
        error: undefined,
        phase: undefined,
        notices: [...t.notices, { kind: "recovered", message: "Connection dropped, reply recovered from the server" }],
      }));

    case "snapshot":
      return { ...state, snapshot: a.snapshot };

    case "traces":
      return { ...state, traces: a.traces };
  }
}

/** Voice events don't carry a client turn id: route them to the "open" voice turn. */
function voiceReducer(state: AgentState, event: string, d: Record<string, unknown>): AgentState {
  if (event === "session") return { ...state, snapshot: d as unknown as SessionSnapshot };
  if (event === "trace") {
    const tr = d as unknown as Trace;
    // attach to the most recent finished voice turn without a trace
    const idx = [...state.turns].reverse().findIndex((t) => t.modality === "voice" && t.status !== "streaming" && !t.traceId);
    let turns = state.turns;
    if (idx !== -1) {
      const real = state.turns.length - 1 - idx;
      turns = state.turns.map((t, i) => (i === real ? { ...t, traceId: tr.trace_id } : t));
    }
    return { ...state, turns, traces: upsertTrace(state.traces, tr) };
  }

  const routed = ["transcript", "step", "guardrail", "escalation", "notice", "status", "interrupted", "turn_complete"];
  if (!routed.includes(event)) return state;

  let s = state;
  let id = s.openVoiceTurnId;
  if (!id) {
    if (event === "turn_complete") return s; // nothing open
    id = newId();
    s = { ...s, turns: [...s.turns, blankTurn(id, "", "voice")], openVoiceTurnId: id };
  }
  const openId = id;

  switch (event) {
    case "transcript": {
      const role = d.role as string;
      const text = d.text as string;
      return patch(s, openId, (t) =>
        role === "user"
          ? { ...t, user: d.typed ? text : t.user + text }
          : { ...t, agent: t.agent + text },
      );
    }
    case "interrupted":
      return patch(s, openId, (t) => ({ ...t, interrupted: true }));
    case "turn_complete":
      return {
        ...patch(s, openId, (t) => ({
          ...t,
          status: "done",
          phase: undefined,
          interrupted: Boolean(d.interrupted) || t.interrupted,
          latencyMs: (d.latency_ms as number | null) ?? null,
          grounding: (d.grounding as Grounding) ?? t.grounding,
          finishedAt: Date.now(),
        })),
        openVoiceTurnId: null,
      };
    default:
      return patch(s, openId, (t) => applyToTurn(t, event, d));
  }
}
