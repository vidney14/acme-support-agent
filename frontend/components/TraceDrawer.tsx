"use client";

import { useMemo, useState } from "react";
import { CloseIcon, MicIcon } from "./icons";
import type { Span, Trace } from "@/lib/types";

const KIND_COLOUR: Record<string, string> = {
  agent: "bg-rule",
  llm: "bg-ink",
  tool: "bg-ok",
  retrieval: "bg-slate",
  embedding: "bg-mist",
  guardrail: "bg-handoff",
  voice: "bg-work",
};

const BLOCKS = new Set(["prompt", "output", "system_prompt", "result", "args", "agent_transcript", "error", "grounding", "reasons", "tokens_by_modality", "thresholds"]);

function fmtMs(ms: number | null | undefined) {
  if (ms == null) return "n/a";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

function fmtCost(c: number) {
  if (c === 0) return "$0";
  return c < 0.01 ? `$${c.toFixed(5)}` : `$${c.toFixed(3)}`;
}

function median(xs: number[]) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

export function TraceDrawer({
  traces,
  selected,
  onSelect,
  onClose,
}: {
  traces: Trace[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  onClose: () => void;
}) {
  const ordered = useMemo(() => [...traces].reverse(), [traces]);
  const slowest = useMemo(() => Math.max(1, ...traces.map((t) => t.total_ms ?? 0)), [traces]);
  const totals = useMemo(() => {
    const tin = traces.reduce((a, t) => a + t.totals.input_tokens, 0);
    const tout = traces.reduce((a, t) => a + t.totals.output_tokens, 0);
    const cost = traces.reduce((a, t) => a + t.totals.cost_usd, 0);
    const ttft = median(traces.map((t) => t.ttft_ms).filter((x): x is number => x != null));
    const total = median(traces.map((t) => t.total_ms).filter((x): x is number => x != null));
    return { tin, tout, cost, ttft, total };
  }, [traces]);

  return (
    <aside className="flex h-full flex-col bg-surface" aria-label="Observability">
      <div className="flex items-start justify-between border-b border-rule px-5 py-4">
        <div>
          <h2 className="font-display text-title">Traces</h2>
          <p className="text-caption text-slate">Every turn: model calls, tools, guardrails, latency and cost.</p>
        </div>
        <button onClick={onClose} aria-label="Close traces" className="rounded-lg p-1.5 text-slate hover:bg-paper">
          <CloseIcon />
        </button>
      </div>

      <dl className="tnum grid grid-cols-2 gap-2 border-b border-rule px-5 py-3 text-caption">
        <Stat label="Turns" value={String(traces.length)} />
        <Stat label="Tokens in / out" value={`${totals.tin.toLocaleString()} / ${totals.tout.toLocaleString()}`} />
        <Stat label="Median first token" value={fmtMs(totals.ttft)} />
        <Stat label="Est. cost" value={fmtCost(totals.cost)} />
      </dl>

      <div className="scroll-thin flex-1 overflow-y-auto">
        {!ordered.length && (
          <p className="px-5 py-6 text-small text-slate">Send a message and its trace appears here as it finishes.</p>
        )}
        <ul>
          {ordered.map((t) => (
            <li key={t.trace_id} className="border-b border-rule-soft">
              <button
                onClick={() => onSelect(selected === t.trace_id ? null : t.trace_id)}
                aria-expanded={selected === t.trace_id}
                className={`w-full px-5 py-3 text-left hover:bg-paper ${selected === t.trace_id ? "bg-paper" : ""}`}
              >
                <div className="flex items-center gap-2">
                  {t.modality === "voice" && <MicIcon width={13} height={13} className="shrink-0 text-work" />}
                  <span className="truncate text-small text-ink">{t.user_input || "(voice greeting)"}</span>
                  {t.status !== "ok" && <span className="shrink-0 text-caption text-fault">failed</span>}
                </div>
                <span aria-hidden className="mt-1.5 block h-1 overflow-hidden rounded-full bg-rule-soft">
                  <span
                    className={`block h-full rounded-full ${t.status !== "ok" ? "bg-fault" : "bg-brand-gradient"}`}
                    style={{ width: `${Math.max(3, ((t.total_ms ?? 0) / slowest) * 100)}%` }}
                  />
                </span>
                <div className="tnum mt-1 flex flex-wrap gap-x-3 text-caption text-mist">
                  <span>{fmtMs(t.total_ms)} total</span>
                  <span>first {t.modality === "voice" ? "audio" : "token"} {fmtMs(t.ttft_ms)}</span>
                  <span>{t.totals.llm_calls} model, {t.totals.tool_calls} tool</span>
                  <span>{(t.totals.input_tokens + t.totals.output_tokens).toLocaleString()} tok</span>
                  <span>{fmtCost(t.totals.cost_usd)}</span>
                </div>
              </button>
              {selected === t.trace_id && <TraceDetail trace={t} />}
            </li>
          ))}
        </ul>
      </div>
      <p className="border-t border-rule px-5 py-2 text-caption text-mist">
        Costs are estimated at Gemini paid-tier prices. The free tier bills $0.
      </p>
    </aside>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-rule-soft bg-paper/60 px-2.5 py-2">
      <dt className="leading-tight text-mist">{label}</dt>
      <dd className="mt-1 font-display text-title font-semibold leading-none text-ink">{value}</dd>
    </div>
  );
}

function TraceDetail({ trace }: { trace: Trace }) {
  const [spanId, setSpanId] = useState<string | null>(null);
  const spans = useMemo(() => [...trace.spans].sort((a, b) => a.start_ms - b.start_ms), [trace]);
  const total = Math.max(trace.total_ms ?? 0, ...spans.map((s) => s.start_ms + (s.duration_ms ?? 0)), 1);
  const depth = (s: Span) => {
    let d = 0;
    let p = s.parent_id;
    while (p && d < 4) {
      p = spans.find((x) => x.span_id === p)?.parent_id ?? null;
      d++;
    }
    return d;
  };
  const sel = spans.find((s) => s.span_id === spanId);

  return (
    <div className="bg-paper px-5 pb-4">
      <ul className="space-y-1 pt-1" aria-label="Span waterfall">
        {spans.map((s) => (
          <li key={s.span_id}>
            <button
              onClick={() => setSpanId(spanId === s.span_id ? null : s.span_id)}
              className={`grid w-full grid-cols-[9.5rem_1fr_3.5rem] items-center gap-2 rounded px-1 py-0.5 text-left hover:bg-surface ${spanId === s.span_id ? "bg-surface" : ""}`}
            >
              <span
                className={`truncate text-caption ${s.status === "error" ? "text-fault" : "text-ink"}`}
                style={{ paddingLeft: depth(s) * 8 }}
              >
                {s.name}
              </span>
              <span className="relative h-2.5">
                <span
                  className={`absolute top-0 h-2.5 min-w-[2px] rounded-sm ${s.status === "error" ? "bg-fault" : KIND_COLOUR[s.kind] ?? "bg-slate"}`}
                  style={{ left: `${(s.start_ms / total) * 100}%`, width: `${((s.duration_ms ?? 0) / total) * 100}%` }}
                />
              </span>
              <span className="tnum text-right text-caption text-slate">{fmtMs(s.duration_ms)}</span>
            </button>
          </li>
        ))}
      </ul>
      {sel && <SpanDetail span={sel} />}
    </div>
  );
}

function SpanDetail({ span }: { span: Span }) {
  const entries = Object.entries(span.attrs);
  const scalars = entries.filter(([k, v]) => !BLOCKS.has(k) && (typeof v !== "object" || v === null));
  const blocks = entries.filter(([k, v]) => BLOCKS.has(k) || (typeof v === "object" && v !== null));
  return (
    <div className="mt-3 rounded-lg border border-rule bg-surface p-3">
      <p className="text-small font-semibold text-ink">{span.name}</p>
      <dl className="tnum mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-caption">
        <dt className="text-mist">kind</dt>
        <dd className="text-ink">{span.kind}</dd>
        {scalars.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-mist">{k.replaceAll("_", " ")}</dt>
            <dd className="break-all text-ink">{String(v)}</dd>
          </div>
        ))}
      </dl>
      {blocks.map(([k, v]) => (
        <details key={k} className="mt-2" open={k === "prompt" || k === "output" || k === "result"}>
          <summary className="cursor-pointer text-caption text-slate">{k.replaceAll("_", " ")}</summary>
          <pre className="scroll-thin mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded bg-paper p-2 font-mono text-[0.7rem] leading-relaxed text-ink">
            {typeof v === "string" ? v : JSON.stringify(v, null, 2)}
          </pre>
        </details>
      ))}
    </div>
  );
}
