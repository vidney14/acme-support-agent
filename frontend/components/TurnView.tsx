"use client";

import { AnswerText } from "./AnswerText";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { AlertIcon, CheckIcon, CloudIcon, HandoffIcon, MicIcon, RefreshIcon } from "./icons";
import { ToolTrack } from "./ToolTrack";
import type { Grounding, Ticket, Turn } from "@/lib/types";

interface Props {
  turn: Turn;
  low: number;
  high: number;
  onRetry: (id: string) => void;
  onOpenTrace: (traceId: string) => void;
  onAskHuman: () => void;
}

export function TurnView({ turn, low, high, onRetry, onOpenTrace, onAskHuman }: Props) {
  const streaming = turn.status === "streaming";
  const voice = turn.modality === "voice";
  const showAgent =
    turn.agent || turn.steps.length || turn.status === "error" || streaming || turn.tickets.length;

  return (
    <article className="space-y-4">
      {turn.user && (
        <div className="rise flex justify-end">
          <div className="max-w-[80%] rounded-2xl rounded-br-md bg-gradient-to-br from-ink to-[#24344a] px-4 py-2.5 text-body text-paper shadow-[0_10px_24px_-14px_var(--color-ink)]">
            {voice && (
              <span className="mb-0.5 flex items-center gap-1 text-caption text-paper/60">
                <MicIcon width={12} height={12} /> Spoken
              </span>
            )}
            <p className="whitespace-pre-wrap">{turn.user}</p>
          </div>
        </div>
      )}

      {showAgent ? (
        <div className="rise flex gap-3 pr-2 sm:pr-10" style={{ animationDelay: "80ms" }}>
          <AgentAvatar active={streaming} />
          <div className="min-w-0 flex-1 pt-1">
          <ToolTrack turn={turn} />
          {turn.confidences.map((c, i) => (
            <ConfidenceGauge key={i} c={c} low={low} high={high} />
          ))}
          {turn.notices.map((n, i) => (
            <p key={i} className="mb-2 flex items-start gap-2 text-small text-work">
              <RefreshIcon width={15} height={15} className="mt-[3px] shrink-0" />
              <span>{n.message}</span>
            </p>
          ))}

          {turn.agent ? (
            <div className="relative">
              {voice && (
                <span className="mb-1 flex items-center gap-1 text-caption text-mist">
                  <MicIcon width={12} height={12} /> Spoken reply
                </span>
              )}
              <AnswerText text={turn.agent} sources={turn.sources} streaming={streaming && !voice} />
              {turn.interrupted && (
                <p className="mt-1 text-caption text-work">You interrupted here, so the rest wasn&apos;t spoken.</p>
              )}
            </div>
          ) : (
            streaming &&
            !turn.steps.length &&
            !turn.phase && <p className="shimmer-text text-small">Connecting to the agent</p>
          )}

          {turn.tickets.map((t) => (
            <EscalationCard key={t.ticket_id} t={t} />
          ))}

          {turn.status === "error" && turn.error && (
            <div role="alert" className="mt-3 flex flex-wrap items-center gap-3 rounded-lg bg-fault-soft px-3 py-2">
              <AlertIcon width={16} height={16} className="text-fault" />
              <p className="text-small text-fault">{turn.error.message}</p>
              {turn.error.recoverable && (
                <button
                  onClick={() => onRetry(turn.id)}
                  className="rounded-md border border-fault/40 px-2.5 py-1 text-small font-semibold text-fault hover:bg-surface"
                >
                  Retry
                </button>
              )}
            </div>
          )}

          {!streaming && turn.status !== "error" && (
            <Footer turn={turn} onOpenTrace={onOpenTrace} onAskHuman={onAskHuman} />
          )}
          </div>
        </div>
      ) : null}
    </article>
  );
}

function AgentAvatar({ active }: { active: boolean }) {
  return (
    <span className="relative mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center" aria-hidden>
      <span className={`avatar-ring absolute inset-0 rounded-full ${active ? "is-active" : "opacity-60"}`} />
      <span className="relative flex h-[26px] w-[26px] items-center justify-center rounded-full bg-surface text-brand">
        <CloudIcon width={15} height={15} />
      </span>
    </span>
  );
}

function EscalationCard({ t }: { t: Ticket }) {
  return (
    <div className="rise relative mt-3 max-w-xl overflow-hidden rounded-2xl border border-handoff/30 bg-gradient-to-br from-handoff-soft to-surface py-3 pl-5 pr-4 shadow-[0_12px_28px_-18px_var(--color-handoff)]">
      <span aria-hidden className="absolute inset-y-0 left-0 w-1.5 bg-handoff" />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-2 font-display text-title text-handoff">
          <HandoffIcon /> Handed to a human
        </p>
        <span className="tnum rounded-md border border-dashed border-handoff/50 bg-surface px-2 py-0.5 font-mono text-caption font-semibold text-handoff">
          {t.ticket_id}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5 text-caption">
        <span className="rounded-full bg-surface px-2 py-0.5 text-ink">Priority {t.priority}</span>
        <span className="rounded-full bg-surface px-2 py-0.5 text-ink">{t.category}</span>
        <span className="rounded-full bg-surface px-2 py-0.5 text-ink">Reply within {t.response_target}</span>
      </div>
      <p className="mt-2 text-caption text-slate">
        {t.created_by === "guardrail"
          ? "Opened automatically by a guardrail, so the case can't fall through the cracks."
          : t.summary}
      </p>
    </div>
  );
}

const GROUNDING: Record<Grounding["status"], { text: (g: Grounding) => string; tone: string } | null> = {
  grounded: { text: (g) => `Grounded in ${g.citations.length} knowledge base ${g.citations.length === 1 ? "source" : "sources"}`, tone: "text-ok" },
  account_data: { text: () => "Based on your account data", tone: "text-slate" },
  escalated: { text: () => "Didn't guess: the knowledge base doesn't cover this, so a human will answer", tone: "text-handoff" },
  no_claims: null,
  unsupported: { text: () => "Not verified against the knowledge base", tone: "text-work" },
  invalid_citation: { text: (g) => `Cited ${g.invalid_citations.length} source(s) that weren't retrieved`, tone: "text-fault" },
};

function Footer({ turn, onOpenTrace, onAskHuman }: { turn: Turn; onOpenTrace: (id: string) => void; onAskHuman: () => void }) {
  const g = turn.grounding ? GROUNDING[turn.grounding.status] : null;
  const warn = turn.grounding && (turn.grounding.status === "unsupported" || turn.grounding.status === "invalid_citation");
  const elapsed = turn.finishedAt && !turn.restored ? (turn.finishedAt - turn.startedAt) / 1000 : null;
  if (!g && !turn.traceId && elapsed === null) return null;
  return (
    <div className="tnum mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-caption text-mist">
      {g && turn.grounding && (
        <span className={`flex items-center gap-1 ${g.tone}`}>
          {warn ? <AlertIcon width={13} height={13} /> : <CheckIcon width={13} height={13} />}
          {g.text(turn.grounding)}
        </span>
      )}
      {warn && (
        <button onClick={onAskHuman} className="text-handoff underline-offset-2 hover:underline">
          Ask a human instead
        </button>
      )}
      {turn.modality === "voice" && turn.latencyMs != null && (
        <span>
          {turn.user
            ? `Replied in ${(turn.latencyMs / 1000).toFixed(2)}s after you stopped talking`
            : `Started speaking in ${(turn.latencyMs / 1000).toFixed(2)}s`}
        </span>
      )}
      {turn.modality === "text" && elapsed !== null && <span>{elapsed.toFixed(1)}s</span>}
      {turn.traceId && (
        <button onClick={() => onOpenTrace(turn.traceId!)} className="underline-offset-2 hover:text-slate hover:underline">
          View trace
        </button>
      )}
    </div>
  );
}
