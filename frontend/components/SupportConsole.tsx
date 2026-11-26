"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { CallBar } from "./CallBar";
import { Composer } from "./Composer";
import { DemoGuide, SUGGESTIONS } from "./DemoGuide";
import { BoltIcon, BookIcon, BugIcon, CloudIcon, HandoffIcon, MicIcon, PanelIcon, PulseIcon, ReceiptIcon, ShieldIcon } from "./icons";
import { TraceDrawer } from "./TraceDrawer";
import { TurnView } from "./TurnView";
import { useSupportAgent } from "@/lib/useSupportAgent";
import type { SessionSnapshot } from "@/lib/types";

const WIDE = "(min-width: 1180px)";
function subscribeWide(cb: () => void) {
  const mq = window.matchMedia(WIDE);
  mq.addEventListener("change", cb);
  return () => mq.removeEventListener("change", cb);
}

export function SupportConsole() {
  const agent = useSupportAgent();
  const { state, config, voice } = agent;
  const [draft, setDraft] = useState("");
  const [guideOpen, setGuideOpen] = useState(false);
  // Drawer defaults to open on wide screens; once the user toggles it, their choice wins.
  const wide = useSyncExternalStore(subscribeWide, () => window.matchMedia(WIDE).matches, () => false);
  const [drawerPref, setDrawerPref] = useState<boolean | null>(null);
  const drawerOpen = drawerPref ?? wide;
  const setDrawerOpen = (v: boolean | ((o: boolean) => boolean)) =>
    setDrawerPref(typeof v === "function" ? v(drawerOpen) : v);
  const [selectedTrace, setSelectedTrace] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  const low = config?.guardrails.confidence_low ?? 0.55;
  const high = config?.guardrails.confidence_high ?? 0.72;
  const inCall = voice.status !== "idle";

  // Stick to the bottom while content streams, unless the reader scrolled up.
  useEffect(() => {
    const el = scroller.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [state.turns]);

  const onScroll = () => {
    const el = scroller.current;
    if (el) pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  const openTrace = useCallback((id: string) => {
    setSelectedTrace(id);
    setDrawerPref(true);
  }, []);

  const send = (text: string) => {
    pinned.current = true;
    void agent.send(text);
  };

  if (agent.connError) {
    return (
      <main className="mx-auto flex h-full max-w-xl flex-col justify-center px-6">
        <h1 className="font-display text-display">AcmeCloud Support is offline</h1>
        <p className="mt-2 text-body text-slate">{agent.connError}</p>
        <button
          onClick={() => window.location.reload()}
          className="mt-4 self-start rounded-xl bg-ink px-4 py-2 text-small font-semibold text-paper"
        >
          Try again
        </button>
      </main>
    );
  }

  return (
    <div className="flex h-full">
      <div className="flex h-full min-w-0 flex-1 flex-col">
        <header className="relative z-10 flex items-center gap-3 border-b border-rule/70 bg-surface/70 px-4 py-3 backdrop-blur-md sm:px-6">
          <span className="bg-brand-gradient flex h-8 w-8 shrink-0 items-center justify-center rounded-xl text-surface shadow-[0_6px_16px_-6px_var(--color-brand)]">
            <CloudIcon width={18} height={18} />
          </span>
          <div className="leading-tight">
            <h1 className="font-display text-title font-semibold tracking-tight">
              AcmeCloud <span className="font-normal text-slate">Support</span>
            </h1>
            <p className="flex items-center gap-1.5 text-[0.7rem] text-mist">
              <span className="live-dot h-1.5 w-1.5 rounded-full bg-ok" />
              {agent.busy ? "Acme Assist is working" : "Acme Assist is online"}
            </p>
          </div>
          <SessionPill snap={state.snapshot} />
          <div className="ml-auto flex items-center gap-1">
            <HeaderButton onClick={() => setGuideOpen(true)} icon={<BookIcon />} label="Demo guide" />
            <HeaderButton
              onClick={() => setDrawerOpen((o) => !o)}
              icon={<PanelIcon />}
              label="Traces"
              pressed={drawerOpen}
            />
            <button
              onClick={() => void agent.newSession()}
              className="ml-1 hidden rounded-lg px-2.5 py-1.5 text-small text-slate hover:bg-surface hover:text-ink sm:block"
            >
              New conversation
            </button>
          </div>
        </header>

        <div ref={scroller} onScroll={onScroll} className="app-canvas scroll-thin flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
            {state.turns.length === 0 && config ? (
              <Welcome voice={config.voice_enabled} onPick={send} />
            ) : (
              <div className="space-y-10">
                {state.turns.map((t) => (
                  <TurnView
                    key={t.id}
                    turn={t}
                    low={low}
                    high={high}
                    onRetry={(id) => void agent.retry(id)}
                    onOpenTrace={openTrace}
                    onAskHuman={() => send("I'd like to talk to a human about this.")}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6">
          {inCall ? (
            <CallBar
              voice={voice}
              simulated={config?.voice_simulated}
              session={agent.voiceSession}
              onEnd={agent.stopVoice}
              onMute={agent.toggleMute}
              onRetry={() => {
                agent.stopVoice();
                setTimeout(() => void agent.startVoice(), 50);
              }}
              onSendText={(t) => void agent.send(t)}
            />
          ) : (
            <Composer
              disabled={!agent.sessionId}
              busy={agent.busy}
              cooldownUntil={agent.cooldownUntil}
              voiceAvailable={Boolean(config?.voice_enabled)}
              draft={draft}
              onDraftChange={setDraft}
              onSend={send}
              onStartVoice={() => void agent.startVoice()}
            />
          )}
        </div>
      </div>

      {drawerOpen && (
        <div className="fixed inset-0 z-30 border-l border-rule bg-surface lg:static lg:z-auto lg:w-[27rem] lg:shrink-0">
          <TraceDrawer
            traces={state.traces}
            selected={selectedTrace}
            onSelect={setSelectedTrace}
            onClose={() => setDrawerOpen(false)}
          />
        </div>
      )}

      {guideOpen && config && (
        <DemoGuide
          config={config}
          faults={state.snapshot?.faults}
          onClose={() => setGuideOpen(false)}
          onUseAccount={(t) => {
            setDraft(t);
            setGuideOpen(false);
          }}
          onPrompt={(t) => {
            setGuideOpen(false);
            send(t);
          }}
          onFaults={(f) => void agent.setFaults(f)}
        />
      )}
    </div>
  );
}

function HeaderButton({ onClick, icon, label, pressed }: { onClick: () => void; icon: React.ReactNode; label: string; pressed?: boolean }) {
  return (
    <button
      onClick={onClick}
      aria-pressed={pressed}
      className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-small hover:bg-surface ${pressed ? "text-ink" : "text-slate"}`}
    >
      {icon}
      <span className="hidden md:inline">{label}</span>
    </button>
  );
}

function SessionPill({ snap }: { snap: SessionSnapshot | null }) {
  if (!snap) return null;
  const ticket = snap.tickets[snap.tickets.length - 1];
  if (snap.status === "escalated" && ticket)
    return (
      <span className="hidden rounded-full bg-handoff-soft px-2.5 py-0.5 text-caption text-handoff sm:inline">
        With a human: {ticket.ticket_id}
      </span>
    );
  if (snap.customer)
    return (
      <span className="hidden rounded-full bg-ok-soft px-2.5 py-0.5 text-caption text-ok sm:inline">
        Verified: {snap.customer.name}, {snap.customer.tier[0].toUpperCase() + snap.customer.tier.slice(1)}
      </span>
    );
  if (snap.identity_locked)
    return <span className="hidden rounded-full bg-fault-soft px-2.5 py-0.5 text-caption text-fault sm:inline">Verification locked</span>;
  return <span className="hidden rounded-full border border-rule px-2.5 py-0.5 text-caption text-slate sm:inline">Not verified</span>;
}

const CARDS = [
  { icon: ReceiptIcon, tag: "Billing", tone: "text-brand bg-brand-soft" },
  { icon: BugIcon, tag: "Troubleshoot", tone: "text-brand-2 bg-brand-2/10" },
  { icon: CloudIcon, tag: "Policy", tone: "text-brand bg-brand-soft" },
  { icon: HandoffIcon, tag: "Guardrail", tone: "text-handoff bg-handoff-soft" },
];

const CAPABILITIES = [
  { icon: BoltIcon, label: "Live tool calls" },
  { icon: ShieldIcon, label: "Grounded answers, no guessing" },
  { icon: MicIcon, label: "Real-time voice" },
  { icon: PulseIcon, label: "Full traces and cost" },
];

function Welcome({ onPick, voice }: { onPick: (t: string) => void; voice: boolean }) {
  return (
    <div className="pt-2 sm:pt-4">
      <span className="rise inline-flex items-center gap-2 rounded-full border border-rule bg-surface/80 px-3 py-1 text-caption text-slate backdrop-blur">
        <ShieldIcon width={13} height={13} className="text-brand" />
        {voice ? "Chat or call" : "Chat"}, every answer checked against the help center
      </span>
      <h2
        className="rise mt-5 max-w-[18ch] font-display text-[2.5rem] font-semibold leading-[1.05] tracking-tight text-ink sm:text-[3.5rem]"
        style={{ animationDelay: "60ms" }}
      >
        What&apos;s going on with your <span className="text-gradient">AcmeCloud</span> account?
      </h2>
      <p className="rise mt-4 max-w-[52ch] text-body text-slate" style={{ animationDelay: "120ms" }}>
        Acme Assist can check your plan, explain a policy, fix a sync error or issue a refund within policy.
        {voice ? " Type below, or start a voice call and just talk." : ""} Anything it can&apos;t confirm goes to a
        person.
      </p>

      <div className="mt-7 grid gap-3 sm:grid-cols-2">
        {SUGGESTIONS.slice(0, 4).map((s, i) => {
          const card = CARDS[i % CARDS.length];
          const Icon = card.icon;
          return (
            <button
              key={s}
              onClick={() => onPick(s)}
              style={{ animationDelay: `${180 + i * 60}ms` }}
              className="rise lift group flex flex-col items-start gap-3 rounded-2xl border border-rule bg-surface/90 p-4 text-left backdrop-blur hover:border-brand/40"
            >
              <span className="flex w-full items-center justify-between">
                <span className={`flex h-8 w-8 items-center justify-center rounded-lg ${card.tone}`}>
                  <Icon width={17} height={17} />
                </span>
                <span className="text-caption text-mist transition-transform group-hover:translate-x-0.5 group-hover:text-brand">
                  Try it &rarr;
                </span>
              </span>
              <span className="text-body leading-snug text-ink">{s}</span>
              <span className="text-caption uppercase tracking-wider text-mist">{card.tag}</span>
            </button>
          );
        })}
      </div>

      <ul className="rise mt-6 flex flex-wrap gap-x-5 gap-y-2" style={{ animationDelay: "460ms" }}>
        {CAPABILITIES.map(({ icon: Icon, label }) => (
          <li key={label} className="flex items-center gap-1.5 text-caption text-slate">
            <Icon width={14} height={14} className="text-brand" />
            {label}
          </li>
        ))}
      </ul>
    </div>
  );
}
