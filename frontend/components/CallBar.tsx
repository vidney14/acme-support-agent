"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import { MicIcon, MicOffIcon, PhoneOffIcon, SendIcon } from "./icons";
import type { VoiceUi } from "@/lib/useSupportAgent";
import type { VoiceSession } from "@/lib/voice/voiceSession";

interface Props {
  voice: VoiceUi;
  simulated?: boolean;
  session: RefObject<VoiceSession | null>;
  onEnd: () => void;
  onMute: () => void;
  onRetry: () => void;
  onSendText: (t: string) => void;
}

export function CallBar({ voice, simulated, session, onEnd, onMute, onRetry, onSendText }: Props) {
  const [typed, setTyped] = useState("");
  const [seconds, setSeconds] = useState(0);
  const live = voice.status === "live";

  useEffect(() => {
    if (!live) return;
    const start = Date.now();
    const t = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(t);
  }, [live]);

  if (voice.status === "error" || voice.status === "ended") {
    return (
      <div className="rounded-2xl border border-rule bg-surface px-4 py-3" role="status">
        <p className="text-small text-ink">
          {voice.status === "ended" ? "The call ended." : "The voice call couldn't continue."}{" "}
          <span className="text-slate">{voice.error ?? "You can keep going by text; the agent remembers the call."}</span>
        </p>
        <div className="mt-2 flex gap-2">
          <button onClick={onRetry} className="rounded-xl bg-ink px-3 py-1.5 text-small font-semibold text-paper hover:bg-ink/85">
            Start the call again
          </button>
          <button onClick={onEnd} className="rounded-xl px-3 py-1.5 text-small font-semibold text-ink hover:bg-paper">
            Back to chat
          </button>
        </div>
      </div>
    );
  }

  const status =
    voice.status === "connecting"
      ? "Connecting the call"
      : voice.muted
        ? "Muted"
        : voice.agentSpeaking && voice.userSpeaking
          ? "Talking over the agent"
          : voice.agentSpeaking
            ? "Agent speaking"
            : voice.userSpeaking
              ? "Listening"
              : "Your turn to talk";

  const mm = String(Math.floor(seconds / 60)).padStart(1, "0");
  const ss = String(seconds % 60).padStart(2, "0");

  return (
    <div className="rise rounded-2xl border border-brand/40 bg-surface p-3 shadow-[0_0_0_4px_color-mix(in_srgb,var(--color-brand)_10%,transparent),0_16px_36px_-18px_var(--color-brand)]">
      <div className="flex items-center gap-3">
        <div className="w-40 shrink-0 sm:w-48">
          <p className="text-small font-semibold text-ink" aria-live="polite">
            {status}
          </p>
          <p className="tnum text-caption text-mist">{live ? `Voice call, ${mm}:${ss}` : "Allow the microphone"}</p>
          {simulated && live && <p className="text-caption text-work">Simulated call</p>}
        </div>
        <Seismograph session={session} active={live} />
        <button
          onClick={onMute}
          disabled={!live}
          aria-pressed={voice.muted}
          aria-label={voice.muted ? "Unmute microphone" : "Mute microphone"}
          className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full border ${voice.muted ? "border-work bg-work-soft text-work" : "border-rule text-ink hover:bg-paper"}`}
        >
          {voice.muted ? <MicOffIcon /> : <MicIcon />}
        </button>
        <button
          onClick={onEnd}
          aria-label="End call"
          className="flex h-10 shrink-0 items-center gap-1.5 rounded-full bg-fault px-4 text-small font-semibold text-surface hover:bg-fault/90"
        >
          <PhoneOffIcon /> <span className="hidden sm:inline">End call</span>
        </button>
      </div>
      <form
        className="mt-2 flex items-center gap-2 border-t border-rule-soft pt-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (typed.trim() && live) {
            onSendText(typed);
            setTyped("");
          }
        }}
      >
        <label htmlFor="call-text" className="sr-only">
          Type during the call
        </label>
        <input
          id="call-text"
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          disabled={!live}
          placeholder="Type during the call, e.g. an email address"
          className="min-w-0 flex-1 bg-transparent px-1 text-small text-ink placeholder:text-mist focus:outline-none focus-visible:outline-none"
        />
        <button type="submit" disabled={!typed.trim() || !live} aria-label="Send typed message" className="text-ink disabled:text-mist">
          <SendIcon width={16} height={16} />
        </button>
      </form>
    </div>
  );
}

/** Scrolling two-sided level history: agent above the line, customer below. */
function Seismograph({ session, active }: { session: RefObject<VoiceSession | null>; active: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvas.current;
    if (!el || !active) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const css = getComputedStyle(document.documentElement);
    const work = css.getPropertyValue("--color-work").trim() || "#b86e00";
    const ink = css.getPropertyValue("--color-ink").trim() || "#16202a";
    const rule = css.getPropertyValue("--color-rule").trim() || "#d8dee4";
    const step = 4;
    let agent: number[] = [];
    let user: number[] = [];
    const bufA = new Uint8Array(1024);
    const bufU = new Uint8Array(1024);
    let raf = 0;
    let last = 0;

    const level = (an: AnalyserNode | null, buf: Uint8Array<ArrayBuffer>) => {
      if (!an) return 0;
      an.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128;
        sum += v * v;
      }
      return Math.min(1, Math.sqrt(sum / buf.length) * 4);
    };

    const draw = (t: number) => {
      raf = requestAnimationFrame(draw);
      if (t - last < 33) return; // ~30 fps is plenty
      last = t;
      const dpr = window.devicePixelRatio || 1;
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (el.width !== w * dpr) {
        el.width = w * dpr;
        el.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const vs = session.current;
      const cap = Math.floor(w / step);
      agent = [...agent, level(vs?.outAnalyser ?? null, bufA)].slice(-cap);
      user = [...user, level(vs?.micAnalyser ?? null, bufU)].slice(-cap);
      ctx.clearRect(0, 0, w, h);
      const mid = h / 2;
      ctx.fillStyle = rule;
      ctx.fillRect(0, mid, w, 1);
      const x0 = w - agent.length * step;
      for (let i = 0; i < agent.length; i++) {
        const x = x0 + i * step;
        const a = Math.max(0.5, agent[i] * (mid - 2));
        const u = Math.max(0.5, user[i] * (mid - 2));
        ctx.fillStyle = work;
        ctx.fillRect(x, mid - a, step - 1.5, a);
        ctx.fillStyle = ink;
        ctx.fillRect(x, mid + 1, step - 1.5, u);
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [session, active]);

  return (
    <div className="relative h-12 min-w-0 flex-1">
      <canvas ref={canvas} className="h-full w-full" aria-hidden />
      <span className="pointer-events-none absolute left-1 top-0 text-[0.65rem] text-work">Agent</span>
      <span className="pointer-events-none absolute bottom-0 left-1 text-[0.65rem] text-slate">You</span>
    </div>
  );
}
