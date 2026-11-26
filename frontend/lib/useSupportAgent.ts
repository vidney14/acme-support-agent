"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { API_URL, ApiError, api, streamChat, voiceSocketUrl } from "./api";
import { initialState, newId, reducer } from "./turns";
import type { AppConfig, Faults } from "./types";
import { VoiceSession, type VoiceStatus } from "./voice/voiceSession";

const SESSION_KEY = "acme-support-session";

export interface VoiceUi {
  status: VoiceStatus;
  muted: boolean;
  agentSpeaking: boolean;
  userSpeaking: boolean;
  error?: string;
}

export function useSupportAgent() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [connError, setConnError] = useState<string | null>(null);
  const [cooldownUntil, setCooldownUntil] = useState(0);
  const [voice, setVoice] = useState<VoiceUi>({ status: "idle", muted: false, agentSpeaking: false, userSpeaking: false });
  const voiceRef = useRef<VoiceSession | null>(null);

  // ---------------------------------------------------------------- bootstrap
  const startFresh = useCallback(async () => {
    const snap = await api.createSession();
    localStorage.setItem(SESSION_KEY, snap.session_id);
    setSessionId(snap.session_id);
    dispatch({ type: "reset" });
    dispatch({ type: "snapshot", snapshot: snap });
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cfg = await api.config();
        if (cancelled) return;
        setConfig(cfg);
        const saved = localStorage.getItem(SESSION_KEY);
        if (saved) {
          try {
            const [sess, tr] = await Promise.all([api.session(saved), api.traces(saved)]);
            if (cancelled) return;
            setSessionId(saved);
            dispatch({ type: "restore", messages: sess.messages, traces: tr.traces, snapshot: sess });
            return;
          } catch (e) {
            if (!(e instanceof ApiError && e.status === 404)) throw e;
          }
        }
        await startFresh();
      } catch {
        if (!cancelled) setConnError(`Can't reach the support service at ${API_URL}. Check that the backend is running.`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [startFresh]);

  // ---------------------------------------------------------------- chat
  /** The stream dropped mid-turn. The turn keeps running server-side, so ask the
   *  server for the finished message before declaring failure. */
  const recover = useCallback(async (turnId: string, serverTurnId?: string) => {
    if (sessionId && serverTurnId) {
      for (let i = 0; i < 4; i++) {
        await new Promise((r) => setTimeout(r, 1200));
        try {
          const sess = await api.session(sessionId);
          const msg = sess.messages.find((m) => m.turn_id === serverTurnId && m.role === "assistant");
          if (msg) {
            dispatch({ type: "recovered", id: turnId, text: msg.text, grounding: msg.meta.grounding, sources: msg.meta.sources });
            dispatch({ type: "snapshot", snapshot: sess });
            const tr = await api.traces(sessionId);
            dispatch({ type: "traces", traces: tr.traces });
            return;
          }
        } catch {
          /* keep trying */
        }
      }
    }
    dispatch({
      type: "turn_error",
      id: turnId,
      error: { message: "The connection dropped before the reply finished. Retry to continue.", recoverable: true, kind: "stream" },
    });
  }, [sessionId]);

  const runChat = useCallback(
    async (turnId: string, text: string, retry: boolean) => {
      setBusy(true);
      let serverTurnId: string | undefined;
      try {
        const completed = await streamChat({ session_id: sessionId ?? undefined, message: text, retry }, (event, data) => {
          if (event === "turn_start") serverTurnId = data.turn_id as string;
          dispatch({ type: "event", id: turnId, event, data });
        });
        if (!completed) await recover(turnId, serverTurnId);
      } catch (e) {
        if (e instanceof ApiError && e.status === 429) {
          const wait = Math.ceil(e.retryAfterS ?? 5);
          setCooldownUntil(Date.now() + wait * 1000);
          dispatch({
            type: "turn_error",
            id: turnId,
            error: { message: `You're sending messages quickly. Try again in ${wait}s.`, recoverable: true, kind: "rate_limited", retry_after_s: wait },
          });
        } else if (serverTurnId) {
          await recover(turnId, serverTurnId);
        } else {
          dispatch({
            type: "turn_error",
            id: turnId,
            error: { message: "Couldn't reach the support service. Check your connection, then retry.", recoverable: true, kind: "network" },
          });
        }
      } finally {
        setBusy(false);
      }
    },
    [sessionId, recover],
  );

  const send = useCallback(
    async (text: string) => {
      const msg = text.trim();
      if (!msg) return;
      if (voiceRef.current && voice.status === "live") {
        voiceRef.current.sendText(msg); // typed input during a call goes to the voice agent
        return;
      }
      if (busy || Date.now() < cooldownUntil) return;
      const id = newId();
      dispatch({ type: "turn_start", id, user: msg, modality: "text" });
      await runChat(id, msg, false);
    },
    [busy, cooldownUntil, runChat, voice.status],
  );

  const retry = useCallback(
    async (turnId: string) => {
      const t = state.turns.find((x) => x.id === turnId);
      if (!t || busy) return;
      dispatch({ type: "turn_retry", id: turnId });
      await runChat(turnId, t.user, t.error?.kind !== "rate_limited" && t.error?.kind !== "network");
    },
    [busy, runChat, state.turns],
  );

  const newSession = useCallback(async () => {
    voiceRef.current?.stop();
    try {
      await startFresh();
    } catch {
      setConnError(`Can't reach the support service at ${API_URL}.`);
    }
  }, [startFresh]);

  const setFaults = useCallback(
    async (f: Partial<Faults>) => {
      if (!sessionId) return;
      const faults = await api.setFaults(sessionId, f);
      if (state.snapshot) dispatch({ type: "snapshot", snapshot: { ...state.snapshot, faults } });
    },
    [sessionId, state.snapshot],
  );

  // ---------------------------------------------------------------- voice
  const startVoice = useCallback(async () => {
    if (!sessionId || voiceRef.current) return;
    const vs = new VoiceSession({
      url: voiceSocketUrl(sessionId),
      onEvent: (event, data) => dispatch({ type: "voice_event", event, data }),
      onState: (p) => setVoice((v) => ({ ...v, ...p })),
    });
    voiceRef.current = vs;
    setVoice({ status: "connecting", muted: false, agentSpeaking: false, userSpeaking: false });
    await vs.start();
  }, [sessionId]);

  const stopVoice = useCallback(() => {
    voiceRef.current?.stop();
    voiceRef.current = null;
    setVoice((v) => ({ ...v, status: "idle", agentSpeaking: false, userSpeaking: false }));
  }, []);

  const toggleMute = useCallback(() => {
    const vs = voiceRef.current;
    if (!vs) return;
    const muted = !voice.muted;
    vs.setMuted(muted);
    setVoice((v) => ({ ...v, muted }));
  }, [voice.muted]);

  // A voice session that ends on its own (error, server close) releases the ref.
  useEffect(() => {
    if ((voice.status === "ended" || voice.status === "error") && voiceRef.current) {
      voiceRef.current.stop();
      voiceRef.current = null;
    }
  }, [voice.status]);

  useEffect(() => () => voiceRef.current?.stop(), []);

  return {
    state,
    config,
    sessionId,
    busy,
    connError,
    cooldownUntil,
    voice,
    voiceSession: voiceRef,
    send,
    retry,
    newSession,
    setFaults,
    startVoice,
    stopVoice,
    toggleMute,
  };
}
