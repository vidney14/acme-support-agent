// Browser side of the voice call.
//
//   mic -> AudioWorklet (16 kHz PCM16) -> WebSocket (binary) -> backend -> Gemini Live
//   speaker <- scheduled AudioBufferSources <- WebSocket (binary, 24 kHz PCM16) <- backend
//   JSON events (transcripts, tool steps, interrupted, ...) -> onEvent -> shared turn reducer
//
// Interruption handling happens in two layers:
//  1. Local, instant: when the mic gets loud while the agent is talking, duck the agent's
//     volume right away (feels responsive even before the server reacts).
//  2. Authoritative: when Gemini's VAD confirms barge-in it sends `interrupted`; we flush
//     every scheduled audio buffer, so the agent stops mid-word.

export type VoiceStatus = "idle" | "connecting" | "live" | "ended" | "error";

export interface VoiceStatePatch {
  status?: VoiceStatus;
  agentSpeaking?: boolean;
  userSpeaking?: boolean;
  error?: string;
}

interface Options {
  url: string;
  onEvent: (type: string, data: Record<string, unknown>) => void;
  onState: (patch: VoiceStatePatch) => void;
}

const OUT_RATE = 24000;
const SPEECH_RMS = 0.02;

export class VoiceSession {
  micAnalyser: AnalyserNode | null = null;
  outAnalyser: AnalyserNode | null = null;
  private ws: WebSocket | null = null;
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private outGain: GainNode | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private nextTime = 0;
  private muted = false;
  private stopped = false;
  private live = false;
  private loud = 0;
  private quiet = 0;
  private userSpeaking = false;
  private agentSpeaking = false;
  private ducked = false;

  constructor(private opts: Options) {}

  async start(): Promise<void> {
    try {
      this.ctx = new AudioContext({ latencyHint: "interactive" });
      await this.ctx.resume();
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      });
      await this.ctx.audioWorklet.addModule("/worklets/pcm-capture.js");
    } catch (e) {
      const denied = e instanceof DOMException && (e.name === "NotAllowedError" || e.name === "SecurityError");
      this.fail(
        denied
          ? "Microphone access was blocked. Allow the microphone in your browser's site settings, then start the call again."
          : "Couldn't start the microphone on this device.",
      );
      return;
    }
    if (this.stopped) return;

    const ctx = this.ctx!;
    const src = ctx.createMediaStreamSource(this.stream!);
    this.micAnalyser = ctx.createAnalyser();
    this.micAnalyser.fftSize = 1024;
    src.connect(this.micAnalyser);

    this.node = new AudioWorkletNode(ctx, "pcm-capture", { processorOptions: { targetRate: 16000 } });
    const sink = ctx.createGain();
    sink.gain.value = 0; // keep the worklet pulled by the graph without echoing the mic
    src.connect(this.node);
    this.node.connect(sink).connect(ctx.destination);
    this.node.port.onmessage = (e) => this.onMicFrame(e.data as { pcm: ArrayBuffer; rms: number });

    this.outGain = ctx.createGain();
    this.outAnalyser = ctx.createAnalyser();
    this.outAnalyser.fftSize = 1024;
    this.outGain.connect(this.outAnalyser).connect(ctx.destination);

    const ws = new WebSocket(this.opts.url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onmessage = (e) => {
      if (typeof e.data === "string") this.onJson(JSON.parse(e.data));
      else this.play(e.data as ArrayBuffer);
    };
    ws.onerror = () => {
      if (!this.stopped && !this.live) this.fail("Couldn't connect to the voice service.");
    };
    ws.onclose = () => {
      if (this.stopped) return;
      this.flush();
      this.opts.onState({ status: this.live ? "ended" : "error", agentSpeaking: false, userSpeaking: false });
    };
  }

  setMuted(muted: boolean) {
    this.muted = muted;
    this.stream?.getAudioTracks().forEach((t) => (t.enabled = !muted));
    if (muted) this.sendJson({ type: "audio_stream_end" }); // lets server VAD close the turn
  }

  sendText(text: string) {
    this.sendJson({ type: "text", text });
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    this.sendJson({ type: "stop" });
    this.ws?.close();
    this.flush();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.node?.port.close();
    this.ctx?.close().catch(() => {});
    this.micAnalyser = this.outAnalyser = null;
  }

  // ------------------------------------------------------------------ internals
  private fail(message: string) {
    this.opts.onState({ status: "error", error: message, agentSpeaking: false, userSpeaking: false });
    this.stop();
  }

  private sendJson(msg: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  private onJson(msg: Record<string, unknown>) {
    const { type, ...data } = msg as { type: string } & Record<string, unknown>;
    switch (type) {
      case "ready":
        this.live = true;
        this.opts.onState({ status: "live", error: undefined });
        return;
      case "interrupted":
        this.flush(); // authoritative barge-in: stop the agent mid-word
        this.opts.onEvent("interrupted", data);
        return;
      case "error":
        this.opts.onState({ status: "error", error: String(data.message ?? "Voice error") });
        return;
      case "closed":
        this.opts.onState({
          status: "ended",
          error: data.reason === "idle" ? "Call ended after 2 minutes of silence." : "Call reached the 10-minute limit.",
        });
        return;
      default:
        this.opts.onEvent(type, data);
    }
  }

  private onMicFrame({ pcm, rms }: { pcm: ArrayBuffer; rms: number }) {
    // Local voice-activity estimate (hysteresis: start fast, stop slow).
    if (rms > SPEECH_RMS) {
      this.loud++;
      this.quiet = 0;
    } else {
      this.quiet++;
      this.loud = 0;
    }
    if (!this.userSpeaking && this.loud >= 2 && !this.muted) this.setUserSpeaking(true);
    if (this.userSpeaking && this.quiet >= 10) this.setUserSpeaking(false);

    if (this.live && !this.muted && this.ws?.readyState === WebSocket.OPEN) this.ws.send(pcm);
  }

  private setUserSpeaking(v: boolean) {
    this.userSpeaking = v;
    this.opts.onState({ userSpeaking: v });
    // Optimistic duck: lower the agent's voice the moment the customer starts talking.
    if (v && this.agentSpeaking) this.duck(true);
    if (!v) this.duck(false);
  }

  private duck(on: boolean) {
    if (!this.outGain || !this.ctx || this.ducked === on) return;
    this.ducked = on;
    this.outGain.gain.setTargetAtTime(on ? 0.25 : 1, this.ctx.currentTime, 0.03);
  }

  private play(buf: ArrayBuffer) {
    const ctx = this.ctx;
    if (!ctx || !this.outGain || buf.byteLength < 2) return;
    const pcm = new Int16Array(buf, 0, Math.floor(buf.byteLength / 2));
    const audio = ctx.createBuffer(1, pcm.length, OUT_RATE); // browser resamples to device rate
    const ch = audio.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 0x8000;
    const src = ctx.createBufferSource();
    src.buffer = audio;
    src.connect(this.outGain);
    const at = Math.max(ctx.currentTime + 0.03, this.nextTime); // gapless queue
    src.start(at);
    this.nextTime = at + audio.duration;
    this.sources.add(src);
    if (!this.agentSpeaking) {
      this.agentSpeaking = true;
      this.opts.onState({ agentSpeaking: true });
    }
    src.onended = () => {
      this.sources.delete(src);
      if (this.sources.size === 0) {
        this.agentSpeaking = false;
        this.duck(false);
        this.opts.onState({ agentSpeaking: false });
      }
    };
  }

  private flush() {
    for (const s of this.sources) {
      try {
        s.onended = null;
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear();
    this.nextTime = 0;
    this.duck(false);
    if (this.agentSpeaking) {
      this.agentSpeaking = false;
      this.opts.onState({ agentSpeaking: false });
    }
  }
}
