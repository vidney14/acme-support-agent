// AudioWorklet: microphone Float32 at the device rate (usually 48 kHz)
// -> 16 kHz mono PCM16 little-endian frames, which is what Gemini Live expects.
//
// Runs on the audio rendering thread, so capture never stutters when React re-renders.
// Downsampling uses box-filter averaging (cheap anti-aliasing, fine for speech).
// Each frame also carries an RMS level, used for the waveform and local barge-in detection.

class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const target = (options && options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.ratio = sampleRate / target; // `sampleRate` is a global in AudioWorkletGlobalScope
    this.frame = Math.round(target * 0.04); // 40 ms per WebSocket message
    this.out = new Int16Array(this.frame);
    this.idx = 0;
    this.t = 0;
    this.acc = 0;
    this.n = 0;
    this.sumSq = 0;
    this.count = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      const x = ch[i];
      this.acc += x;
      this.n++;
      this.sumSq += x * x;
      this.count++;
      this.t += 1;
      if (this.t >= this.ratio) {
        this.t -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.n));
        this.acc = 0;
        this.n = 0;
        this.out[this.idx++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.idx === this.frame) {
          const rms = Math.sqrt(this.sumSq / Math.max(1, this.count));
          this.port.postMessage({ pcm: this.out.buffer, rms }, [this.out.buffer]);
          this.out = new Int16Array(this.frame);
          this.idx = 0;
          this.sumSq = 0;
          this.count = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
