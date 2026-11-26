# Frontend (Next.js)

```bash
cp .env.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev                  # http://localhost:3000
```

- `lib/api.ts`: REST + POST-SSE parser
- `lib/turns.ts`: pure reducer folding chat/voice events into turns
- `lib/useSupportAgent.ts`: session lifecycle, retry/recovery, voice
- `lib/voice/voiceSession.ts` + `public/worklets/pcm-capture.js`: mic capture, playback, barge-in
- `components/`: ToolTrack, ConfidenceGauge, AnswerText, TurnView, CallBar, TraceDrawer, DemoGuide

See the root README for architecture.
