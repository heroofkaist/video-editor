# Contributing to Symposium Studio

We welcome contributions across video tooling adapters, research connectors, UI improvements, multilingual transcription/captioning, and animation templates.

## Engineering Standards
1. **No Fake Outputs or Fabricated Metrics**: Never invent transcript words, retention/CTR analytics, scientific measurements, or external video URLs.
2. **License Hygiene**: Core Python/TypeScript code is Apache-2.0. GPL or source-available optional renderers must remain isolated behind external subprocess/opt-in boundaries.
3. **Local-First Privacy**: Never transmit source video or user transcripts to third-party APIs unless explicitly opted in.
4. **Tests Required**: Add pytest coverage under `tests/` and verify real media rendering when modifying timeline, animation, or FFmpeg pipelines.

## Pull Request Checklist
- `pytest` passes without skipped assertions on changed components.
- `cd web && npm run build` succeeds cleanly.
- No large binary media files or personal recordings committed to Git.
