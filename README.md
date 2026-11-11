# Symposium Studio

**Open-Source AI Content Production, Creative Symposium, Programmatic Animation, Multi-Platform Editing & Community Growth Studio**

Symposium Studio turns raw source footage into an end-to-end, evidence-backed content campaign across **YouTube**, **YouTube Shorts**, **TikTok**, **Instagram Reels**, **Threads**, and **Telegram** — operable both by external AI agents (via REST API and JSON CLI) and human creators (via a 10-workspace web studio and OpenTimelineIO exports).

---

## Core Philosophy: Five Connected Engines

1. **Research Intelligence** — Discovers and indexes successful content mechanics across 16 categories (Entertainment, Gaming, Comedy, Science, Technology, Engineering, Programming, Education, Business, Startups, Sports, Lifestyle, Personal Storytelling, Documentaries, Popular Culture, Internet Trends) and 6 platforms via official APIs, public RSS/oEmbed feeds, and explicit URL/JSON imports.
2. **AI Symposium** — A 18-agent research & creative conference (7 Research Team + 11 Creative Team + Moderator) with `inexpensive`, `standard`, and `comprehensive` modes that debates hooks, narrative arc, animations, pacing, and scientific accuracy before selecting a traceable storyboard.
3. **Creative Production** — Local-first multilingual speech-to-text (`mlx-whisper` / `faster-whisper`), silence & shot detection (`PySceneDetect`), face-tracked framing (`OpenCV`), multi-factor highlight scoring, editable source-of-truth **OpenTimelineIO** (`.otio`) timelines, original procedural animations (**Headless Chromium Canvas/SVG**, **Motion Canvas**, **Manim**, **Matplotlib**, isolated **Blender**), synthesized SFX, and **FFmpeg** assembly.
4. **Multi-Platform Distribution** — Generates platform-native edits (16:9 YouTube, face-tracked 9:16 Shorts/TikTok/Reels with safe-zone captions), YouTube SEO packages & thumbnails, natural Threads posts, and a **Telegram Community Growth Engine** with tagged funnel links, technical deep-dive notes, and community polls.
5. **Growth Intelligence** — Evaluates authorized platform analytics and Telegram funnel conversions within each platform cohort to refine future creative recommendations.

---

## Quick Start

### Prerequisites
- Python `>=3.11` (Python `3.12` recommended on Apple Silicon)
- `ffmpeg` & `ffprobe` on `PATH`
- Node.js `>=20` (for Web UI and optional Motion Canvas / Playwright Chromium rendering)

### Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[apple-silicon,dev]"   # or ".[whisper-cpu,dev]" on Linux/Windows
```

Build the Web UI:
```bash
cd web && npm install && npm run build && cd ..
```

### Run End-to-End Campaign via CLI
```bash
symposium health
symposium campaign "Electromagnetic Induction Lab" ./my_footage.mp4 \
  --niche science_engineering \
  --style educational_explainer \
  --mode standard \
  --telegram-channel @symposium_lab
```

### Launch Web Studio & Agent REST API
```bash
symposium serve --host 127.0.0.1 --port 8420
# Open http://127.0.0.1:8420 (Studio UI) and http://127.0.0.1:8420/docs (OpenAPI)
```

---

## Iterative AI Agent Operations

Every step emits structured JSON over CLI and FastAPI:
```bash
symposium instruct proj_abc123 "Make the opening more engaging"
symposium instruct proj_abc123 "Make the captions smaller"
symposium instruct proj_abc123 "Create an animation showing how the magnetic field changes"
symposium instruct proj_abc123 "Turn this explanation into three shorts"
symposium instruct proj_abc123 "Generate a Threads post that leads people to the Telegram discussion"
```

---

## Documentation
- [Architecture Overview](docs/architecture.md)
- [Tool Evaluation Matrix](docs/tool-evaluation-matrix.md)
- [Platform APIs & Compliance](docs/platform-apis-and-compliance.md)
- [CLI & REST API Reference](docs/cli-and-api.md)
- [Plugin Development Guide](docs/plugin-development.md)
- [Benchmarks & Verification](docs/benchmarks.md)
- [Security Policy](SECURITY.md) · [Contributing](CONTRIBUTING.md) · [Roadmap](ROADMAP.md) · [Third-Party Notices](THIRD_PARTY_NOTICES.md)

## License
Apache License 2.0 (`LICENSE`). Optional external adapters (such as Blender `GPL-3.0` or Remotion source-available) remain strictly isolated outside the core package (`THIRD_PARTY_NOTICES.md`).
