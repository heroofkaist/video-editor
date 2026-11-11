# CLI & REST API Reference

## CLI (`symposium`)
All commands emit machine-readable JSON on `stdout`.

| Command | Purpose |
|---|---|
| `symposium health` | Inspect workspace config, adapter health, and platform connectors |
| `symposium create <name> --niche ... --style ...` | Initialize structured workspace |
| `symposium ingest <project_id> <video_path>` | Copy/hash raw footage into `source/` |
| `symposium analyze <project_id> [--language xx]` | Transcribe, diarize, detect scenes/faces, score highlights |
| `symposium research-search [--query ...] [--category ...]` | Search cross-niche Viral Intelligence Database |
| `symposium research-discover [--target-count 100]` | Discover public multi-niche creator videos via official Atom feeds / YouTube API |
| `symposium research-inspect <url>` | Inspect a public YouTube URL via official oEmbed metadata |
| `symposium style-clone <project_id> <reference_video>` | Extract pacing & color signature from a reference video |
| `symposium telegram-sync <project_id> <chat_id>` | Sync official subscriber counts via `TELEGRAM_BOT_TOKEN` |
| `symposium growth-experiments [--q ...] [--platform ...]` | Search recorded growth experiments & learned improvement hypotheses |
| `symposium growth-evaluate <project_id>` | Compare post-publication metrics vs creator same-platform medians & calibrate predictions |
| `symposium symposium <project_id> [--mode inexpensive\|standard\|comprehensive]` | Run multi-agent conference, 11-factor candidate prediction, Creative Director gate & storyboard |
| `symposium animate <project_id> [--template ...] [--adapter ...]` | Build `SceneSpec` & render original animation |
| `symposium render <project_id> [--short-variants N]` | Build `.otio` timelines, render 16:9 + 9:16 MP4s + thumbnail + QA |
| `symposium instruct <project_id> "<instruction>"` | Apply iterative natural-language edit / animation / shorts / social command |
| `symposium repurpose <project_id> [--telegram-channel ...]` | Generate YouTube metadata, Threads posts, Shorts copy, Telegram campaign |
| `symposium campaign <name> <video_path>` | Run complete end-to-end pipeline |
| `symposium search <query>` | SQLite FTS5 search across projects, transcripts, research, and assets |
| `symposium serve --port 8420` | Start FastAPI server & Web UI |

## FastAPI Endpoints (`/docs` for full OpenAPI 3.1 schemas)
- `GET /api/health`
- `GET /api/projects`, `POST /api/projects`, `GET /api/projects/{id}`
- `POST /api/projects/{id}/ingest`, `POST /api/projects/{id}/upload`
- `POST /api/projects/{id}/analyze`, `GET /api/projects/{id}/analysis`
- `GET /api/research/items`, `POST /api/research/items`, `POST /api/research/discover`, `POST /api/research/inspect-url`, `POST /api/projects/{id}/research`
- `GET /api/symposium/agents`, `POST /api/projects/{id}/symposium`, `POST /api/projects/{id}/style-from-reference`
- `GET /api/animations/adapters`, `POST /api/projects/{id}/animate`, `GET /api/effects`
- `POST /api/projects/{id}/render`, `POST /api/projects/{id}/instruct`
- `POST /api/projects/{id}/social`, `POST /api/projects/{id}/publish`, `POST /api/projects/{id}/report`
- `POST /api/projects/{id}/growth`, `GET /api/growth`, `POST /api/growth/{id}/import`, `POST /api/growth/{id}/telegram-sync`, `POST /api/growth/{id}/evaluate`, `GET /api/growth/experiments`, `POST /api/funnel/{link_id}/event`, `GET /r/{link_id}`, `GET /api/search`
