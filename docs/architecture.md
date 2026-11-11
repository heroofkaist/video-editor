# Symposium Studio Architecture

```
                         +------------------------------+
                         |   External AI Agents / CLI   |
                         |   React Web Studio (10 tabs) |
                         +--------------+---------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                      FastAPI Backend & SymposiumStudio Facade                 |
+---+--------------------+--------------------+--------------------+------------+
    |                    |                    |                    |
    v                    v                    v                    v
[1. Research]      [2. AI Symposium]   [3. Production]      [4. Distribution & Growth]
- Viral DB (FTS5)  - 7 Research Agents - MLX/Faster Whisper - YouTube SEO + Thumbnail
- Cross-Niche      - 11 Creative Agents- PySceneDetect      - Threads (6 tone profiles)
  Pattern Engine   - Moderator Checks  - OpenCV Face Track  - Telegram Funnel + Polls
- Official APIs    - 12-Step Traceable - OTIO + RenderGraph - Isolated Publishers
  + Explicit         Storyboard          - Animation Registry   (export/draft/approved)
  Imports                                (Chromium, Motion    - Growth Intelligence
                                          Canvas, Manim,
                                          Matplotlib, Blender)
```

## Workspace Structure (`workspaces/<slug>/`)
Every project creates an isolated, Git-ignored workspace with complete SHA-256 provenance in `manifest.json`:
- `source/` — unmodified original media
- `transcripts/` — `analysis_bundle.json` (verbatim transcript, word timings, confidence flags, topics, highlights)
- `research/` — evidence-backed research reports
- `symposium/` — multi-agent proposals, moderator challenges, decisions
- `storyboards/` — selected narrative beat plans
- `timelines/` — canonical `RenderTimeline` JSON + OpenTimelineIO `.otio` + `.srt`/`.ass`
- `animations/` — editable `.spec.json` + HTML/TSX/PY sources + alpha `.mov` / `.mp4` renders
- `effects/` — procedural DSP sound effects (`.wav`)
- `renders/` — 16:9 YouTube master MP4s
- `shorts/` — face-tracked 9:16 Shorts / TikTok / Reels MP4s
- `thumbnails/` — 1280x720 rendered thumbnails
- `social/` & `telegram/` — copy packages, funnel links, validated publish payloads
- `reports/` — automated QA reports, extracted QA check frames, `production_report.md` & `.json`
