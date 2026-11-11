# Phase 0: Tool Evaluation Matrix & Architecture Decisions

This evaluation reflects hands-on testing on macOS (`arm64`, Apple M5 Pro, 24 GB unified memory, macOS 27.2.1), Python 3.12.14, Node 26.10.0, and FFmpeg 9.0.2, combined with license and maintenance verification.

## 1. Core Video & Timeline Processing

| Tool | Role | Quality / Maturity | Performance on Apple Silicon | License | macOS ARM64 Status | Decision |
|---|---|---|---|---|---|---|
| **FFmpeg 9.0.2** | Encoding, filtergraphs, audio `loudnorm`, `sidechaincompress`, `libass` burn-in, final muxing | Industry standard, ultra-mature | Very fast (`h264_videotoolbox` / `hevc_videotoolbox` HW + `libx264` SW) | LGPL/GPL (CLI build) | Preinstalled (`/opt/homebrew/bin/ffmpeg`) | **Core Subprocess Engine** (assembly & final encode, never sole editorial logic) |
| **OpenTimelineIO 0.18.1** | Structured editable timeline source of truth | Academy Software Foundation standard, active | Native Python + C++ bindings | Apache-2.0 | Native PyPI `arm64` wheel | **Selected** as canonical timeline interchange (`.otio` + Studio JSON) |
| **PyAV 19.0.2** | Packet/frame-level probing, frame extraction, exact timestamps | Mature FFmpeg Cython bindings | Native C speed | BSD-3-Clause | Native PyPI `arm64` wheel | **Selected** for frame inspection, verification & media probing |
| **PySceneDetect 0.7.3** | Shot boundary & cut detection | Mature, clean Python API | Fast frame scanning | BSD-3-Clause | Native PyPI `arm64` wheel | **Selected** for shot/scene segmentation |
| **OpenCV 5.0 / 4.x (`opencv-python-headless`)** | Face tracking, motion scoring, blur/sharpness, framing | Mature | Fast NEON/ARM64 kernels | Apache-2.0 | Native PyPI `arm64` wheel | **Selected** for face-aware 9:16 reframing & visual context |
| **MLT (`melt`)** | NLE engine | Mature (Shotcut/Kdenlive) | CPU-bound | LGPL-2.1+ | No official macOS `arm64` PyPI wheel; requires Homebrew build | **Optional External Adapter** (not core dependency) |
| **VapourSynth** | Frame server & filtering | Mature | Fast scripted frame filters | LGPL-2.1 | Conda-Forge only for reliable `arm64` | **Optional External Adapter** (not core dependency) |

## 2. Speech, Audio & Visual Semantics

| Tool | Role | Multilingual Support | Performance on Apple Silicon | License | Decision |
|---|---|---|---|---|---|
| **`mlx-whisper` 0.4.3** | Local transcription + word timestamps + language ID | Excellent (`en`, `ru`, `kk`, `ko`, mixed) | Uses Apple Metal + Unified Memory via MLX (`whisper-small-mlx` & `whisper-large-v3-mlx` cached) | MIT | **Primary Local Transcriber** on Apple Silicon |
| **`faster-whisper` 1.2.1** | Cross-platform transcription fallback | Excellent | CPU/OpenMP on macOS (CUDA on Linux/Win) | MIT | **Cross-Platform Fallback** |
| **Acoustic Diarization + `silero-vad` / `sherpa-onnx`** | Silence detection, turn detection, speaker clustering | Language-agnostic | Lightweight local execution | Apache-2.0 / MIT | **Selected** (zero gated weights required by default; optional `pyannote` hook) |
| **Apple Vision (`pyobjc`) / OpenCV** | Local OCR & face/saliency tracking | Multilingual OCR | Hardware ANE/GPU accelerated on macOS | Apple SDK / MIT | **Selected** for macOS OCR/vision with cross-platform OpenCV fallback |

## 3. Programmatic Animation & 3D Rendering Adapters

Symposium Studio implements an interchangeable **Unified Render Graph** with explicit adapter metadata (capabilities, constraints, licensing, alpha-channel support):

| Adapter | Best For | Alpha Channel / Output | License | macOS ARM64 Status | Decision |
|---|---|---|---|---|---|
| **Chromium Canvas / SVG / WebGL (`playwright`)** | Keyframed 2D motion graphics, physics field lines, circuits, data structures, timelines, kinetic UI cards, particle effects | Native RGBA transparent PNG sequence (`png_rgba`) + WebM/MP4 | Apache-2.0 / BSD | Cached `chromium_headless_shell-1248` ready | **Primary Built-in Animation Renderer** |
| **Motion Canvas (`@motion-canvas/core` + `2d`)** | TypeScript generator-driven vector explainers & technical scenes | PNG sequence / MP4 via headless Vite exporter | MIT | Supported via Node 26 + Playwright | **First-Class Plugin Adapter** |
| **Matplotlib / Cairo Vector** | Exact factual scientific plots, equations, circuit schematics, fast preview renders | RGBA PNG sequence / SVG / MP4 | PSF / LGPL / MIT | Native Python `arm64` | **Fast Scientific Plot Adapter** |
| **Manim Community (`manim` 0.22.1)** | Advanced mathematical derivations, equations, geometric proofs | Transparent MOV/PNG/MP4 | MIT | Requires `brew install pango` (and optional MacTeX for TeX) | **Optional Plugin Adapter** (detected dynamically via health check) |
| **Blender** | Procedural 3D scenes, camera rigs, PBR lighting, 3D motion tracking | EXR / RGBA PNG / ProRes4444 / MP4 | **GPL-3.0** | Not currently installed on host | **Isolated Optional Subprocess Adapter** (GPL isolation preserved) |
| **Remotion 4.x** | React-based compositions & reusable video templates | PNG / MP4 / ProRes | **Source-available (paid for >=4 employees)** | Works via Node 26 | **Opt-in Adapter Only** (explicit license policy flag required) |

## 4. Backend, Storage & Interface Stack

- **Backend & Agent API:** Python 3.12 + FastAPI + Pydantic v2 + Uvicorn (`OpenAPI 3.1` schemas + machine-readable CLI via Typer).
- **Database & Search:** SQLite 3.54 with **FTS5** full-text indexing across projects, transcripts, viral intelligence corpus, and assets; designed behind a repository interface so a **PostgreSQL 18** adapter drops in cleanly.
- **Web Interface:** React + TypeScript + Vite + Tailwind CSS in `web/`, built to static assets or run in dev mode, connecting to FastAPI for real media playback, timeline editing, animation preview, viral database search, symposium inspection, and multi-platform export.
