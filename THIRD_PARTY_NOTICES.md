# Third-Party Notices & Optional Adapter Licensing

Symposium Studio core is licensed under the **Apache License 2.0** (`LICENSE`).

Because video production combines tools across permissive, copyleft, and source-available ecosystems, Symposium Studio strictly isolates optional or restrictive tools behind **adapter boundaries** (subprocess invocation or optional plugin packages) rather than linking them directly into the core codebase.

## Core Runtime Dependencies (Permissive / Compatible)

| Component | License | Integration Boundary |
|---|---|---|
| Python (`pydantic`, `fastapi`, `uvicorn`, `typer`, `rich`, `httpx`, `numpy`, `scipy`, `pillow`, `matplotlib`) | MIT / BSD / PSF | Core Python package |
| OpenTimelineIO (`opentimelineio`) | Apache-2.0 | Source-of-truth timeline interchange |
| PyAV (`av`) | BSD-3-Clause | Frame decoding, probe, stream analysis |
| PySceneDetect (`scenedetect`) | BSD-3-Clause | Shot & content cut detection |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | Face tracking, motion analysis, frame metrics |
| `mlx-whisper` | MIT | Local Apple Silicon speech-to-text |
| `faster-whisper` (optional fallback) | MIT | Cross-platform CPU/CUDA speech-to-text |
| Playwright + Chromium (HTML5 Canvas / SVG / WebGL renderer) | Apache-2.0 / BSD | Headless programmatic animation adapter |
| Motion Canvas (`@motion-canvas/*`, optional adapter) | MIT | Programmatic TypeScript 2D motion graphics |
| Manim Community (`manim`, optional adapter) | MIT | Math equations, plots, physics vector scenes |
| React / Next.js / Vite / Tailwind (Web UI) | MIT | Studio frontend |

## Copyleft / Dual-Licensed / Restricted Optional Adapters

These tools are **never bundled or linked** inside Symposium Studio core. They are executed strictly as **optional external subprocess adapters** when installed by the user and permitted by their policy:

1. **FFmpeg** (`ffmpeg` / `ffprobe`)
   - License: LGPL-2.1+ or GPL-2.0+ depending on build flags (`--enable-gpl` builds such as Homebrew's default build with `libx264`/`x265`).
   - Boundary: Invoked exclusively via CLI subprocess (`ffmpeg`, `ffprobe`), separate from Symposium Studio source code.
2. **Blender** (optional 3D adapter)
   - License: GPL-3.0.
   - Boundary: Invoked exclusively as an isolated external process (`blender -b -P script.py`). Not imported via `bpy` inside the Apache-2.0 core package.
3. **Remotion** (optional React video adapter)
   - License: Source-available custom license (free for individuals, non-profits, and for-profit organizations with <= 3 employees; paid Company License required for 4+ employees and certain automated rendering SaaS deployments).
   - Boundary: Opt-in isolated adapter disabled unless explicitly enabled (`SYMPOSIUM_ALLOW_REMOTION=1`) and installed externally by the user.
4. **MLT Framework / VapourSynth** (optional adapters)
   - License: LGPL-2.1+.
   - Boundary: External CLI (`melt`, `vspipe`) when present on host.
5. **Pre-trained Diarization Checkpoints** (e.g., Hugging Face `pyannote`)
   - License: Model weights carry separate Hugging Face terms/gating; Symposium Studio defaults to acoustic embedding clustering locally and only activates optional third-party checkpoints if explicitly configured by the user.
