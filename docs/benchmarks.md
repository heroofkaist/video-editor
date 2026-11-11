# Measured Benchmarks & Quality Assurance Log

Recorded from real end-to-end execution of Symposium Studio `v0.2.0` on Apple Silicon hardware.

## Reference Test Environment

- **Machine**: macOS 27.2.1 (`arm64` / Apple M5 Pro, 24 GB unified memory)
- **Python**: `3.12.14` (`.venv`)
- **Node.js**: `v26.10.0` (with local `playwright-core` driving cached `chrome-headless-shell`)
- **FFmpeg**: `9.0.2` (`libx264`, `aac`, `prores_ks`, `libass`)
- **Local ASR Engine**: `whisper.cpp` `1.9.5` (`whisper-cli`, `MTL0` Metal GPU backend, `ggml-base.bin`)

## Observed End-to-End Timings

1. **Test Fixture Generation** (`scripts/generate_test_fixture.py`):
   - Real 3-minute (`194.542s`) `1280x720` @ `24 fps` talking-head H.264/AAC MP4 synthesized via macOS `say` (`Daniel`) + OpenCV frame generation: **`12.48s`** (measured on fresh fixture generation run).

2. **Full 15-Deliverable Campaign Pipeline** (`scripts/run_acceptance_test.py` with `ResearchMode.COMPREHENSIVE`):
   - **End-to-end execution**: **`42.38s`** campaign pipeline + **`13.65s`** iterative re-render (`"Make the captions smaller"`), total **`56.14s`**.
   - Stages executed in sequence:
     - Media probe + `whisper.cpp` (`ggml-base.bin` Metal) ASR with word timestamps and language auto-detection (`en`)
     - Audio silence detection & spectral speaker turn diarization
     - PySceneDetect cut detection + OpenCV face/skin tracking (`recommended_crop_center_x = 0.3707`) + sharpness/motion sampling + Apple Vision OCR sampling
     - Semantic clustering & 8-factor highlight scoring (`18` highlights extracted)
     - Cross-niche Research Intelligence FTS5 query + `18`-agent AI Symposium comprehensive deliberation + pre-edit Relevance Investigation + 11-factor Candidate Performance Prediction ranking (uncertainty-bounded) + Creative Director gate review + pre-publication Experiment Ledger registration + traceable storyboard generation
     - Original procedural `em_induction_faraday` animation rendered via `chromium_canvas` (`1920x1080`, RGBA ProRes4444 alpha `.mov`)
     - OpenTimelineIO `.otio` export + FFmpeg `1080p` 16:9 YouTube master render + SFX/ambient mixing + kinetic ASS/SRT caption burning
     - Three face-tracked `1080x1920` 9:16 short-form renders (`youtube_shorts`, `tiktok`, `instagram_reels`) with platform-specific safe zones
     - YouTube `1280x720` PNG thumbnail generation from clean source footage
     - 6-style grounded Threads post generation + Telegram deep-dive campaign with 5 tagged funnel links and community poll
     - Automated QA verification & PNG frame extraction (`black_frame_intervals = []`, `freeze_frame_intervals = []`)

3. **Messy 3-Minute Multi-Voice Benchmark + Live 100-Video Discovery** (`scripts/run_messy_benchmark.py`):
   - **Fixture**: `175.0s` spoken fixture combining 3 macOS voices/accents (`Samantha`, `Daniel`, `Aman`), explicit `1.8s–2.4s` dead-air gaps, `-34 dB` noise floor, quiet clause, and off-center subject (`recommended_crop_center_x = 0.3699`); generated in **`14.61s`** on initial build.
   - **Live Multi-Niche Discovery**: **`100`** verified-metric videos fetched via official YouTube creator Atom RSS feeds across **`9`** categories and **`13`** creators, scored against channel-median baselines (`audience_normalized_ratio`, e.g. `15.1x` / `98.6` breakout).
   - **Symposium Deliberation**: `18` participating agents, `20` traceable debate turns, `2` Moderator challenges, `1` dead-pause trim revision (`deterministic_analytical` backend when no local Ollama model is pulled).
   - **Multi-Format Deliverables**: **`69.80s`** for live discovery + campaign + `1080p` 16:9 (`66.792s`, `-17.8 dB` mean audio, `14` captions, `1` overlay, QA passed) + `3` distinct non-overlapping 9:16 Shorts per short-form platform (starts `[0.0, 19.5, 44.3]`), plus showcase Chromium renders of `wave_interference_sim` and `system_architecture_flow` and cumulative Growth snapshot verification (`4200` total after `1000 -> 2500 -> 4200` captures). Total end-to-end warm run: **`80.84s`**.

4. **Standalone Render & Test Micro-Benchmarks**:
   - **Standalone Chromium Canvas smoke render** (`em_induction_faraday`, 2.0s @ 15 fps = 30 RGBA PNG frames -> ProRes4444 `.mov`): **`3.466s`**.
   - **Unit, security, adapter, growth, & API contract suite** (`.venv/bin/pytest -v`, **15 tests**): **`3.05s`** (all 15 passed; includes 11-factor prediction + Creative Director gate, closed-loop per-platform `1.0x -> 1.5x -> 2.0x` experiment learning, and API evaluation routes; RSS/Telegram/Ollama external calls mocked in hermetic suite, live RSS tested separately in `run_messy_benchmark.py`).
   - **Frontend TypeScript + Vite production bundle** (`cd web && npm run build`): **`423ms`** (`189.87 kB` JS bundle + `4.43 kB` CSS).

## Verified YouTube Deliverable QA Telemetry

- Resolution / FPS / Duration: `1920x1080` @ `24.0 fps`, `87.875s`
- Audio Loudness: mean `-17.0 dB`, max `-2.6 dB` (`has_audio = true`)
- Black / Freeze Frame Intervals: none (`[]`)
- Rendered Captions / Overlays: `29` caption cues, `1` animated overlay (`chromium_canvas`)
- Vertical Adaptation Output (`youtube_shorts`): `1080x1920` (`H.264` + `AAC`, `16.000s`, subject crop centered at `x = 0.3707`)
