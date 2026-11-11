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
   - **Cold/warm full campaign execution**: **`39.26s`** (post-QA layout fix run) to **`40.78s`** (initial run).
   - Stages executed in sequence:
     - Media probe + `whisper.cpp` (`ggml-base.bin` Metal) ASR with word timestamps and language auto-detection (`en`)
     - Audio silence detection & spectral speaker turn diarization
     - PySceneDetect cut detection + OpenCV face/skin tracking (`recommended_crop_center_x = 0.3707`) + sharpness/motion sampling + Apple Vision OCR sampling
     - Semantic clustering & 8-factor highlight scoring (`18` highlights extracted)
     - Cross-niche Research Intelligence FTS5 query + `18`-agent AI Symposium comprehensive deliberation + traceable storyboard generation
     - Original procedural `em_induction_faraday` animation rendered via `chromium_canvas` (`1920x1080`, RGBA ProRes4444 alpha `.mov`)
     - OpenTimelineIO `.otio` export + FFmpeg `1080p` 16:9 YouTube master render + SFX mixing + kinetic ASS/SRT caption burning
     - Three face-tracked `1080x1920` 9:16 short-form renders (`youtube_shorts`, `tiktok`, `instagram_reels`) with platform-specific safe zones
     - YouTube `1280x720` PNG thumbnail generation from clean source footage
     - 6-style grounded Threads post generation + Telegram deep-dive campaign with 5 tagged funnel links and community poll
     - Automated QA verification & PNG frame extraction (`black_frame_intervals = []`, `freeze_frame_intervals = []`)

3. **Iterative Natural-Language Re-Render** (`"Make the captions smaller"`):
   - Re-scales `CaptionSettings.scale_factor` in timeline JSON & `.otio` and re-renders YouTube `1080p` + QA verification: **`11.51s`** (second run; `11.55s` initial run).

4. **Standalone Render & Test Micro-Benchmarks**:
   - **Standalone Chromium Canvas smoke render** (`em_induction_faraday`, 2.0s @ 15 fps = 30 RGBA PNG frames -> ProRes4444 `.mov`): **`3.466s`**.
   - **Unit & API contract test suite** (`.venv/bin/pytest -v`, 6 tests): **`1.04s`**.
   - **Frontend TypeScript + Vite production bundle** (`cd web && npm run build`): **`425ms`** (`173.26 kB` JS bundle + `4.43 kB` CSS).

## Verified YouTube Deliverable QA Telemetry

- Resolution / FPS / Duration: `1920x1080` @ `24.0 fps`, `73.667s`
- Audio Loudness: mean `-17.0 dB`, max `-3.7 dB` (`has_audio = true`)
- Black / Freeze Frame Intervals: none (`[]`)
- Rendered Captions / Overlays: `24` caption cues, `1` animated overlay (`chromium_canvas`)
- Vertical Adaptation Output (`youtube_shorts`): `1080x1920` (`H.264` + `AAC`, `16.000s`, subject crop centered at `x = 0.3707`)
