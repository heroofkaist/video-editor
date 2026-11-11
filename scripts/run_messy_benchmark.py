"""P0 Real-World Messy 3-Minute Benchmark + Live Multi-Niche Research + Full Multi-Platform Verification.

Generates a challenging 180s spoken fixture containing:
- Accent & speaking-rate variety (Samantha US 175wpm, Daniel UK 150wpm, Aman IN 145wpm)
- Explicit 2.0s-2.5s dead-air silence gaps
- Steady -34 dB room noise floor + one low-SNR quiet muffled clause (stress-testing ASR confidence & Moderator audit)
- Off-center presenter at x=0.37 (exercising face-tracked 9:16 vertical crop across 3 distinct Shorts cuts)

Then runs live YouTube creator Atom RSS discovery (~100 videos), full media analysis, COMPREHENSIVE AI Symposium,
Chromium animation renders (em_induction_faraday + wave_interference_sim + system_architecture_flow),
16:9 YouTube master + 3 distinct 9:16 Shorts per short-form platform, Threads/Telegram funnel, and growth cumulative audit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from symposium_studio.config import StudioConfig  # noqa: E402
from symposium_studio.models.symposium import ResearchMode  # noqa: E402
from symposium_studio.studio import SymposiumStudio  # noqa: E402


MESSY_CHAPTERS: list[tuple[str, str, str, int, float, float]] = [
    # (label, voice, text, wpm_rate, voice_gain, pause_after_sec)
    (
        "HOOK: Zero-Volt Paradox",
        "Samantha",
        "Hold on. Watch what happens when I park this neodymium magnet dead center inside three hundred turns of copper wire. Zero volts! Not even a flicker on the bench multimeter! Why does a massive magnetic field generate zero electricity until the instant my hand starts shaking the coil?",
        172,
        0.92,
        2.3,
    ),
    (
        "MISTAKE: Winding Failure",
        "Daniel",
        "Last month I burned an entire Saturday winding enamel magnet wire around an aluminum pipe instead of insulating fiberboard. Every time the field surged, eddy currents circulated inside the aluminum sleeve itself and opposed the magnetic flux change before the copper turns ever saw clean signal. It is a classic bench mistake nobody warns beginners about.",
        150,
        0.90,
        2.1,
    ),
    (
        "PHYSICS: Faraday Derivation",
        "Aman",
        "Let us strip away hand-waving and trace Faraday's law accurately. Induced electromotive force equals minus N times the time derivative of magnetic flux, where flux equals magnetic field B integrated across coil area A cosine theta. Notice what sits inside that derivative: amplitude N turns, magnetic geometry, and relative velocity omega!",
        146,
        0.90,
        2.4,
    ),
    (
        "QUIET NOTE: Fringe Capacitance",
        "Daniel",
        "As a quick caveat measured quietly near the shielded bench, stray inter-winding capacitance shifts the self-resonant notch above forty kilohertz when lead inductance climbs past two microhenries.",
        162,
        0.18,
        1.8,
    ),
    (
        "MECHANISM: Wave & Impedance",
        "Samantha",
        "Push that magnet faster and look at the oscilloscope trace: voltage peaks do not align with maximum magnetic field! Voltage hits zero at peak magnet displacement because velocity crosses zero at turnaround, giving a crisp ninety-degree phase shift! That phase shift explains regenerative braking, guitar pickups, and wireless resonant power transfer.",
        168,
        0.92,
        2.2,
    ),
    (
        "TAKEAWAY: 3 Bench Rules",
        "Aman",
        "So whenever an induction sensor reads suspiciously low voltage, inspect three things before rebuilding the circuit: first, measure mechanical velocity relative to coil turn density; second, eliminate conductive bobbins that waste energy in eddy loops; and third, match oscilloscope probe impedance to the coil source. Grab the complete schematics, raw CSV plots, and parameter notebooks via our Telegram engineering channel below.",
        146,
        0.90,
        1.4,
    ),
]


def _extract_png(video_path: Path, timestamp_sec: float, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, timestamp_sec):.2f}",
            "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2",
            str(out_png),
        ],
        check=True,
    )
    return out_png


def build_messy_benchmark_fixture(out_mp4: Path, target_min_duration: float = 175.0) -> Path:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    sr = 48000

    with tempfile.TemporaryDirectory(prefix="symposium_messy_") as td:
        tmp = Path(td)
        wav_parts: list[np.ndarray] = []
        chapter_spans: list[tuple[float, float, str]] = []
        cursor = 0.0

        for idx, (label, voice, text, rate, gain, pause_sec) in enumerate(MESSY_CHAPTERS):
            aiff = tmp / f"p_{idx}.aiff"
            wav = tmp / f"p_{idx}.wav"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(aiff), "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                check=True,
            )
            with wave.open(str(wav), "rb") as wf:
                pcm = (np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0) * gain

            # Inject -34 dB pink-ish noise over speech portion (and light -58 dB room floor in dead-air gaps so silencedetect works reliably!)
            noise_speech = rng.normal(0.0, 0.018, size=len(pcm)).astype(np.float32)
            pcm_noisy = np.clip(pcm + noise_speech, -0.95, 0.95)
            dur = len(pcm_noisy) / sr
            chapter_spans.append((cursor, cursor + dur + pause_sec, label))
            wav_parts.append(pcm_noisy)
            pause_samples = int(pause_sec * sr)
            wav_parts.append(rng.normal(0.0, 0.001, size=pause_samples).astype(np.float32))
            cursor += dur + pause_sec

        # Append extra technical recap if under target_min_duration (~3m)
        if cursor < target_min_duration:
            recap = (
                "To verify Faraday induction quantitatively on the lab bench, mount a pickup coil on a linear actuator, "
                "record peak open-circuit voltage against stroke frequency across five steps from five hertz to forty hertz, "
                "and plot the slope directly to recover turns-area calibration in seconds. That simple linear check validates every magnetic model."
            )
            aiff = tmp / "p_extra.aiff"
            wav = tmp / "p_extra.wav"
            subprocess.run(["say", "-v", "Daniel", "-r", "142", "-o", str(aiff), recap], check=True)
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(aiff), "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                check=True,
            )
            with wave.open(str(wav), "rb") as wf:
                pcm = (np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0) * 0.90
            noise_speech = rng.normal(0.0, 0.018, size=len(pcm)).astype(np.float32)
            pcm_noisy = np.clip(pcm + noise_speech, -0.95, 0.95)
            pad_sec = max(1.2, target_min_duration - cursor - len(pcm_noisy) / sr)
            chapter_spans.append((cursor, cursor + len(pcm_noisy) / sr + pad_sec, "RECAP: Calibration Slope"))
            wav_parts.append(pcm_noisy)
            wav_parts.append(rng.normal(0.0, 0.001, size=int(pad_sec * sr)).astype(np.float32))
            cursor += len(pcm_noisy) / sr + pad_sec

        full_audio = np.concatenate(wav_parts)
        master_wav = tmp / "master.wav"
        with wave.open(str(master_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((np.clip(full_audio, -0.95, 0.95) * 32767.0).astype(np.int16).tobytes())

        fps = 24
        w, h = 1280, 720
        total_frames = int(round(cursor * fps))
        raw_video = tmp / "video_only.mp4"
        writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:, :] = (24, 24, 30)
        cv2.rectangle(bg, (0, int(h * 0.78)), (w, h), (38, 34, 32), -1)
        cv2.rectangle(bg, (int(w * 0.61), int(h * 0.14)), (int(w * 0.95), int(h * 0.68)), (18, 28, 44), -1)
        cv2.rectangle(bg, (int(w * 0.61), int(h * 0.14)), (int(w * 0.95), int(h * 0.68)), (56, 189, 248), 3)

        cx_base = int(w * 0.37)
        cy_base = int(h * 0.41)
        samples_per_frame = sr // fps

        for f_idx in range(total_frames):
            t_sec = f_idx / fps
            i0 = f_idx * samples_per_frame
            i1 = min(len(full_audio), i0 + samples_per_frame)
            rms = float(np.sqrt(np.mean(full_audio[i0:i1] ** 2))) if i1 > i0 else 0.0

            frame = bg.copy()
            dx = int(round(7 * np.sin(t_sec * 1.1)))
            dy = int(round(4 * np.cos(t_sec * 1.6)))
            cx, cy = cx_base + dx, cy_base + dy

            cv2.ellipse(frame, (cx, cy + 200), (165, 130), 0, 180, 360, (105, 80, 52), -1)
            cv2.rectangle(frame, (cx - 28, cy + 60), (cx + 28, cy + 125), (155, 185, 222), -1)
            cv2.ellipse(frame, (cx, cy), (88, 112), 0, 0, 360, (165, 196, 232), -1)
            cv2.ellipse(frame, (cx, cy - 58), (88, 62), 0, 180, 360, (42, 34, 32), -1)
            for ex in (cx - 34, cx + 34):
                cv2.ellipse(frame, (ex, cy - 18), (15, 9), 0, 0, 360, (245, 245, 245), -1)
                cv2.circle(frame, (ex, cy - 18), 6, (35, 30, 30), -1)
                cv2.line(frame, (ex - 16, cy - 36), (ex + 16, cy - 38), (35, 30, 30), 4)
            cv2.line(frame, (cx, cy - 12), (cx - 4, cy + 20), (130, 155, 190), 3)
            mouth_open = int(min(18, max(2, rms * 130)))
            cv2.ellipse(frame, (cx, cy + 52), (24, mouth_open), 0, 0, 360, (65, 65, 135), -1)

            active_lbl = next((lbl for s, e, lbl in chapter_spans if s <= t_sec <= e), "Induction Benchmark")
            cv2.putText(frame, "LAB BENCHMARK", (int(w * 0.63), int(h * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
            cv2.putText(frame, active_lbl[:28], (int(w * 0.63), int(h * 0.31)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 250), 2)
            cv2.putText(frame, f"t = {t_sec:05.1f}s", (int(w * 0.63), int(h * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 180, 210), 2)

            writer.write(frame)
        writer.release()

        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(raw_video), "-i", str(master_wav),
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                str(out_mp4),
            ],
            check=True,
        )
    return out_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P0 messy 3-minute benchmark + live discovery verification.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of messy 3-minute benchmark video.")
    args = parser.parse_args()

    t_total = time.monotonic()
    cfg = StudioConfig.from_env(ROOT)
    fixture = cfg.workspaces_root / "fixtures" / "messy_induction_benchmark_3min.mp4"

    t0 = time.monotonic()
    if args.rebuild or not fixture.exists():
        print("[1/5] Generating messy multi-voice 3-minute fixture (noise + silent gaps + quiet clause + off-center face)...")
        build_messy_benchmark_fixture(fixture, target_min_duration=175.0)
    else:
        print(f"[1/5] Using existing messy fixture: {fixture}")
    t_fixture = round(time.monotonic() - t0, 2)

    studio = SymposiumStudio(cfg)
    print("[2/5] Running live research + COMPREHENSIVE Symposium + 16:9 + 3x3 distinct 9:16 Shorts pipeline...")
    t1 = time.monotonic()
    result = studio.run_campaign_pipeline(
        name="Messy Benchmark — Faraday, Impedance & Coil Losses",
        video_path=fixture,
        creator_niche="science_engineering",
        editing_style="educational_explainer",
        symposium_mode=ResearchMode.COMPREHENSIVE,
        telegram_channel="@symposium_physics_lab",
        discover_live_research=True,
        short_variants_per_platform=3,
    )
    t_pipeline = round(time.monotonic() - t1, 2)
    pid = result["project_id"]
    ws = Path(result["workspace_dir"])
    analysis = studio.load_analysis(pid)

    # Generate showcase Chromium renders of our new templates too
    print("[3/5] Rendering new Chromium showcase templates (wave_interference_sim + system_architecture_flow)...")
    anim_wave = studio.generate_animation(
        pid,
        template_id="wave_interference_sim",
        title="Standing Wave & Eddy Phase Interference",
        duration_sec=3.5,
        preferred_adapter="chromium_canvas",
    )
    anim_arch = studio.generate_animation(
        pid,
        template_id="system_architecture_flow",
        title="Signal Acquisition & Induction Pipeline",
        duration_sec=3.5,
        preferred_adapter="chromium_canvas",
        parameters={"stages": [{"name": "Coil EMF", "metric": "dPhi/dt"}, {"name": "Preamp", "metric": "Impedance"}, {"name": "FFT Trace", "metric": "90° Phase"}]},
    )

    # Verify cumulative growth audit behaves accurately
    studio.growth.record_snapshot(pid, "youtube", views_or_impressions=1000, likes_or_reactions=55, external_post_id="bench_yt")
    studio.growth.record_snapshot(pid, "youtube", views_or_impressions=2500, likes_or_reactions=140, external_post_id="bench_yt")
    studio.growth.record_snapshot(pid, "youtube", views_or_impressions=4200, likes_or_reactions=240, external_post_id="bench_yt")
    growth_sum = studio.growth.summarize(pid)
    yt_growth = next(p for p in growth_sum["platform_summaries"] if p["platform"] == "youtube")
    assert yt_growth["total_views_or_impressions"] == 4200, f"Double-counting bug! Got {yt_growth}"

    # Extract crisp verification frames for multimodal visual inspection
    print("[4/5] Extracting PNG verification frames from 16:9, 3 distinct Shorts, and new animations...")
    insp_dir = ws / "qa" / "benchmark_inspection"
    frames: dict[str, str] = {}
    yt_video = Path(result["renders"]["youtube"]["video_path"])
    yt_dur = float(result["renders"]["youtube"]["qa"]["duration_sec"])
    frames["youtube_mid"] = str(_extract_png(yt_video, yt_dur * 0.45, insp_dir / "yt_mid.png"))
    frames["thumbnail"] = str(result["renders"]["youtube"]["thumbnail_path"])

    shorts = result["renders"]["vertical_variants"]["youtube_shorts"]
    assert len(shorts) == 3, f"Expected 3 distinct YouTube Shorts, got {len(shorts)}"
    for i, sh in enumerate(shorts):
        assert sh["qa"]["passed"] is True, f"Short #{i+1} failed QA: {sh['qa']}"
        sh_dur = float(sh["qa"]["duration_sec"])
        frames[f"short_{i+1}"] = str(_extract_png(Path(sh["video_path"]), sh_dur * 0.4, insp_dir / f"short_{i+1}.png"))

    if anim_wave.output_path:
        frames["anim_wave"] = str(_extract_png(Path(anim_wave.output_path), 1.5, insp_dir / "anim_wave.png"))
    if anim_arch.output_path:
        frames["anim_arch"] = str(_extract_png(Path(anim_arch.output_path), 1.5, insp_dir / "anim_arch.png"))

    # Read symposium session to surface debate/challenge counts accurately
    sym_files = sorted((ws / "symposium").glob("sym_*.json"))
    sym_payload = json.loads(sym_files[-1].read_text(encoding="utf-8"))

    # Strict assertions on messy video + live discovery + distinct shorts
    disc = result["discovery"] or {}
    assert result["detected_language"] == "en", f"Expected 'en', got {result['detected_language']}"
    assert len(analysis.silences) >= 2, f"Expected >=2 dead-air silences detected, got {len(analysis.silences)}"
    assert result["renders"]["youtube"]["qa"]["passed"] is True
    assert disc.get("discovered_count", 0) >= 60, f"Expected >=60 live RSS items, got {disc}"
    assert disc.get("verified_metric_items_count", 0) >= 50

    tl_files = sorted((ws / "timelines").glob("*youtube_shorts*.json"))
    distinct_starts = set()
    for tf in tl_files:
        tj = json.loads(tf.read_text(encoding="utf-8"))
        if tj.get("clips"):
            distinct_starts.add(round(float(tj["clips"][0]["source_start_sec"]), 1))
    assert len(distinct_starts) == 3, f"Expected 3 distinct non-overlapping Short source starts, got {distinct_starts}"

    summary = {
        "project_id": pid,
        "workspace_dir": str(ws),
        "timings_sec": {
            "fixture_generation": t_fixture,
            "live_research_and_full_campaign": t_pipeline,
            "total": round(time.monotonic() - t_total, 2),
        },
        "source_video": {
            "duration_sec": analysis.probe.duration_sec,
            "silences_detected": len(analysis.silences),
            "transcript_segments": len(analysis.transcript.segments),
            "uncertain_segments": len(analysis.transcript.uncertain_segment_ids),
            "recommended_crop_center_x": analysis.recommended_crop_center_x,
        },
        "live_discovery": {
            "discovered_count": disc.get("discovered_count"),
            "verified_metric_items_count": disc.get("verified_metric_items_count"),
            "categories_count": len(disc.get("categories_covered", [])),
            "creators_sampled": disc.get("creators_sampled"),
            "top_outlier": (disc.get("top_breakout_outliers") or [None])[0],
        },
        "symposium": {
            "backend_used": sym_payload.get("backend_used"),
            "agents_count": len(sym_payload.get("participating_agents", [])),
            "debate_turns_count": len(sym_payload.get("debate_turns", [])),
            "challenges_count": len(sym_payload.get("challenges", [])),
            "revision_history": sym_payload.get("revision_history", []),
        },
        "youtube_16x9": result["renders"]["youtube"]["qa"],
        "distinct_shorts_count": len(shorts),
        "distinct_short_starts_sec": sorted(distinct_starts),
        "growth_cumulative_total_after_3_snapshots": yt_growth["total_views_or_impressions"],
        "inspection_frames": frames,
    }
    (insp_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[5/5] P0 MESSY 3-MIN BENCHMARK PASSED!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
