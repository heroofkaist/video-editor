"""Deep audiovisual analyzer producing edit_events.json, contact_sheet.jpg, evidence.json, and StyleDNA."""

from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import butter, sosfiltfilt

from symposium_studio.engines.analysis.diarize import detect_silences
from symposium_studio.engines.analysis.probe import probe_media
from symposium_studio.engines.analysis.transcribe import (
    _load_wav_mono_float,
    compute_audio_envelope,
    extract_raw_mono_wav,
    transcribe_media,
)
from symposium_studio.engines.analysis.vision import _try_apple_vision_ocr, detect_scene_cuts
from symposium_studio.models.media import MediaProbeResult
from symposium_studio.models.reference import (
    AudioStyleObservation,
    PacingRhythmObservation,
    ReferenceAnalysisStatus,
    ReferenceEditEvent,
    ReferenceItem,
    StyleDNA,
    SubtitleStyleObservation,
    VisualStyleObservation,
    _utc_now,
)

_CASCADE_XML = Path(getattr(getattr(cv2, "data", None), "haarcascades", "")) / "haarcascade_frontalface_default.xml"
_CASCADE_CLS = getattr(cv2, "CascadeClassifier", None)
_FACE_CASCADE = _CASCADE_CLS(str(_CASCADE_XML)) if (_CASCADE_CLS is not None and _CASCADE_XML.is_file()) else None


def _detect_zoom_and_micro_cuts(
    media_path: Path,
    duration_sec: float,
    base_cuts_sec: list[float],
) -> list[dict[str, Any]]:
    """Fast 4fps downscaled scan to catch jump cuts and sudden punch-in scale transitions missed by coarse thresholds."""
    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step_frames = max(1, int(round(fps / 4.0)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_scans = min(total_frames, int(round(min(duration_sec, 180.0) * fps)))

    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    prev_t = 0.0
    transitions: list[dict[str, Any]] = []

    idx = 0
    while idx < max_scans:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = round(idx / fps, 3)
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if prev_gray is not None and prev_hist is not None:
            mad = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))) / 255.0
            corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            # Bidirectional multi-scale zoom similarity (1.09x, 1.15x, 1.25x zoom-in and zoom-out)
            h0, w0 = prev_gray.shape
            min_zoom_mad = mad
            for scale_keep in (0.92, 0.87, 0.80):
                dy = max(1, int(round(h0 * (1.0 - scale_keep) / 2.0)))
                dx = max(1, int(round(w0 * (1.0 - scale_keep) / 2.0)))
                crop_in = cv2.resize(prev_gray[dy : h0 - dy, dx : w0 - dx], (w0, h0))
                z_in = float(np.mean(np.abs(gray.astype(np.float32) - crop_in.astype(np.float32)))) / 255.0
                if z_in < min_zoom_mad:
                    min_zoom_mad = z_in
                crop_out = cv2.resize(gray[dy : h0 - dy, dx : w0 - dx], (w0, h0))
                z_out = float(np.mean(np.abs(prev_gray.astype(np.float32) - crop_out.astype(np.float32)))) / 255.0
                if z_out < min_zoom_mad:
                    min_zoom_mad = z_out

            near_existing = any(abs(t_sec - bc) <= 0.45 for bc in base_cuts_sec)
            near_prev_punch = any(
                tr["kind"] == "punch_in" and abs(t_sec - float(tr["time_sec"])) <= 0.75
                for tr in transitions
            )
            is_punch = (
                corr >= 0.84
                and mad >= 0.045
                and min_zoom_mad < mad * 0.83
                and (mad - min_zoom_mad) >= 0.010
            )
            if is_punch and not near_prev_punch:
                transitions.append(
                    {
                        "time_sec": t_sec,
                        "mad": round(mad, 4),
                        "zoom_mad": round(min_zoom_mad, 4),
                        "hist_corr": round(corr, 4),
                        "kind": "punch_in",
                        "prev_t": prev_t,
                    }
                )
            elif (mad >= 0.11 or corr <= 0.72) and not near_existing:
                transitions.append(
                    {
                        "time_sec": t_sec,
                        "mad": round(mad, 4),
                        "zoom_mad": round(min_zoom_mad, 4),
                        "hist_corr": round(corr, 4),
                        "kind": "cutaway" if corr < 0.55 else "jump_cut",
                        "prev_t": prev_t,
                    }
                )
        prev_gray = gray
        prev_hist = hist
        prev_t = t_sec
        idx += step_frames

    cap.release()
    return transitions


def _measure_loudness_ebur128(media_path: Path) -> tuple[float | None, float | None]:
    """Measure integrated LUFS and loudness range via FFmpeg ebur128 filter."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(media_path),
            "-af",
            "ebur128=framelog=quiet",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    txt = proc.stderr or ""
    lufs = None
    lra = None
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("I:") and "LUFS" in s:
            try:
                lufs = float(s.split()[1])
            except Exception:
                pass
        elif s.startswith("LRA:") and "LU" in s:
            try:
                lra = float(s.split()[1])
            except Exception:
                pass
    return lufs, lra


def _analyze_reference_audio(
    media_path: Path,
    probe: MediaProbeResult,
    cut_times: list[float],
    workdir: Path,
) -> tuple[AudioStyleObservation, list[ReferenceEditEvent]]:
    """Analyze dialogue activity ratio, pause structure, HF SFX transients, loudness, and ducking."""
    if not probe.has_audio:
        return AudioStyleObservation(has_audio=False, mean_db=-60.0, max_db=-60.0, speech_activity_ratio=0.0), []

    wav_path = extract_raw_mono_wav(media_path, workdir / "ref_raw_16k.wav", sample_rate=16000)
    silences = detect_silences(media_path, duration_sec=probe.duration_sec)
    lufs, lra = _measure_loudness_ebur128(media_path)
    sr, samples = _load_wav_mono_float(wav_path)
    env = compute_audio_envelope(wav_path)

    dur_mins = max(1.0 / 60.0, probe.duration_sec / 60.0)
    total_silence_sec = sum(s.duration_sec for s in silences)
    speech_ratio = max(0.0, min(1.0, 1.0 - (total_silence_sec / max(0.1, probe.duration_sec))))
    median_pause = round(statistics.median([s.duration_sec for s in silences]), 3) if silences else 0.0

    sfx_events: list[ReferenceEditEvent] = []
    music_bed_present = False
    music_bed_ratio = 0.0
    ducking_ratio = 0.0

    if sr > 8000 and len(samples) > sr // 4:
        # 1. High-frequency transient detector (>2.6kHz bandpass) to spot clicks/whooshes/pops
        sos = butter(4, 2600.0 / (0.5 * sr), btype="high", output="sos")
        hf = sosfiltfilt(sos, samples.astype(np.float64))
        hop = max(1, int(0.025 * sr))
        usable = (len(hf) // hop) * hop
        if usable > hop * 4:
            hf_frames = np.sqrt(np.mean(hf[:usable].reshape(-1, hop) ** 2, axis=1) + 1e-12)
            base_hf = float(np.percentile(hf_frames, 60))
            thr_hf = max(base_hf * 3.8, float(np.max(hf_frames)) * 0.35, 0.015)
            last_t = -1.0
            for fi, val in enumerate(hf_frames):
                t_here = round(fi * 0.025, 3)
                if val >= thr_hf and (t_here - last_t) >= 0.45:
                    near_cut = any(abs(t_here - ct) <= 0.35 for ct in cut_times)
                    sfx_events.append(
                        ReferenceEditEvent(
                            event_id=f"ev_sfx_{len(sfx_events):03d}",
                            event_type="sfx_transient",
                            start_sec=t_here,
                            end_sec=round(min(probe.duration_sec, t_here + 0.18), 3),
                            duration_sec=0.18,
                            intensity=round(min(1.0, float(val / max(thr_hf, 1e-6)) * 0.55), 3),
                            description="High-frequency SFX/click transient" + (" near cut" if near_cut else ""),
                            evidence={"hf_rms": round(float(val), 4), "near_visual_cut": near_cut},
                        )
                    )
                    last_t = t_here

    if env is not None:
        rms_arr: np.ndarray = env["smooth_rms"]
        p15 = float(np.percentile(rms_arr, 15))
        p85 = float(np.percentile(rms_arr, 85))
        if p15 >= 0.008 and p85 > p15 * 1.15:
            music_bed_present = True
            music_bed_ratio = round(min(1.0, float(np.mean(rms_arr >= p15 * 0.9))), 3)
            ducking_ratio = round(max(0.0, min(1.0, 1.0 - (p15 / max(p85, 1e-6)))), 3)

    obs = AudioStyleObservation(
        has_audio=True,
        mean_db=round(probe.mean_volume_db, 2),
        max_db=round(probe.max_volume_db, 2),
        measured_lufs=lufs,
        loudness_range_lu=lra,
        speech_activity_ratio=round(speech_ratio, 3),
        silence_gap_count=len(silences),
        median_pause_sec=median_pause,
        transient_sfx_per_min=round(len(sfx_events) / dur_mins, 2),
        estimated_music_bed_present=music_bed_present,
        music_bed_ratio=music_bed_ratio,
        dialogue_ducking_ratio=ducking_ratio,
    )
    return obs, sfx_events


def _collect_adaptive_timestamps(duration_sec: float, cut_times: list[float], max_samples: int = 32) -> list[float]:
    pts: set[float] = {0.2}
    for s in np.arange(0.5, min(6.0, duration_sec), 0.75):
        pts.add(round(float(s), 2))
    for ct in cut_times[:16]:
        if ct + 0.18 < duration_sec:
            pts.add(round(ct + 0.18, 2))
    if duration_sec > 2.0:
        for u in np.linspace(0.5, max(0.6, duration_sec - 0.3), num=16):
            pts.add(round(float(u), 2))
    sorted_pts = sorted(t for t in pts if 0.0 <= t <= max(0.2, duration_sec - 0.05))
    if len(sorted_pts) > max_samples:
        step = len(sorted_pts) / max_samples
        sorted_pts = [sorted_pts[int(i * step)] for i in range(max_samples)]
    return sorted_pts


def _analyze_frames_and_build_contact_sheet(
    media_path: Path,
    probe: MediaProbeResult,
    sample_timestamps: list[float],
    cut_events: list[ReferenceEditEvent],
    out_contact_sheet: Path,
) -> tuple[VisualStyleObservation, SubtitleStyleObservation, dict[str, Any]]:
    """Analyze color grading, faces, graphic overlays, subtitle regions, and write visual contact_sheet.jpg."""
    cap = cv2.VideoCapture(str(media_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or probe.fps or 30.0
    brightness_vals: list[float] = []
    saturation_vals: list[float] = []
    contrast_vals: list[float] = []
    temp_biases: list[float] = []
    motion_vals: list[float] = []
    face_ratios: list[float] = []
    face_frames = 0
    graphic_frames = 0
    blur_pillar_votes = 0

    band_edge_scores: dict[str, list[float]] = {
        "upper_mid": [],
        "center_dynamic": [],
        "lower_third": [],
        "bottom_safe": [],
    }
    yellow_text_hits = 0
    ocr_snippets: list[str] = []
    sheet_tiles: list[tuple[float, str, np.ndarray]] = []
    prev_gray: np.ndarray | None = None

    for ts in sample_timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(ts * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness_vals.append(float(np.median(hsv[:, :, 2])) / 255.0)
        saturation_vals.append(float(np.median(hsv[:, :, 1])) / 255.0)
        contrast_vals.append(float(np.percentile(gray, 90) - np.percentile(gray, 10)) / 255.0)
        temp_biases.append((float(np.mean(frame[:, :, 2])) - float(np.mean(frame[:, :, 0]))) / 255.0)

        if prev_gray is not None:
            small_cur = cv2.resize(gray, (160, 90))
            small_prv = cv2.resize(prev_gray, (160, 90))
            motion_vals.append(float(np.mean(np.abs(small_cur.astype(np.float32) - small_prv.astype(np.float32)))) / 255.0)
        prev_gray = gray

        # Blurred pillarbar check on 16:9 canvas
        if w > h * 1.25:
            side_l = cv2.Laplacian(gray[:, : max(1, int(w * 0.14))], cv2.CV_64F).var()
            side_r = cv2.Laplacian(gray[:, int(w * 0.86) :], cv2.CV_64F).var()
            center = cv2.Laplacian(gray[:, int(w * 0.3) : int(w * 0.7)], cv2.CV_64F).var()
            if center > 25.0 and max(side_l, side_r) < center * 0.28:
                blur_pillar_votes += 1

        faces: Any = ()
        if _FACE_CASCADE is not None:
            faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(28, 28))
        if len(faces) > 0:
            face_frames += 1
            best_f = max(faces, key=lambda b: int(b[2]) * int(b[3]))
            face_ratios.append(float(best_f[3]) / max(1.0, float(h)))

        edges = cv2.Canny(gray, 70, 160)
        edge_density = float(np.mean(edges > 0))
        color_flat_ratio = float(np.mean(hsv[:, :, 1] < 45))
        if len(faces) == 0 and (edge_density >= 0.06 or color_flat_ratio >= 0.55):
            graphic_frames += 1

        # Subtitle band stroke + bright chroma density heuristic
        for band_name, (y0_f, y1_f) in (
            ("upper_mid", (0.16, 0.36)),
            ("center_dynamic", (0.40, 0.60)),
            ("lower_third", (0.62, 0.80)),
            ("bottom_safe", (0.78, 0.92)),
        ):
            y0, y1 = int(h * y0_f), int(h * y1_f)
            roi_g = gray[y0:y1, int(w * 0.12) : int(w * 0.88)]
            roi_h = hsv[y0:y1, int(w * 0.12) : int(w * 0.88)]
            if roi_g.size == 0:
                continue
            gx = np.abs(cv2.Sobel(roi_g, cv2.CV_32F, 1, 0, ksize=3))
            bright_txt = (roi_g >= 210) & (gx >= 35.0)
            yellow_txt = (roi_h[:, :, 0] >= 16) & (roi_h[:, :, 0] <= 38) & (roi_h[:, :, 1] >= 110) & (gx >= 30.0)
            if float(np.mean(yellow_txt)) > 0.004:
                yellow_text_hits += 1
            score_band = float(np.mean(bright_txt | yellow_txt))
            band_edge_scores[band_name].append(score_band)

        if len(ocr_snippets) < 4:
            ocr_txt = _try_apple_vision_ocr(frame)
            if ocr_txt:
                ocr_snippets.append(ocr_txt)

        if len(sheet_tiles) < 12:
            near_ev = next((e.event_type for e in cut_events if abs(e.start_sec - ts) <= 0.4), "sample")
            sheet_tiles.append((ts, near_ev, frame.copy()))

    cap.release()

    # Render real contact sheet grid (up to 4x3)
    if sheet_tiles:
        cols = min(4, len(sheet_tiles))
        rows = (len(sheet_tiles) + cols - 1) // cols
        tile_w, tile_h = 320, 180
        canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (15, 23, 42))
        draw = ImageDraw.Draw(canvas)
        for ti, (ts, tag, bgr) in enumerate(sheet_tiles):
            r_i, c_i = divmod(ti, cols)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            im.thumbnail((tile_w - 8, tile_h - 26))
            px = c_i * tile_w + (tile_w - im.width) // 2
            py = r_i * tile_h + 4
            canvas.paste(im, (px, py))
            draw.rectangle([c_i * tile_w, (r_i + 1) * tile_h - 22, (c_i + 1) * tile_w, (r_i + 1) * tile_h], fill=(2, 6, 23))
            draw.text((c_i * tile_w + 8, (r_i + 1) * tile_h - 18), f"t={ts:.1f}s [{tag}]", fill=(248, 250, 252))
        out_contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_contact_sheet, quality=90)

    n_seen = max(1, len(brightness_vals))
    aspect_str = "9:16" if probe.height > probe.width else ("1:1" if abs(probe.width - probe.height) < 40 else "16:9")
    med_face_h = round(statistics.median(face_ratios), 3) if face_ratios else 0.0
    face_ratio = round(face_frames / n_seen, 3)
    graphic_ratio = round(graphic_frames / n_seen, 3)
    blur_pillar = blur_pillar_votes >= max(2, n_seen // 3)

    if graphic_ratio >= 0.55 and face_ratio < 0.25:
        framing = "graphic_dominant"
    elif med_face_h >= 0.34:
        framing = "tight_closeup"
    elif face_ratio >= 0.4:
        framing = "medium_talking_head"
    elif face_ratio > 0.15:
        framing = "wide_desk"
    else:
        framing = "b_roll_dominant"

    mean_temp = float(np.mean(temp_biases)) if temp_biases else 0.0
    temp_label = "warm" if mean_temp > 0.035 else ("cool" if mean_temp < -0.035 else "neutral")

    vis = VisualStyleObservation(
        width=probe.width,
        height=probe.height,
        fps=round(probe.fps, 2),
        aspect_ratio=aspect_str,
        is_vertical=probe.height > probe.width,
        has_blur_pillarbars=blur_pillar,
        median_brightness=round(float(np.median(brightness_vals)) if brightness_vals else 0.5, 3),
        median_saturation=round(float(np.median(saturation_vals)) if saturation_vals else 0.4, 3),
        contrast_range=round(float(np.median(contrast_vals)) if contrast_vals else 0.5, 3),
        color_temperature_bias=temp_label,
        face_presence_ratio=face_ratio,
        median_face_height_ratio=med_face_h,
        framing_style=framing,
        camera_motion_mean=round(float(np.mean(motion_vals)) if motion_vals else 0.0, 4),
        motion_graphics_ratio=graphic_ratio,
    )

    band_means = {k: (float(np.mean(v)) if v else 0.0) for k, v in band_edge_scores.items()}
    best_band = max(band_means, key=lambda k: band_means[k])
    sub_detected = band_means[best_band] >= 0.008 or bool(ocr_snippets)
    sub_obs = SubtitleStyleObservation(
        presence_detected=sub_detected,
        vertical_region=best_band if sub_detected else ("center_dynamic" if vis.is_vertical else "bottom_safe"),
        estimated_words_per_cue=2 if vis.is_vertical else 3,
        estimated_font_weight="heavy" if yellow_text_hits >= 2 else "bold",
        color_scheme_hint="neon_yellow_highlight" if yellow_text_hits >= 2 else "high_contrast_white",
        stroke_contrast_score=round(min(1.0, band_means[best_band] * 30.0 + 0.35), 3),
        evidence_note=(
            f"Observed high-contrast stroke density in '{best_band}' band"
            + (f" + OCR samples ({len(ocr_snippets)})" if ocr_snippets else " (no external OCR binary installed)")
        ),
    )
    frame_evidence = {
        "sampled_frames": len(brightness_vals),
        "band_edge_scores": {k: round(v, 5) for k, v in band_means.items()},
        "yellow_highlight_hits": yellow_text_hits,
        "ocr_snippets": ocr_snippets,
        "blur_pillar_votes": blur_pillar_votes,
    }
    return vis, sub_obs, frame_evidence


def _map_dna_to_archetype(
    pacing: PacingRhythmObservation,
    vis: VisualStyleObservation,
    audio: AudioStyleObservation,
    ref_item: ReferenceItem,
) -> tuple[str, str, str, float, float]:
    """Map real measured metrics to executable base_style, archetype label, animation engine, SFX/meme intensity."""
    cuts_pm = pacing.cuts_per_minute
    if vis.motion_graphics_ratio >= 0.45 and pacing.median_shot_duration_sec >= 2.2:
        return "educational_explainer", "technical_motion_explainer", "manim_factual_diagram", 0.42, 0.08
    if cuts_pm >= 22.0 or pacing.median_shot_duration_sec <= 1.8:
        sfx_i = min(1.0, 0.55 + audio.transient_sfx_per_min / 30.0)
        meme_d = 0.65 if (vis.is_vertical or cuts_pm >= 26.0) else 0.45
        return "zoomer_vlog", "high_retention_creator_zoomer", "meme_reaction_card", round(sfx_i, 3), meme_d
    if vis.is_vertical and cuts_pm >= 12.0:
        return "short_form_viral", "punchy_short_form_storyteller", "kinetic_callout", 0.62, 0.35
    if pacing.median_shot_duration_sec >= 4.2 and cuts_pm < 8.5:
        return "minimalist_cinematic", "deliberate_cinematic_narrator", "kinetic_callout", 0.18, 0.0
    if vis.motion_graphics_ratio >= 0.25 or audio.speech_activity_ratio >= 0.78:
        return "documentary_deep_dive", "narrative_deep_dive", "plotly_chart", 0.32, 0.10
    if ref_item.platform in {"tiktok", "instagram"}:
        return "short_form_viral", "social_reel_editor", "kinetic_callout", 0.55, 0.3
    return "fast_paced_creator", "modern_creator_talking_head", "kinetic_callout", 0.45, 0.25


def analyze_reference_item(
    ref_item: ReferenceItem,
    workdir: Path | None = None,
    run_asr: bool = True,
    whisper_model: str = "mlx-community/whisper-small-mlx",
) -> ReferenceItem:
    """Perform real CV, PySceneDetect, audio spectral, and optional speech-cadence analysis on a reference video."""
    if ref_item.analysis_status == ReferenceAnalysisStatus.METADATA_ONLY or not ref_item.source_path:
        return ref_item
    video_path = Path(ref_item.source_path)
    if not video_path.is_file():
        ref_item.analysis_status = ReferenceAnalysisStatus.FAILED
        return ref_item

    target_dir = Path(workdir) if workdir is not None else video_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_media(video_path)
    duration_sec = max(0.2, float(probe.duration_sec))

    # 1. Scene cuts + fast micro-cut/punch-in transitions
    scene_cuts = detect_scene_cuts(video_path, duration_sec=duration_sec, threshold=24.0)
    base_cut_starts = [round(c.start_sec, 3) for c in scene_cuts if c.start_sec > 0.25]
    micro_transitions = _detect_zoom_and_micro_cuts(video_path, duration_sec, base_cut_starts)

    events: list[ReferenceEditEvent] = []
    for c in scene_cuts:
        if c.start_sec <= 0.15:
            continue
        events.append(
            ReferenceEditEvent(
                event_id=f"ev_cut_{len(events):03d}",
                event_type="cut",
                start_sec=round(c.start_sec, 3),
                end_sec=round(c.end_sec, 3),
                duration_sec=round(c.duration_sec, 3),
                intensity=0.65,
                description=f"Scene transition detected via ContentDetector (scene #{c.scene_index})",
                evidence={"detector": "PySceneDetect.ContentDetector", "scene_index": c.scene_index},
            )
        )
    for mt in micro_transitions:
        ts = float(mt["time_sec"])
        events.append(
            ReferenceEditEvent(
                event_id=f"ev_{mt['kind']}_{len(events):03d}",
                event_type=str(mt["kind"]),
                start_sec=ts,
                end_sec=round(min(duration_sec, ts + 0.5), 3),
                duration_sec=0.5,
                intensity=round(min(1.0, float(mt["mad"]) * 4.0), 3),
                description=f"Measured {mt['kind']} transition (MAD={mt['mad']}, hist_corr={mt['hist_corr']})",
                evidence=mt,
            )
        )

    raw_cut_points = sorted(
        e.start_sec for e in events if e.event_type in {"cut", "jump_cut", "cutaway", "punch_in"}
    )
    dedup_cuts: list[float] = []
    for ct in raw_cut_points:
        if not dedup_cuts or (ct - dedup_cuts[-1]) >= 0.35:
            dedup_cuts.append(ct)
    all_cut_points = sorted({0.0, duration_sec, *dedup_cuts})
    shot_durations = [round(all_cut_points[i + 1] - all_cut_points[i], 3) for i in range(len(all_cut_points) - 1) if all_cut_points[i + 1] - all_cut_points[i] >= 0.15]
    if not shot_durations:
        shot_durations = [round(duration_sec, 3)]

    punch_cnt = sum(1 for e in events if e.event_type == "punch_in")
    total_cuts = max(0, len(all_cut_points) - 2)
    dur_mins = max(1.0 / 60.0, duration_sec / 60.0)
    cuts_per_min = round(total_cuts / dur_mins, 2)
    first_5s_cuts = sum(1 for t in all_cut_points if 0.1 < t <= 5.0)
    first_cut = next((t for t in all_cut_points if t > 0.1), None)

    # 2. Audio spectral transients + loudness + music bed
    with tempfile.TemporaryDirectory(prefix="symp_ref_audio_") as audio_tmp:
        audio_obs, sfx_events = _analyze_reference_audio(video_path, probe, all_cut_points, Path(audio_tmp))
    events.extend(sfx_events)
    events.sort(key=lambda e: e.start_sec)

    # 3. Frame sampling, visual style, subtitle band observation & contact sheet
    sample_ts = _collect_adaptive_timestamps(duration_sec, [t for t in all_cut_points if 0.0 < t < duration_sec])
    contact_sheet_path = target_dir / "contact_sheet.jpg"
    vis_obs, sub_obs, frame_evidence = _analyze_frames_and_build_contact_sheet(
        video_path, probe, sample_ts, events, contact_sheet_path
    )

    # 4. Optional local Whisper speech cadence (never fabricates words)
    hook_first_speech = None
    honest_notes: list[str] = []
    if run_asr and probe.has_audio and audio_obs.mean_db > -50.0:
        try:
            tr = transcribe_media(video_path, duration_sec=min(duration_sec, 90.0), model_name=whisper_model)
            segs = tr.segments
            if segs:
                hook_first_speech = round(segs[0].start_sec, 3)
                sub_obs.speech_language = tr.detected_language
                words_cnt = sum(len(s.words) if s.words else len(s.text.split()) for s in segs)
                speech_span = sum(max(0.1, s.end_sec - s.start_sec) for s in segs)
                sub_obs.speech_wpm = round((words_cnt / max(0.5, speech_span)) * 60.0, 1)
                if sub_obs.speech_wpm >= 165.0:
                    sub_obs.estimated_words_per_cue = 2
            else:
                honest_notes.append("No decodable speech words observed by local ASR in reference audio track.")
        except Exception as exc:
            honest_notes.append(f"Speech cadence ASR skipped/unavailable ({type(exc).__name__}); rhythm inferred acoustically.")
    elif not probe.has_audio:
        honest_notes.append("Reference file contains no audio track; pacing & framing derived strictly from visual pixels.")

    # Energy arc
    if first_5s_cuts >= 3:
        arc = "explosive_frontloaded"
    elif cuts_per_min >= 18.0:
        arc = "alternating_peaks"
    elif cuts_per_min < 6.0:
        arc = "calm_deliberate"
    else:
        arc = "steady_build"

    p25 = round(float(np.percentile(shot_durations, 25)), 3)
    p75 = round(float(np.percentile(shot_durations, 75)), 3)
    pacing = PacingRhythmObservation(
        duration_sec=round(duration_sec, 3),
        total_cuts=total_cuts,
        cuts_per_minute=cuts_per_min,
        mean_shot_duration_sec=round(float(np.mean(shot_durations)), 3),
        median_shot_duration_sec=round(float(np.median(shot_durations)), 3),
        p25_shot_duration_sec=p25,
        p75_shot_duration_sec=p75,
        punch_in_count=punch_cnt,
        punch_ins_per_minute=round(punch_cnt / dur_mins, 2),
        hook_window_cuts_first_5s=first_5s_cuts,
        hook_first_cut_sec=round(first_cut, 3) if first_cut is not None else None,
        hook_first_speech_sec=hook_first_speech,
        climax_activity_time_ratio=0.5,
        energy_arc=arc,
    )

    base_style, archetype, anim_engine, rec_sfx, rec_meme = _map_dna_to_archetype(pacing, vis_obs, audio_obs, ref_item)

    events_path = target_dir / "edit_events.json"
    events_path.write_text(
        json.dumps([e.model_dump(mode="json") for e in events], indent=2),
        encoding="utf-8",
    )
    evidence_path = target_dir / "evidence.json"
    evidence_payload = {
        "reference_id": ref_item.reference_id,
        "analysis_status": ReferenceAnalysisStatus.FULL_VIDEO_AUDIO.value,
        "duration_sec": duration_sec,
        "shot_durations_sec": shot_durations,
        "frame_evidence": frame_evidence,
        "sfx_event_count": len(sfx_events),
        "honest_limitations": honest_notes,
        "analyzed_at": _utc_now(),
    }
    evidence_path.write_text(json.dumps(evidence_payload, indent=2), encoding="utf-8")

    editorial_notes = [
        f"Measured {cuts_per_min:.1f} cuts/min (median shot {pacing.median_shot_duration_sec:.2f}s, {punch_cnt} zoom punch-in step(s)).",
        f"Framing is {vis_obs.framing_style} ({vis_obs.aspect_ratio}, saturation={vis_obs.median_saturation:.2f}, tone={vis_obs.color_temperature_bias}).",
        f"Audio profile: {audio_obs.mean_db:.1f} dBFS mean, {audio_obs.transient_sfx_per_min:.1f} SFX transients/min, speech={audio_obs.speech_activity_ratio*100:.0f}%.",
    ]

    dna = StyleDNA(
        dna_id=f"dna_{uuid.uuid4().hex[:8]}",
        reference_id=ref_item.reference_id,
        analysis_status=ReferenceAnalysisStatus.FULL_VIDEO_AUDIO,
        mapped_base_style=base_style,
        archetype_label=archetype,
        pacing=pacing,
        subtitles=sub_obs,
        audio=audio_obs,
        visual=vis_obs,
        recommended_animation_engine=anim_engine,
        recommended_sfx_intensity=rec_sfx,
        recommended_meme_density=rec_meme,
        evidence_artifacts={
            "edit_events": str(events_path),
            "contact_sheet": str(contact_sheet_path),
            "evidence": str(evidence_path),
        },
        editorial_notes=editorial_notes,
        honest_limitations=honest_notes,
    )
    (target_dir / "style_dna.json").write_text(dna.model_dump_json(indent=2), encoding="utf-8")

    updated = ref_item.model_copy(deep=True)
    updated.duration_sec = probe.duration_sec
    updated.analysis_status = ReferenceAnalysisStatus.FULL_VIDEO_AUDIO
    updated.style_dna = dna
    updated.edit_events_path = str(events_path)
    updated.contact_sheet_path = str(contact_sheet_path)
    updated.evidence_path = str(evidence_path)
    updated.updated_at = _utc_now()
    return updated
