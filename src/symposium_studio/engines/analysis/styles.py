"""Selectable editing style profiles and reference-video style extractor."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, detect

from symposium_studio.models.media import EditingStyleName, EditingStyleProfile

STYLE_PROFILES: dict[EditingStyleName, EditingStyleProfile] = {
    EditingStyleName.HIGH_ENERGY_YOUTUBE: EditingStyleProfile(
        name=EditingStyleName.HIGH_ENERGY_YOUTUBE,
        display_name="High-Energy YouTube",
        target_avg_shot_sec=4.5,
        max_silence_gap_sec=0.35,
        transition_style="whip_pan",
        caption_preset="kinetic_caption_pop",
        color_grade_preset="punchy_contrast",
        zoom_punch_intensity=0.85,
        broll_animation_ratio=0.35,
        audio_duck_db=-14.0,
        description="Fast hook, punch-in zooms, crisp silence cuts, energetic SFX accents.",
    ),
    EditingStyleName.CINEMATIC_STORYTELLING: EditingStyleProfile(
        name=EditingStyleName.CINEMATIC_STORYTELLING,
        display_name="Cinematic Storytelling",
        target_avg_shot_sec=8.5,
        max_silence_gap_sec=0.75,
        transition_style="cross_dissolve",
        caption_preset="slide_reveal",
        color_grade_preset="cinematic_teal_amber",
        zoom_punch_intensity=0.25,
        broll_animation_ratio=0.30,
        audio_duck_db=-16.0,
        description="Measured pacing, warm filmic grade, deliberate transitions and breathing room.",
    ),
    EditingStyleName.EDUCATIONAL_EXPLAINER: EditingStyleProfile(
        name=EditingStyleName.EDUCATIONAL_EXPLAINER,
        display_name="Educational Explainer",
        target_avg_shot_sec=6.5,
        max_silence_gap_sec=0.45,
        transition_style="zoom_push",
        caption_preset="word_highlight_karaoke",
        color_grade_preset="clean_clarity",
        zoom_punch_intensity=0.45,
        broll_animation_ratio=0.40,
        audio_duck_db=-14.0,
        description="Clarity-first pacing with factual animated diagrams and high-legibility captions.",
    ),
    EditingStyleName.DOCUMENTARY: EditingStyleProfile(
        name=EditingStyleName.DOCUMENTARY,
        display_name="Documentary",
        target_avg_shot_sec=9.0,
        max_silence_gap_sec=0.8,
        transition_style="dip_to_black",
        caption_preset="lower_third_editorial",
        color_grade_preset="docu_neutral",
        zoom_punch_intensity=0.15,
        broll_animation_ratio=0.35,
        audio_duck_db=-16.0,
        description="Evidence-led storytelling, chapter lower-thirds, controlled dynamics.",
    ),
    EditingStyleName.TECHNICAL_TUTORIAL: EditingStyleProfile(
        name=EditingStyleName.TECHNICAL_TUTORIAL,
        display_name="Technical Tutorial",
        target_avg_shot_sec=7.5,
        max_silence_gap_sec=0.4,
        transition_style="match_motion_cut",
        caption_preset="lower_third_editorial",
        color_grade_preset="crisp_studio",
        zoom_punch_intensity=0.3,
        broll_animation_ratio=0.45,
        audio_duck_db=-14.0,
        description="Step-by-step code/circuit/physics overlays with clean UI callouts.",
    ),
    EditingStyleName.PODCAST_HIGHLIGHTS: EditingStyleProfile(
        name=EditingStyleName.PODCAST_HIGHLIGHTS,
        display_name="Podcast Highlights",
        target_avg_shot_sec=6.0,
        max_silence_gap_sec=0.45,
        transition_style="cut",
        caption_preset="word_highlight_karaoke",
        color_grade_preset="warm_studio",
        zoom_punch_intensity=0.35,
        broll_animation_ratio=0.15,
        audio_duck_db=-12.0,
        description="Speaker-centered framing, tight pause removal, bold quote callouts.",
    ),
    EditingStyleName.COMEDY: EditingStyleProfile(
        name=EditingStyleName.COMEDY,
        display_name="Comedy",
        target_avg_shot_sec=3.5,
        max_silence_gap_sec=0.28,
        transition_style="whip_pan",
        caption_preset="kinetic_caption_pop",
        color_grade_preset="bright_sat",
        zoom_punch_intensity=0.9,
        broll_animation_ratio=0.25,
        audio_duck_db=-12.0,
        description="Tight comic timing, punch-in reaction frames, snappy pop SFX.",
    ),
    EditingStyleName.PERSONAL_VLOG: EditingStyleProfile(
        name=EditingStyleName.PERSONAL_VLOG,
        display_name="Personal Vlog",
        target_avg_shot_sec=5.5,
        max_silence_gap_sec=0.5,
        transition_style="cross_dissolve",
        caption_preset="slide_reveal",
        color_grade_preset="natural_warm",
        zoom_punch_intensity=0.3,
        broll_animation_ratio=0.20,
        audio_duck_db=-15.0,
        description="Intimate conversational cadence, lightweight location/timeline cards.",
    ),
    EditingStyleName.ZOOMER_VLOG: EditingStyleProfile(
        name=EditingStyleName.ZOOMER_VLOG,
        display_name="Zoomer / Gen-Z Vlog",
        target_avg_shot_sec=2.1,
        max_silence_gap_sec=0.22,
        transition_style="cut",
        caption_preset="kinetic_caption_pop",
        color_grade_preset="punchy_contrast",
        zoom_punch_intensity=1.0,
        broll_animation_ratio=0.55,
        audio_duck_db=-12.0,
        description="High-retention ADHD silence-stripped jump cuts, punch-in zoom snaps, word-burst subtitles, trending reaction memes, and layered internet click/pop/boom SFX.",
    ),
    EditingStyleName.SHORT_FORM_VIRAL: EditingStyleProfile(
        name=EditingStyleName.SHORT_FORM_VIRAL,
        display_name="Short-Form Viral",
        target_avg_shot_sec=3.2,
        max_silence_gap_sec=0.22,
        transition_style="zoom_push",
        caption_preset="kinetic_caption_pop",
        color_grade_preset="punchy_contrast",
        zoom_punch_intensity=0.95,
        broll_animation_ratio=0.38,
        audio_duck_db=-12.0,
        description="Immediate 2-second visual hook, 9:16 safe-zone kinetic captions, high tempo.",
    ),
    EditingStyleName.MINIMAL_PROFESSIONAL: EditingStyleProfile(
        name=EditingStyleName.MINIMAL_PROFESSIONAL,
        display_name="Minimal Professional",
        target_avg_shot_sec=8.0,
        max_silence_gap_sec=0.5,
        transition_style="cut",
        caption_preset="lower_third_editorial",
        color_grade_preset="clean_clarity",
        zoom_punch_intensity=0.1,
        broll_animation_ratio=0.25,
        audio_duck_db=-16.0,
        description="Clean studio cuts, subtle typography, distraction-free delivery.",
    ),
}


def get_style_profile(name: str | EditingStyleName) -> EditingStyleProfile:
    if isinstance(name, EditingStyleName):
        return STYLE_PROFILES[name]
    try:
        return STYLE_PROFILES[EditingStyleName(name)]
    except ValueError:
        return STYLE_PROFILES[EditingStyleName.EDUCATIONAL_EXPLAINER]


def analyze_reference_style(reference_video: Path) -> EditingStyleProfile:
    """Derive pacing and visual profile metrics from a user-supplied reference video."""
    if not reference_video.exists():
        raise FileNotFoundError(f"Reference video not found: {reference_video}")

    scenes = detect(str(reference_video), ContentDetector(threshold=27.0))
    shot_lengths = [max(0.5, e.get_seconds() - s.get_seconds()) for s, e in scenes]
    avg_shot = float(np.mean(shot_lengths)) if shot_lengths else 6.0

    cap = cv2.VideoCapture(str(reference_video))
    luma_vals: list[float] = []
    sat_vals: list[float] = []
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1), 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_vals.append(float(np.mean(hsv[:, :, 1])))
        luma_vals.append(float(np.mean(hsv[:, :, 2])))
    cap.release()

    mean_sat = float(np.mean(sat_vals)) if sat_vals else 90.0
    if avg_shot < 3.8:
        base = STYLE_PROFILES[EditingStyleName.SHORT_FORM_VIRAL].model_copy()
    elif avg_shot < 5.5:
        base = STYLE_PROFILES[EditingStyleName.HIGH_ENERGY_YOUTUBE].model_copy()
    elif avg_shot < 7.8:
        base = STYLE_PROFILES[EditingStyleName.EDUCATIONAL_EXPLAINER].model_copy()
    else:
        base = STYLE_PROFILES[EditingStyleName.CINEMATIC_STORYTELLING].model_copy()

    base.target_avg_shot_sec = round(avg_shot, 2)
    base.color_grade_preset = "punchy_contrast" if mean_sat > 110 else "clean_clarity"
    base.description = f"Derived from reference {reference_video.name} (avg shot {avg_shot:.1f}s, mean sat {mean_sat:.0f})."
    return base
