"""Tests for OpenTimelineIO interchange, ASS/SRT captions, effects library, SFX synthesis, and iterative editing."""

from __future__ import annotations

import wave
from pathlib import Path

import opentimelineio as otio

from symposium_studio.engines.effects import search_effects, synthesize_sfx
from symposium_studio.engines.production import (
    apply_iterative_edit,
    build_timeline_from_storyboard,
    export_otio,
    write_ass,
    write_srt,
)
from symposium_studio.models.symposium import Storyboard, StoryboardBeat
from symposium_studio.models.timeline import TargetPlatformFormat
from tests.test_research_and_symposium import _sample_analysis


def test_timeline_otio_captions_and_iterative_edit(tmp_path: Path) -> None:
    analysis = _sample_analysis("proj_tl")
    # Flag segment 2 as uncertain to confirm [?] marker appears cleanly without inventing text
    analysis.transcript.segments[2].low_confidence = True

    sb = Storyboard(
        storyboard_id="sb_test",
        project_id="proj_tl",
        selected_strategy="Test Strategy",
        target_duration_sec=65.0,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=0.0,
                source_end_sec=18.0,
                target_duration_sec=18.0,
                visual_treatment="clean_clarity",
            ),
            StoryboardBeat(
                beat_index=1,
                role="animation_explainer",
                source_start_sec=20.0,
                source_end_sec=65.0,
                target_duration_sec=45.0,
                visual_treatment="cutaway_with_pip",
                animation_template="em_induction_faraday",
            ),
            StoryboardBeat(
                beat_index=2,
                role="payoff",
                source_start_sec=70.0,
                source_end_sec=100.0,
                target_duration_sec=30.0,
                visual_treatment="clean_clarity",
            ),
        ],
    )

    tl_yt = build_timeline_from_storyboard(
        "proj_tl",
        "Induction Test",
        analysis,
        sb,
        platform_format=TargetPlatformFormat.YOUTUBE_LONG,
    )
    tl_tiktok = build_timeline_from_storyboard(
        "proj_tl",
        "Induction TikTok",
        analysis,
        sb,
        platform_format=TargetPlatformFormat.TIKTOK,
    )
    # Platform-specific safe zones and dimensions
    assert (tl_yt.width, tl_yt.height) == (1920, 1080)
    assert (tl_tiktok.width, tl_tiktok.height) == (1080, 1920)
    assert tl_tiktok.caption_settings.safe_margin_bottom_px > tl_yt.caption_settings.safe_margin_bottom_px

    otio_path = tmp_path / "test.otio"
    export_otio(tl_yt, otio_path)
    loaded_otio = otio.adapters.read_from_file(str(otio_path))
    assert loaded_otio.name == "Induction Test"
    assert len(loaded_otio.tracks[0]) == 3

    srt_path = write_srt(tl_yt, tmp_path / "caps.srt")
    ass_path = write_ass(tl_yt, tmp_path / "caps.ass")
    srt_txt = srt_path.read_text(encoding="utf-8")
    ass_txt = ass_path.read_text(encoding="utf-8")
    assert "[uncertain]" in srt_txt
    assert "[?]" in ass_txt

    # Iterative agent instructions
    tl2, changes = apply_iterative_edit(tl_yt, "Make the opening more engaging and make the captions smaller")
    assert tl2.revision == 2
    assert tl2.caption_settings.scale_factor < tl_yt.caption_settings.scale_factor
    assert tl2.clips[0].zoom_scale > tl_yt.clips[0].zoom_scale
    assert len(changes) >= 2


def test_effects_library_and_sfx_synth(tmp_path: Path) -> None:
    transitions = search_effects(category="transition")
    sound = search_effects(category="sound")
    assert len(transitions) >= 5
    assert len(sound) >= 4

    for preset in ("whoosh_transition", "sub_impact", "riser_tension", "ui_pop"):
        wav_p = synthesize_sfx(preset, tmp_path / f"{preset}.wav")
        assert wav_p.exists()
        with wave.open(str(wav_p), "rb") as wf:
            assert wf.getframerate() == 48000
            assert wf.getnframes() > 4000
