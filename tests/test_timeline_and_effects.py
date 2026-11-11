"""Tests for OpenTimelineIO interchange, ASS/SRT captions, effects library, SFX synthesis, and iterative editing."""

from __future__ import annotations

import ast
import subprocess
import wave
from pathlib import Path

import opentimelineio as otio

from symposium_studio.config import StudioConfig
from symposium_studio.engines.animation.adapters.chromium_canvas import ChromiumCanvasAdapter, _build_canvas_html
from symposium_studio.engines.animation.adapters.external_adapters import (
    BlenderAdapter,
    ManimAdapter,
    MotionCanvasAdapter,
)
from symposium_studio.engines.animation.registry import AnimationEngineRegistry
from symposium_studio.engines.animation.scene_specs import build_scene_spec
from symposium_studio.engines.effects import search_effects, synthesize_ambient_bed, synthesize_sfx
from symposium_studio.engines.production import (
    apply_iterative_edit,
    build_timeline_from_storyboard,
    build_vertical_short_timelines,
    export_otio,
    render_timeline,
    write_ass,
    write_srt,
)
from symposium_studio.models.symposium import Storyboard, StoryboardBeat
from symposium_studio.models.timeline import AudioNode, TargetPlatformFormat, TransitionSpec
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

    amb_p = synthesize_ambient_bed(tmp_path / "ambient.wav", duration_sec=4.0, mood="tech_focus")
    assert amb_p.exists()
    with wave.open(str(amb_p), "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnframes() >= 48000 * 3


def test_animation_injection_escaping_and_parameterized_adapters(tmp_path: Path) -> None:
    xss_title = '</script><script>alert("xss")</script>'
    py_inj_title = 'Pwn"; import os; raise RuntimeError("boom") #'

    spec_xss = build_scene_spec("kinetic_callout", title=xss_title, duration_sec=2.0)
    html = _build_canvas_html(spec_xss)
    assert "</script><script>alert" not in html
    assert "\\u003c/script\\u003e" in html

    spec_mc = build_scene_spec(
        "system_architecture_flow",
        title=xss_title,
        duration_sec=3.0,
        parameters={"stages": [{"name": "Ingest", "metric": "FFmpeg"}, {"name": "Publish", "metric": "Threads"}]},
    )
    tsx_path = MotionCanvasAdapter().emit_typescript_scene(spec_mc, tmp_path / "mc")
    tsx_txt = tsx_path.read_text(encoding="utf-8")
    assert "const SCENE_SPEC =" in tsx_txt
    assert "Ingest" in tsx_txt

    spec_py = build_scene_spec("wave_interference_sim", title=py_inj_title, duration_sec=3.0)
    manim_py = ManimAdapter().emit_manim_script(spec_py, tmp_path / "manim")
    manim_src = manim_py.read_text(encoding="utf-8")
    ast.parse(manim_src)
    assert "SPEC = json.loads(" in manim_src

    blender_py = BlenderAdapter().emit_blender_script(spec_py, tmp_path / "blender")
    blender_src = blender_py.read_text(encoding="utf-8")
    ast.parse(blender_src)
    assert "SPEC = json.loads(" in blender_src

    # Real Chromium render check on new templates (only when headless Chromium runtime is available)
    adapter = ChromiumCanvasAdapter()
    if adapter.capability().available:
        cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db")
        engine = AnimationEngineRegistry(cfg)
        for tmpl in ("system_architecture_flow", "wave_interference_sim"):
            s = build_scene_spec(
                tmpl,
                title=f"Smoke {tmpl}",
                duration_sec=1.0,
                width=640,
                height=360,
                fps=15,
                preferred_adapter="chromium_canvas",
            )
            res = engine.render_scene(s, tmp_path / f"anim_{tmpl}")
            assert res.output_path and Path(res.output_path).exists()
            assert Path(res.output_path).stat().st_size > 5000


def test_distinct_shorts_presets_and_sidechain_speed_render(tmp_path: Path) -> None:
    analysis = _sample_analysis("proj_shorts")
    analysis.probe.duration_sec = 72.0

    # Verify fallback slicing guarantees 3 non-overlapping 9:16 variants even with 2 initial highlights
    shorts = build_vertical_short_timelines(
        "proj_shorts",
        "Faraday Induction",
        analysis,
        platform_format=TargetPlatformFormat.YOUTUBE_SHORTS,
        max_variants=3,
    )
    assert len(shorts) == 3
    starts = {round(s.clips[0].source_start_sec, 1) for s in shorts if s.clips}
    assert len(starts) == 3
    assert all(s.width == 1080 and s.height == 1920 for s in shorts)

    # Test expanded iterative edit actions
    tl = shorts[0]
    tl.audio_events.append(
        AudioNode(
            audio_id="bgm_test",
            source_path=str(tmp_path / "ambient.wav"),
            timeline_start_sec=0.0,
            duration_sec=10.0,
            role="music",
            gain_db=-16.0,
            duck_under_speech=True,
        )
    )
    tl_spd, c1 = apply_iterative_edit(tl, "Speed up 1.1x")
    assert tl_spd.clips[0].playback_speed > 1.0
    assert any("playback speeds" in x.lower() for x in c1)

    tl_mute, c2 = apply_iterative_edit(tl_spd, "Mute music")
    assert not any(a.role == "music" for a in tl_mute.audio_events)
    assert any("music" in x.lower() for x in c2)

    # Real FFmpeg render test with speed + non-drifting fade + sidechaincompress ambient bed
    src_mp4 = tmp_path / "src_4s.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=4.5:size=640x360:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=4.5:sample_rate=48000",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(src_mp4),
        ],
        check=True,
    )
    amb_wav = synthesize_ambient_bed(tmp_path / "ambient_duck.wav", duration_sec=3.0)
    analysis_fast = _sample_analysis("proj_fast")
    analysis_fast.probe.path = str(src_mp4)
    analysis_fast.probe.duration_sec = 4.5
    analysis_fast.probe.width = 640
    analysis_fast.probe.height = 360
    sb = Storyboard(
        storyboard_id="sb_fast",
        project_id="proj_fast",
        selected_strategy="High Energy Ducking Test",
        target_duration_sec=4.2,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=0.0,
                source_end_sec=2.0,
                target_duration_sec=2.0,
                visual_treatment="punch_zoom_text",
            ),
            StoryboardBeat(
                beat_index=1,
                role="explanation",
                source_start_sec=2.0,
                source_end_sec=4.2,
                target_duration_sec=2.2,
                visual_treatment="clean_clarity",
            ),
        ],
    )
    tl_render = build_timeline_from_storyboard(
        "proj_fast",
        "Ducking Render Test",
        analysis_fast,
        sb,
        style_name="high_energy_youtube",
    )
    tl_render.width = 640
    tl_render.height = 360
    tl_render.clips[0].source_path = str(src_mp4)
    tl_render.clips[1].source_path = str(src_mp4)
    tl_render.clips[0].playback_speed = 1.15
    tl_render.clips[1].transition_in = TransitionSpec(type="cross_dissolve", duration_sec=0.22)
    tl_render.audio_events.append(
        AudioNode(
            audio_id="bgm_duck",
            source_path=str(amb_wav),
            timeline_start_sec=0.0,
            duration_sec=4.0,
            role="music",
            gain_db=-14.0,
            duck_under_speech=True,
        )
    )
    out_mp4 = tmp_path / "rendered_fast_duck.mp4"
    res = render_timeline(tl_render, out_mp4, tmp_path)
    assert Path(res["video_path"]).exists()
    assert Path(res["video_path"]).stat().st_size > 10000
