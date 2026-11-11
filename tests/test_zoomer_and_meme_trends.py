"""Hermetic tests for live meme research, Gen-Z Zoomer jump cuts, meme pop-ups, vine_boom SFX, and upload API."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from symposium_studio.api.app import create_app
from symposium_studio.config import StudioConfig
from symposium_studio.engines.animation.adapters.chromium_canvas import ChromiumCanvasAdapter
from symposium_studio.engines.animation.scene_specs import build_scene_spec
from symposium_studio.engines.effects import (
    download_click_sounds,
    search_internet_clicks,
    synthesize_sfx,
)
from symposium_studio.engines.production.captions import write_ass
from symposium_studio.engines.production.renderer import ensure_project_sfx
from symposium_studio.engines.production.timeline_builder import (
    add_click_sounds_to_timeline,
    build_timeline_from_storyboard,
    build_vertical_short_timelines,
)
from symposium_studio.engines.research.connectors import MemeTrendConnector, download_trending_meme_assets
from symposium_studio.models.media import (
    EditingStyleName,
    HighlightProposal,
    HighlightScoreBreakdown,
    MediaAnalysisBundle,
    MediaProbeResult,
    SilenceInterval,
    TranscriptResult,
    TranscriptSegment,
    WordTimestamp,
)
from symposium_studio.models.research import ContentCategory
from symposium_studio.models.symposium import Storyboard, StoryboardBeat
from symposium_studio.models.timeline import TargetPlatformFormat


def _fake_png_bytes(color: tuple[int, int, int] = (240, 180, 40)) -> bytes:
    im = Image.new("RGB", (320, 240), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    # Pad PNG comment metadata minimally or ensure >512 bytes for download validation
    raw = buf.getvalue()
    if len(raw) < 600:
        im_big = Image.new("RGB", (640, 480), color)
        for x in range(0, 640, 8):
            im_big.putpixel((x, (x * 7) % 480), (20, 40, 200))
        buf2 = io.BytesIO()
        im_big.save(buf2, format="PNG")
        raw = buf2.getvalue()
    return raw


def test_meme_trend_connector_and_local_asset_cache(monkeypatch: Any, tmp_path: Path) -> None:
    png_data = _fake_png_bytes()

    class _FakeResp:
        def __init__(self, status_code: int, payload: Any = None, content: bytes = b"") -> None:
            self.status_code = status_code
            self._payload = payload
            self.content = content

        def json(self) -> Any:
            return self._payload

    def _mock_get(url: str, **kwargs: Any) -> _FakeResp:
        if "api.imgflip.com/get_memes" in url:
            return _FakeResp(
                200,
                payload={
                    "success": True,
                    "data": {
                        "memes": [
                            {
                                "id": "181913649",
                                "name": "Drake Hotline Bling",
                                "url": "https://i.imgflip.com/30b1gx.jpg",
                                "width": 1200,
                                "height": 1200,
                                "captions": 1500000,
                            },
                            {
                                "id": "87743020",
                                "name": "Two Buttons",
                                "url": "https://i.imgflip.com/1g8my4.jpg",
                                "width": 600,
                                "height": 908,
                                "captions": 1000000,
                            },
                        ]
                    },
                },
            )
        if "reddit.com" in url:
            return _FakeResp(429, payload={})
        if "i.imgflip.com" in url:
            return _FakeResp(200, content=png_data)
        return _FakeResp(404)

    monkeypatch.setattr("symposium_studio.engines.research.connectors.httpx.get", _mock_get)

    conn = MemeTrendConnector()
    items = conn.fetch_public_items(query="expectation vs quiz reality", limit=4)
    assert len(items) == 2
    assert items[0].category == ContentCategory.INTERNET_TRENDS
    assert items[0].metrics.metric_source == "imgflip_live_api"
    assert items[0].metrics.views and items[0].metrics.views >= 1_000_000

    cached = download_trending_meme_assets(items, tmp_path / "memes", max_count=2)
    assert len(cached) == 2
    for entry in cached:
        p = Path(entry["image_path"])
        assert p.is_file() and p.stat().st_size > 512


def test_zoomer_vlog_jump_cuts_memes_vine_boom_and_chromium_card(tmp_path: Path) -> None:
    meme_img = tmp_path / "meme_sample.png"
    meme_img.write_bytes(_fake_png_bytes((220, 80, 60)))

    # Render two real meme reaction overlays via ChromiumCanvasAdapter
    adapter = ChromiumCanvasAdapter()
    if adapter.capability().available:
        anim1 = adapter.render(
            build_scene_spec(
                "meme_reaction_card",
                title="Hook Meme",
                duration_sec=1.5,
                width=640,
                height=480,
                fps=12,
                transparent_background=True,
                parameters={
                    "top_label": "ОЖИДАНИЕ vs РЕАЛЬНОСТЬ 👀",
                    "meme_name": "Drake Hotline Bling",
                    "reaction_text": "Капсула на 2 года: қайырлы күн!",
                    "image_path": str(meme_img),
                },
            ),
            tmp_path / "anims",
        )
        anim2 = adapter.render(
            build_scene_spec(
                "meme_reaction_card",
                title="Payoff Meme",
                duration_sec=1.5,
                width=640,
                height=480,
                fps=12,
                transparent_background=True,
                parameters={
                    "top_label": "STUDENT PAIN 💀",
                    "meme_name": "Two Buttons",
                    "reaction_text": "Data Structures Quiz ертең 💀",
                    "image_path": str(meme_img),
                },
            ),
            tmp_path / "anims",
        )
        assert Path(anim1.output_path).exists()
        assert Path(anim2.output_path).exists()
    else:
        from symposium_studio.models.animation import AnimationRenderResult, FactualityMode

        anim1 = AnimationRenderResult(
            scene_id="anim_m1",
            adapter_used="chromium_canvas",
            output_path=str(tmp_path / "m1.mov"),
            has_alpha=True,
            width=640,
            height=480,
            fps=12.0,
            duration_sec=1.5,
            scene_spec_path=str(tmp_path / "m1.spec.json"),
            render_time_sec=0.1,
            factuality_mode=FactualityMode.ARTISTIC_ILLUSTRATION,
        )
        anim2 = anim1.model_copy(update={"scene_id": "anim_m2", "output_path": str(tmp_path / "m2.mov")})

    sfx_map = ensure_project_sfx(tmp_path / "effects")
    assert "vine_boom" in sfx_map and Path(sfx_map["vine_boom"]).is_file()

    # Build timeline with a 1.2s silent gap inside one long beat to verify silence jump-cutting
    analysis = MediaAnalysisBundle(
        project_id="proj_z",
        probe=MediaProbeResult(
            path=str(tmp_path / "src.mp4"),
            sha256="abc12345",
            size_bytes=1024000,
            duration_sec=12.0,
            width=1620,
            height=1080,
            fps=30.0,
            video_codec="h264",
            has_audio=True,
        ),
        transcript=TranscriptResult(
            engine="mlx-whisper",
            model="whisper-large-v3-mlx",
            detected_language="ru",
            language_probability=0.94,
            languages_observed=["ru", "kk"],
            duration_sec=12.0,
            segments=[
                TranscriptSegment(
                    segment_id="seg_0",
                    start_sec=0.0,
                    end_sec=4.2,
                    text="Привет ожидание реальность",
                    confidence=0.92,
                    words=[
                        WordTimestamp(word="Привет", start_sec=0.1, end_sec=0.9, probability=0.95),
                        WordTimestamp(word="ожидание", start_sec=1.0, end_sec=2.1, probability=0.94),
                        WordTimestamp(word="реальность", start_sec=2.2, end_sec=3.9, probability=0.92),
                    ],
                ),
                TranscriptSegment(
                    segment_id="seg_1",
                    start_sec=5.5,
                    end_sec=11.0,
                    text="Жақсы оқу Data Structures",
                    confidence=0.9,
                    words=[
                        WordTimestamp(word="Жақсы", start_sec=5.6, end_sec=6.6, probability=0.9),
                        WordTimestamp(word="оқу", start_sec=6.8, end_sec=7.8, probability=0.9),
                        WordTimestamp(word="Data", start_sec=8.0, end_sec=9.2, probability=0.9),
                        WordTimestamp(word="Structures", start_sec=9.4, end_sec=10.8, probability=0.9),
                    ],
                ),
            ],
        ),
        silences=[SilenceInterval(start_sec=4.2, end_sec=5.5, duration_sec=1.3)],
        highlights=[
            HighlightProposal(
                highlight_id="hl_0",
                start_sec=0.0,
                end_sec=11.5,
                duration_sec=11.5,
                role="hook",
                title="Ожидание vs Реальность",
                transcript_excerpt="Привет ожидание реальность",
                scores=HighlightScoreBreakdown(
                    information_density=0.8,
                    novelty=0.8,
                    story_relevance=0.9,
                    audience_curiosity=0.9,
                    educational_value=0.6,
                    emotional_intensity=0.8,
                    visual_opportunity=0.75,
                    context_dependency=0.9,
                    composite_score=0.82,
                ),
            )
        ],
    )
    sb = Storyboard(
        storyboard_id="sb_z",
        project_id="proj_z",
        selected_strategy="Zoomer Meme Edit",
        target_duration_sec=10.0,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=0.0,
                source_end_sec=11.5,
                target_duration_sec=10.0,
                visual_treatment="jump_cut_meme",
                animation_template="meme_reaction_card",
            )
        ],
    )
    tl = build_timeline_from_storyboard(
        project_id="proj_z",
        title="Zoomer Vlog",
        analysis=analysis,
        storyboard=sb,
        style_name=EditingStyleName.ZOOMER_VLOG.value,
        animation_result=anim1,
        platform_format=TargetPlatformFormat.YOUTUBE_LONG,
        sfx_paths=sfx_map,
        extra_overlays=[anim2],
    )
    assert len(tl.clips) >= 2, "Expected silence gap stripping to split continuous speech sub-clips"
    assert any(c.zoom_scale > 1.05 for c in tl.clips)
    assert len(tl.overlays) >= 2
    assert any("vine_boom" in a.source_path for a in tl.audio_events)
    assert tl.caption_settings.font_name == "Verdana"

    ass_file = write_ass(tl, tmp_path / "zoomer.ass")
    ass_txt = ass_file.read_text(encoding="utf-8")
    assert "Verdana" in ass_txt
    assert "Жақсы" in ass_txt

    shorts = build_vertical_short_timelines(
        project_id="proj_z",
        base_title="Zoomer Vlog",
        analysis=analysis,
        animation_result=anim1,
        platform_format=TargetPlatformFormat.YOUTUBE_SHORTS,
        max_variants=2,
        sfx_paths=sfx_map,
        style_name="zoomer_vlog",
        extra_overlays=[anim2],
    )
    assert len(shorts) >= 1
    assert shorts[0].overlays, "Expected Zoomer vertical Short to carry live meme card overlay"


def test_api_upload_and_zoomer_project(tmp_path: Path) -> None:
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "up.db")
    app = create_app(cfg)
    client = TestClient(app)

    create_resp = client.post(
        "/api/projects",
        json={"name": "Zoomer Capsule", "creator_niche": "personal_storytelling", "editing_style": "zoomer_vlog"},
    )
    assert create_resp.status_code == 200
    pid = create_resp.json()["project"]["project_id"]

    sample_mp4 = tmp_path / "upload_src.mp4"
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
            "testsrc=duration=1.2:size=320x240:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:duration=1.2:sample_rate=44100",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(sample_mp4),
        ],
        check=True,
    )
    with sample_mp4.open("rb") as fh:
        up = client.post(f"/api/projects/{pid}/upload", files={"file": ("capsule.mp4", fh, "video/mp4")})
    assert up.status_code == 200
    data = up.json()
    assert data["project"]["source_video_path"]
    assert Path(data["project"]["source_video_path"]).is_file()
    assert data["source_asset"]["role"] == "source_media"


def test_internet_click_downloader_search_cache_and_fallback(tmp_path: Path, monkeypatch: Any) -> None:
    import symposium_studio.engines.effects.sound_downloader as sd_mod

    # Create genuine short WAV payload returned by mocked internet download
    raw_src = synthesize_sfx("mouse_click", tmp_path / "seed_click.wav", duration_sec=0.14)
    raw_wav_bytes = raw_src.read_bytes()

    wiki_json = json.dumps(
        {
            "query": {
                "pages": {
                    "101": {
                        "title": "File:Computer Search Click.wav",
                        "imageinfo": [
                            {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/a/ac/Computer_Search_Click.wav?utm=1",
                                "mime": "audio/wav",
                                "size": len(raw_wav_bytes),
                                "mediatype": "AUDIO",
                                "extmetadata": {"LicenseShortName": {"value": "CC0"}},
                            }
                        ],
                    }
                }
            }
        }
    ).encode("utf-8")

    fetch_log: list[str] = []

    def fake_get(url: str, timeout_sec: float = 15.0, retries: int = 3) -> bytes:
        fetch_log.append(url)
        if "w/api.php" in url:
            return wiki_json
        if "Mouse_Click" in url or "Search_Click" in url:
            return raw_wav_bytes
        raise RuntimeError("simulated transient 429/network failure")

    monkeypatch.setattr(sd_mod, "_http_get_bytes", fake_get)

    hits = search_internet_clicks("mouse click", limit=5)
    assert len(hits) == 1
    assert hits[0]["license"] == "CC0"
    assert hits[0]["url"].endswith("Computer_Search_Click.wav")

    cache_dir = tmp_path / "cache"
    eff_dir1 = tmp_path / "proj1_effects"
    dl1 = download_click_sounds(
        eff_dir1,
        presets=["mouse_click", "mechanical_click"],
        search_query="mouse click",
        allow_network=True,
        cache_dir=cache_dir,
    )
    assert dl1["downloaded_count"] >= 2
    assert dl1["fallback_count"] == 1  # mechanical_click gracefully falls back to DSP click
    assert Path(dl1["click_paths"]["mouse_click"]).is_file()
    assert Path(dl1["click_paths"]["mechanical_click"]).is_file()
    assert Path(dl1["manifest_path"]).is_file()

    # Second run against another effects_dir hits local internet cache without re-downloading mouse_click
    eff_dir2 = tmp_path / "proj2_effects"
    before_calls = len(fetch_log)
    dl2 = download_click_sounds(
        eff_dir2,
        presets=["mouse_click"],
        allow_network=True,
        cache_dir=cache_dir,
    )
    assert dl2["cached_count"] == 1
    assert dl2["sources"]["mouse_click"]["origin"] == "cached_internet"
    assert len(fetch_log) == before_calls


def test_add_click_sounds_to_timeline_and_render_route(tmp_path: Path, monkeypatch: Any) -> None:
    import symposium_studio.engines.effects.sound_downloader as sd_mod

    seed_wav = synthesize_sfx("switch_click", tmp_path / "seed_switch.wav", duration_sec=0.12).read_bytes()
    monkeypatch.setattr(sd_mod, "_http_get_bytes", lambda url, timeout_sec=15.0, retries=3: seed_wav)

    effects_dir = tmp_path / "effects"
    dl = download_click_sounds(
        effects_dir,
        presets=["mouse_click", "mechanical_click", "switch_click"],
        allow_network=True,
        cache_dir=tmp_path / "sfx_cache",
    )
    assert len(dl["click_paths"]) == 3

    from symposium_studio.models.animation import AnimationRenderResult
    from symposium_studio.models.symposium import Storyboard, StoryboardBeat

    src_mp4 = tmp_path / "clip_src.mp4"
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
            "testsrc=duration=10.0:size=480x270:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=10.0:sample_rate=48000",
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
    ov_mov = tmp_path / "ov.mov"
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
            "color=c=blue@0.8:s=320x240:d=3.0:r=24",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4444",
            "-pix_fmt",
            "yuva444p10le",
            str(ov_mov),
        ],
        check=True,
    )

    sfx_map = ensure_project_sfx(effects_dir)
    analysis = MediaAnalysisBundle(
        project_id="proj_clk",
        probe=MediaProbeResult(
            path=str(src_mp4),
            sha256="d" * 64,
            size_bytes=src_mp4.stat().st_size,
            duration_sec=10.0,
            width=480,
            height=270,
            fps=24.0,
            video_codec="h264",
            has_audio=True,
        ),
        transcript=TranscriptResult(
            engine="whisper.cpp",
            model="test",
            detected_language="en",
            language_probability=0.95,
            duration_sec=10.0,
            segments=[
                TranscriptSegment(
                    segment_id="s0",
                    start_sec=0.0,
                    end_sec=4.5,
                    text="Wait 100% real!",
                    confidence=0.92,
                    words=[
                        WordTimestamp(word="Wait", start_sec=0.1, end_sec=0.5, probability=0.92),
                        WordTimestamp(word="100%", start_sec=1.2, end_sec=1.8, probability=0.92),
                        WordTimestamp(word="real!", start_sec=2.1, end_sec=2.8, probability=0.92),
                    ],
                ),
                TranscriptSegment(
                    segment_id="s1",
                    start_sec=4.6,
                    end_sec=6.9,
                    text="Check 2 buttons?",
                    confidence=0.92,
                    words=[
                        WordTimestamp(word="Check", start_sec=4.8, end_sec=5.4, probability=0.92),
                        WordTimestamp(word="2", start_sec=5.6, end_sec=6.1, probability=0.92),
                        WordTimestamp(word="buttons?", start_sec=6.2, end_sec=6.8, probability=0.92),
                    ],
                ),
                TranscriptSegment(
                    segment_id="s2",
                    start_sec=7.1,
                    end_sec=9.8,
                    text="Now 3 clicks!",
                    confidence=0.92,
                    words=[
                        WordTimestamp(word="Now", start_sec=7.2, end_sec=7.8, probability=0.92),
                        WordTimestamp(word="3", start_sec=8.0, end_sec=8.6, probability=0.92),
                        WordTimestamp(word="clicks!", start_sec=8.8, end_sec=9.6, probability=0.92),
                    ],
                ),
            ],
        ),
        silences=[
            SilenceInterval(start_sec=4.3, end_sec=4.8, duration_sec=0.5),
            SilenceInterval(start_sec=6.8, end_sec=7.2, duration_sec=0.4),
        ],
    )
    storyboard = Storyboard(
        storyboard_id="sb_clk",
        project_id="proj_clk",
        selected_strategy="Zoomer Click Edit",
        target_duration_sec=9.2,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=0.0,
                source_end_sec=4.5,
                target_duration_sec=4.2,
                animation_template="meme_reaction_card",
                visual_treatment="jump_cut_and_zoom",
            ),
            StoryboardBeat(
                beat_index=1,
                role="escalation",
                source_start_sec=4.5,
                source_end_sec=7.0,
                target_duration_sec=2.5,
                visual_treatment="punchy_contrast",
            ),
            StoryboardBeat(
                beat_index=2,
                role="payoff",
                source_start_sec=7.0,
                source_end_sec=9.8,
                target_duration_sec=2.5,
                visual_treatment="jump_cut_and_zoom",
            ),
        ],
    )
    anim = AnimationRenderResult(
        animation_id="anim_clk",
        scene_id="sc_clk",
        template_id="meme_reaction_card",
        adapter_used="chromium_canvas",
        scene_spec_path=str(tmp_path / "missing.json"),
        output_path=str(ov_mov),
        duration_sec=3.0,
        has_alpha=True,
        width=320,
        height=240,
        fps=24,
        render_time_sec=0.1,
        factuality_mode="artistic_illustration",
    )
    tl = build_timeline_from_storyboard(
        project_id="proj_clk",
        title="Click Test",
        analysis=analysis,
        storyboard=storyboard,
        style_name=EditingStyleName.ZOOMER_VLOG.value,
        animation_result=anim,
        sfx_paths=sfx_map,
    )
    # Click SFX auto-attached onto appropriate cuts/overlays/emphasis spots cleanly avoiding vine_boom conflicts
    clicks = [a for a in tl.audio_events if str(a.audio_id).startswith("clk_")]
    booms = [a for a in tl.audio_events if "vine_boom" in a.source_path]
    assert len(clicks) >= 2
    assert booms, "Expected vine_boom hook SFX to remain intact"
    for ck in clicks:
        for bm in booms:
            assert abs(ck.timeline_start_sec - bm.timeline_start_sec) >= 0.48

    # Idempotent re-application with tighter spacing does not duplicate click events
    tl2, placements2 = add_click_sounds_to_timeline(tl, dl["click_paths"], min_spacing_sec=1.5, replace_existing=True)
    assert len([a for a in tl2.audio_events if str(a.audio_id).startswith("clk_")]) == len(placements2)

    # API route test: POST /api/projects/{pid}/sfx/clicks
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws_api", db_path=tmp_path / "ws_api" / "db.sqlite")
    app = create_app(cfg)
    client = TestClient(app)
    resp = client.post("/api/projects", json={"name": "Click API", "editing_style": "zoomer_vlog"})
    pid = resp.json()["project"]["project_id"]
    ws_dir = Path(resp.json()["project"]["workspace_dir"])
    (ws_dir / "timelines").mkdir(parents=True, exist_ok=True)
    (ws_dir / "timelines" / f"{tl.timeline_id}_youtube.json").write_text(tl.model_dump_json(indent=2), encoding="utf-8")

    sfx_resp = client.post(
        f"/api/projects/{pid}/sfx/clicks",
        json={"allow_network": False, "render_video": True, "include_shorts": False},
    )
    assert sfx_resp.status_code == 200
    body = sfx_resp.json()
    assert body["youtube_click_count"] >= 2
    assert Path(body["video_path"]).is_file()
    assert body["qa"]["passed"] is True
