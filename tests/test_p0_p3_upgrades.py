"""Comprehensive regression and integration tests for P0-P3 Symposium Studio upgrades."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from symposium_studio.api import create_app
from symposium_studio.config import StudioConfig
from symposium_studio.engines.analysis import (
    align_transcript_to_audio,
    apply_transcript_corrections,
    audit_and_verify_transcript,
)
from symposium_studio.engines.analysis.transcribe import _chunk_quality_score, _detect_segment_anomalies
from symposium_studio.engines.analysis.probe import probe_media
from symposium_studio.engines.analysis.semantics import score_highlights, segment_topics
from symposium_studio.engines.distribution import build_social_repurposing_bundle, build_telegram_campaign
from symposium_studio.engines.production import (
    align_captions_to_final_audio,
    apply_iterative_edit,
    build_timeline_from_storyboard,
    build_vertical_short_timelines,
    check_caption_quality,
    evaluate_ending_closure,
    plan_coherent_beat_sequence,
    remap_edited_timeline,
    render_timeline,
    reorder_timeline_clips,
    run_quality_assurance,
    select_best_hook,
    snap_ending_boundary,
    validate_subtitle_integrity,
    write_ass,
)
from symposium_studio.engines.production.timeline_builder import _overlay_placement, _split_segment_into_short_cues
from symposium_studio.engines.research import MemeTrendConnector, download_trending_meme_assets, score_meme_for_beat
from symposium_studio.models.animation import AnimationRenderResult
from symposium_studio.models.media import (
    FaceDetectionBox,
    MediaAnalysisBundle,
    MediaProbeResult,
    SilenceInterval,
    TranscriptResult,
    TranscriptSegment,
    VisualContextSample,
    WordTimestamp,
)
from symposium_studio.models.symposium import Storyboard, StoryboardBeat
from symposium_studio.models.timeline import (
    CaptionCue,
    CaptionSettings,
    RenderTimeline,
    SegmentTimelineMapping,
    TargetPlatformFormat,
    VideoClipNode,
)
from symposium_studio.studio import SymposiumStudio


def _make_speech_video(out_path: Path, duration_sec: float = 12.0) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
            f"color=c=#1E293B:s=1280x720:r=24:d={duration_sec}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:sample_rate=48000:duration={duration_sec}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def _kk_ru_bundle(project_id: str = "proj_kkru", duration_sec: float = 90.0) -> MediaAnalysisBundle:
    segs = [
        TranscriptSegment(
            segment_id="seg_000",
            speaker_id="SPEAKER_00",
            start_sec=0.5,
            end_sec=13.5,
            text="Это просто запись видео себе после двух лет. Я хочу сравнить ожидания и реальность.",
            language="ru",
            confidence=0.92,
            words=[
                WordTimestamp(word=w, start_sec=round(0.6 + i * 0.85, 3), end_sec=round(1.3 + i * 0.85, 3), probability=0.92)
                for i, w in enumerate("Это просто запись видео себе после двух лет. Я хочу сравнить ожидания и реальность.".split())
            ],
        ),
        TranscriptSegment(
            segment_id="seg_001",
            speaker_id="SPEAKER_00",
            start_sec=15.0,
            end_sec=36.0,
            text="Бірінші семестр пока что қиын, бірақ екі жылда major-ымды тауып резюмемді жақсартамын.",
            language="kk",
            confidence=0.87,
            words=[
                WordTimestamp(word=w, start_sec=round(15.2 + i * 1.4, 3), end_sec=round(16.4 + i * 1.4, 3), probability=0.87)
                for i, w in enumerate("Бірінші семестр пока что қиын, бірақ екі жылда major-ымды тауып резюмемді жақсартамын.".split())
            ],
        ),
        TranscriptSegment(
            segment_id="seg_002",
            speaker_id="SPEAKER_00",
            start_sec=38.0,
            end_sec=56.0,
            text="ойлаймын, еще не знаю типа, бир шаруа тауып",
            language="kk",
            confidence=0.45,
            low_confidence=True,
            low_confidence_reason="avg_logprob=-1.21",
            words=[
                WordTimestamp(word=w, start_sec=round(38.2 + i * 1.8, 3), end_sec=round(39.7 + i * 1.8, 3), probability=0.45)
                for i, w in enumerate("ойлаймын, еще не знаю типа, бир шаруа тауып".split())
            ],
        ),
        TranscriptSegment(
            segment_id="seg_003",
            speaker_id="SPEAKER_00",
            start_sec=58.0,
            end_sec=86.0,
            text="Бір аптадан менде Data Structure quiz бар. Осыдан екі жылда қарап туған күнімде ашамын. Хорошо.",
            language="kk",
            confidence=0.90,
            words=[
                WordTimestamp(word=w, start_sec=round(58.2 + i * 1.5, 3), end_sec=round(59.4 + i * 1.5, 3), probability=0.90)
                for i, w in enumerate("Бір аптадан менде Data Structure quiz бар. Осыдан екі жылда қарап туған күнімде ашамын. Хорошо.".split())
            ],
        ),
    ]
    transcript = audit_and_verify_transcript(
        TranscriptResult(
            detected_language="kk",
            language_probability=0.88,
            duration_sec=duration_sec,
            engine="whisper.cpp",
            model="large-v3-q5_0",
            segments=segs,
        )
    )
    topics = segment_topics(transcript.segments)
    visual = [
        VisualContextSample(
            time_sec=float(t),
            frame_width=1280,
            frame_height=720,
            sharpness_score=0.82,
            suggested_crop_center_x=0.52,
            faces=[FaceDetectionBox(time_sec=float(t), x_norm=0.42, y_norm=0.18, w_norm=0.22, h_norm=0.42, center_x_norm=0.53, center_y_norm=0.39)],
        )
        for t in range(0, int(duration_sec), 5)
    ]
    highlights = score_highlights(transcript.segments, topics, visual, duration_sec)
    return MediaAnalysisBundle(
        project_id=project_id,
        probe=MediaProbeResult(
            path="/tmp/fake_kkru.mp4",
            sha256="deadbeef",
            size_bytes=1024,
            duration_sec=duration_sec,
            width=1280,
            height=720,
            fps=30.0,
            video_codec="h264",
            has_audio=True,
        ),
        transcript=transcript,
        topics=topics,
        highlights=highlights,
        visual_samples=visual,
        recommended_crop_center_x=0.52,
    )


def test_transcript_audit_verification_and_patch_propagation(tmp_path: Path) -> None:
    raw = TranscriptResult(
        detected_language="kk",
        language_probability=0.85,
        duration_sec=12.0,
        engine="whisper.cpp",
        model="large-v3-q5_0",
        segments=[
            TranscriptSegment(
                segment_id="seg_0",
                start_sec=0.0,
                end_sec=6.0,
                text="Добавил субтитры DimaTorzok. Это запись видео себе после двух лет.",
                language="ru",
                confidence=0.91,
            ),
            TranscriptSegment(
                segment_id="seg_1",
                start_sec=6.2,
                end_sec=11.8,
                text="структура құйыз панделик Қазақстан Қазақстан Қазақстан",
                language="ko",
                confidence=0.48,
                low_confidence=True,
                candidate_alternatives=[{"language": "kk", "text": "Data Structure quiz понедельник"}],
            ),
        ],
    )
    audited = audit_and_verify_transcript(raw)
    assert "DimaTorzok" not in audited.segments[0].text
    assert audited.quality_report is not None
    assert audited.quality_report.hallucinations_cleaned >= 1
    assert "seg_1" in audited.uncertain_segment_ids
    assert any(item.segment_id == "seg_1" for item in audited.quality_report.review_queue)

    # Wire end-to-end API PATCH /api/projects/{project_id}/transcript without re-running Whisper
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "studio.db")
    studio = SymposiumStudio(cfg)
    manifest = studio.create_project("KK-RU Capsule", editing_style="zoomer_vlog")
    pid = manifest.project.project_id
    bundle = _kk_ru_bundle(pid, duration_sec=90.0)
    bundle.transcript = audited
    (Path(manifest.project.workspace_dir) / "transcripts" / "analysis_bundle.json").write_text(
        bundle.model_dump_json(indent=2), encoding="utf-8"
    )
    studio.db.index_transcript_segments(pid, [s.model_dump(mode="json") for s in bundle.transcript.segments])

    client = TestClient(create_app(cfg))
    resp = client.patch(
        f"/api/projects/{pid}/transcript",
        json={
            "corrections": {"seg_1": "Бір аптадан менде Data Structure quiz понедельник."},
            "language_overrides": {"seg_1": "kk"},
            "mark_confirmed": True,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    seg1 = next(s for s in payload["transcript"]["segments"] if s["segment_id"] == "seg_1")
    assert seg1["text"] == "Бір аптадан менде Data Structure quiz понедельник."
    assert seg1["language"] == "kk"
    assert seg1["verification_status"] == "human_confirmed"
    assert seg1["low_confidence"] is False
    assert len(seg1["words"]) == 7
    assert seg1["words"][0]["start_sec"] >= 6.2
    assert seg1["words"][-1]["end_sec"] <= 11.8

    # Verify FTS search immediately picks up the confirmed wording
    hits = studio.db.search_all("Structure", project_id=pid)
    assert len(hits) >= 1


def test_multilingual_kk_ru_en_semantics_and_directors() -> None:
    bundle = _kk_ru_bundle("proj_directors", duration_sec=90.0)

    # Topics should extract meaningful multilingual domain words without KK/RU stopwords
    all_kw = {k for t in bundle.topics for k in t.keywords}
    assert all(sw not in all_kw for sw in ("осы", "деп", "типа", "это", "просто"))
    assert any(k in all_kw for k in ("data", "structure", "quiz", "семестр", "major", "ожидания"))

    # Highlights should score the dangling low-confidence seg_002 lower than complete premise/payoff beats
    dangling_hls = [h for h in bundle.highlights if "бір" not in h.transcript_excerpt.lower() and "ойлаймын" in h.transcript_excerpt.lower() and h.start_sec >= 35.0 and h.end_sec <= 58.0]
    clean_hls = [h for h in bundle.highlights if "после двух лет" in h.transcript_excerpt.lower() or "data structure" in h.transcript_excerpt.lower()]
    assert clean_hls
    if dangling_hls:
        assert max(h.scores.composite_score for h in clean_hls) > max(h.scores.composite_score for h in dangling_hls)

    # Hook Director must pick clean premise start, not dangling mid-clause seg_002
    hook = select_best_hook(bundle)
    assert hook["clean_start"] is True
    assert "ойлаймын" not in hook["excerpt"].lower()[:15]

    # Ending Director: cutting mid-clause at 76s must be flagged, then snap_ending_boundary extends to 86.3s sentence close
    abrupt = evaluate_ending_closure(bundle.transcript.segments, 58.0, 74.0)
    assert abrupt["closure_ok"] is False
    _, snapped_end, closure = snap_ending_boundary(bundle, 58.0, 74.0, max_extend_sec=14.0)
    assert snapped_end >= 85.5
    assert closure["closure_ok"] is True

    # Plan chronological narrative beats with no duplicate overlapping clips
    seq = plan_coherent_beat_sequence(bundle, target_min_sec=50.0, target_max_sec=80.0, strategy="hook_first")
    assert len(seq) >= 2
    assert seq[0].role == "hook"
    assert seq[-1].role == "payoff"
    for prev, nxt in zip(seq, seq[1:]):
        assert prev.end_sec <= nxt.start_sec + 1.0


def test_caption_quality_gates_wrap_and_safe_zone() -> None:
    bundle = _kk_ru_bundle("proj_caps", duration_sec=35.0)
    sb = Storyboard(
        storyboard_id="sb_cap",
        project_id="proj_caps",
        target_format="youtube_shorts",
        selected_strategy="Hook",
        target_duration_sec=20.0,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=0.5,
                source_end_sec=25.0,
                target_duration_sec=20.0,
                visual_treatment="face_tracked",
                caption_directive="kinetic_caption_pop",
                audio_directive="speech_loudness_polish",
            )
        ],
    )
    tl_v = build_timeline_from_storyboard(
        "proj_caps",
        "Test Caps",
        bundle,
        sb,
        style_name="zoomer_vlog",
        platform_format=TargetPlatformFormat.YOUTUBE_SHORTS,
    )
    res_ok = check_caption_quality(tl_v)
    assert res_ok["ok"] is True, res_ok
    assert res_ok["max_line_chars"] <= 32

    bad_tl = tl_v.model_copy(deep=True)
    bad_tl.caption_settings = CaptionSettings(safe_margin_bottom_px=25)
    res_bad = check_caption_quality(bad_tl)
    assert res_bad["ok"] is False
    assert any("safe margin" in iss.lower() for iss in res_bad["issues"])


def test_meme_connector_no_fake_views_scoring_and_visual_gate(tmp_path: Path) -> None:
    sample_html = '<a href="https://media.tenor.com/abc123AAAPo/locked-in-focus.mp4">Locked In Focus</a>'

    class _MockResp:
        def __init__(self, status_code: int = 200, text: str = "", content: bytes = b"") -> None:
            self.status_code = status_code
            self.text = text
            self.content = content

        def json(self) -> dict[str, object]:
            return {"memes": []}

    with patch("httpx.get", return_value=_MockResp(status_code=200, text=sample_html)):
        items = MemeTrendConnector().fetch_public_items(query="ожидание реальность data structure quiz", limit=4)
    assert len(items) >= 1
    for it in items:
        assert it.metrics.views is None
        assert it.metrics.audience_normalized_ratio is None
        assert it.platform_relative_score == 0.0
        assert "unscored" in it.evidence.lower()

    # Semantic matching & anti-mismatch penalty
    exp_meme = {"topic": "Expectation vs Reality", "title": "Expectation vs Reality Meme", "width": 640, "height": 480, "media_kind": "video"}
    lock_meme = {"topic": "Locked In Studying", "title": "Locked In Grind", "width": 640, "height": 480, "media_kind": "video"}
    assert score_meme_for_beat(exp_meme, "Мен өзімнің ожиданиям мен реальностьті салыстырып көремін.") > 0.7
    assert score_meme_for_beat(exp_meme, "Бір аптадан менде Data Structure quiz бар.") < 0.2
    assert score_meme_for_beat(lock_meme, "Бір аптадан менде Data Structure quiz бар.") > score_meme_for_beat(
        exp_meme, "Бір аптадан менде Data Structure quiz бар."
    )

    # Visual gate rejects narrow tall comic PNG and tiny thumbnail while accepting crisp >=220x180 landscape image
    small_png = tmp_path / "small.png"
    tall_png = tmp_path / "tall.png"
    good_png = tmp_path / "good.png"
    Image.new("RGB", (120, 120), color=(20, 30, 40)).save(small_png)
    Image.new("RGB", (280, 620), color=(20, 30, 40)).save(tall_png)
    # Paint varied stripes so PNG payload exceeds 512 bytes cleanly
    good_im = Image.new("RGB", (480, 360))
    px = good_im.load()
    for y in range(360):
        for x in range(480):
            px[x, y] = ((x * 5 + y) % 256, (y * 7) % 256, (x * 3 + y * 2) % 256)
    good_im.save(good_png)

    from symposium_studio.models.research import ContentCategory, PublicEngagementMetrics, ResearchPlatform, ViralContentItem

    test_items = [
        ViralContentItem(
            item_id="meme_small",
            platform=ResearchPlatform.INSTAGRAM_POST,
            creator="r/memes",
            original_url="https://example.com/small.png",
            title="Tiny meme",
            published_at="2026-11-11",
            category=ContentCategory.ENTERTAINMENT,
            topic="Small",
            opening_hook_structure="reaction",
            editing_style="cut",
            narrative_structure="setup",
            metrics=PublicEngagementMetrics(platform=ResearchPlatform.INSTAGRAM_POST, metric_source="reddit_public_meme_api"),
            evidence="Unscored meme test item",
            confidence=0.8,
            platform_relative_score=0.0,
        ),
        ViralContentItem(
            item_id="meme_tall",
            platform=ResearchPlatform.INSTAGRAM_POST,
            creator="r/memes",
            original_url="https://example.com/tall.png",
            title="Tall comic",
            published_at="2026-11-11",
            category=ContentCategory.ENTERTAINMENT,
            topic="Tall",
            opening_hook_structure="reaction",
            editing_style="cut",
            narrative_structure="setup",
            metrics=PublicEngagementMetrics(platform=ResearchPlatform.INSTAGRAM_POST, metric_source="reddit_public_meme_api"),
            evidence="Unscored meme test item",
            confidence=0.8,
            platform_relative_score=0.0,
        ),
        ViralContentItem(
            item_id="meme_good",
            platform=ResearchPlatform.INSTAGRAM_POST,
            creator="r/memes",
            original_url="https://example.com/good.png",
            title="Clean reaction",
            published_at="2026-11-11",
            category=ContentCategory.ENTERTAINMENT,
            topic="Studying",
            opening_hook_structure="reaction",
            editing_style="cut",
            narrative_structure="setup",
            metrics=PublicEngagementMetrics(platform=ResearchPlatform.INSTAGRAM_POST, metric_source="reddit_public_meme_api"),
            evidence="Unscored meme test item",
            confidence=0.8,
            platform_relative_score=0.0,
        ),
    ]

    data_map = {
        "https://example.com/small.png": small_png.read_bytes(),
        "https://example.com/tall.png": tall_png.read_bytes(),
        "https://example.com/good.png": good_png.read_bytes(),
    }

    def _fake_httpx_get(url: str, timeout: float = 8.0, follow_redirects: bool = True, **kwargs: object) -> _MockResp:
        return _MockResp(status_code=200, content=data_map[str(url)])

    with patch("httpx.get", side_effect=_fake_httpx_get):
        dl = download_trending_meme_assets(test_items, tmp_path / "meme_cache", max_count=4)
    assert len(dl) == 1
    assert dl[0]["item_id"] == "meme_good"
    assert dl[0]["width"] == 480 and dl[0]["height"] == 360


def test_face_safe_corner_overlay_and_qa_lufs_closure(tmp_path: Path) -> None:
    # Face bbox (x0, y0, x1, y1) sits on upper-right side -> _overlay_placement must avoid top-right and pick left corner
    right_face = [(0.62, 0.05, 0.92, 0.40)]
    ox, oy, ow, oh, mode = _overlay_placement(
        is_vertical=False,
        crop_x=0.50,
        is_meme=True,
        slot_idx=1,
        is_direct_meme_asset=True,
        face_boxes=right_face,
    )
    assert mode == "corner_sticker_clean"
    assert ox + ow < 0.62

    # Real short FFmpeg render + QA verification (measured_lufs + caption_quality + closure)
    vid = _make_speech_video(tmp_path / "src.mp4", duration_sec=8.0)
    bundle = _kk_ru_bundle("proj_qa_real", duration_sec=8.0)
    bundle.probe = probe_media(vid)
    bundle.transcript.segments = [
        TranscriptSegment(
            segment_id="seg_q0",
            speaker_id="SPEAKER_00",
            start_sec=0.3,
            end_sec=7.4,
            text="Это запись видео себе после двух лет. Хорошо.",
            language="ru",
            confidence=0.95,
            words=[
                WordTimestamp(word=w, start_sec=round(0.4 + i * 0.7, 3), end_sec=round(0.95 + i * 0.7, 3), probability=0.95)
                for i, w in enumerate("Это запись видео себе после двух лет. Хорошо.".split())
            ],
        )
    ]
    sb = Storyboard(
        storyboard_id="sb_qa",
        project_id="proj_qa_real",
        target_format="youtube_long",
        selected_strategy="Hook -> Closure",
        target_duration_sec=7.5,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=0.2,
                source_end_sec=7.6,
                target_duration_sec=7.4,
                visual_treatment="clean_frame",
                caption_directive="clean_subtitle",
                audio_directive="loudness_polish",
            )
        ],
    )
    tl = build_timeline_from_storyboard("proj_qa_real", "QA Test", bundle, sb, style_name="educational_explainer")
    out_mp4 = tmp_path / "render_qa.mp4"
    render_timeline(tl, out_mp4, tmp_path / "ws_qa")
    qa = run_quality_assurance("proj_qa_real", out_mp4, tl, tmp_path / "qa_frames", analysis=bundle)
    assert qa.passed is True
    assert qa.measured_lufs is not None
    assert -20.5 <= qa.measured_lufs <= -10.5
    assert qa.caption_quality.get("ok") is True
    assert qa.editorial_checks.get("ending_closure", {}).get("closure_ok") is True


def test_bilingual_repurposing_and_hashtags() -> None:
    bundle = _kk_ru_bundle("proj_bil", duration_sec=90.0)
    tg = build_telegram_campaign("proj_bil", "2 Year Time Capsule", bundle, "@capsule_club")
    sb = Storyboard(
        storyboard_id="sb_bil",
        project_id="proj_bil",
        target_format="youtube_long",
        selected_strategy="Capsule",
        target_duration_sec=60.0,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=0.5,
                source_end_sec=20.0,
                target_duration_sec=19.0,
                visual_treatment="talking_head",
                caption_directive="kinetic_caption_pop",
                audio_directive="speech_loudness_polish",
            )
        ],
    )
    tl = build_timeline_from_storyboard("proj_bil", "2 Year Time Capsule", bundle, sb, style_name="zoomer_vlog")
    social = build_social_repurposing_bundle("proj_bil", "2 Year Time Capsule", bundle, tl, tg)

    # Bilingual KK/RU footage should not get English Faraday physics defaults
    assert all("WHY IT FLOWS" != tc.get("overlay_text") for tc in social.youtube.thumbnail_concepts)
    assert any(any(ord(ch) > 127 for ch in title) for title in social.youtube.title_candidates)
    assert any(len(sc.hashtags) >= 2 and all(len(tag) >= 3 for tag in sc.hashtags) for sc in social.shorts_copy)
    assert len(social.threads_posts) == 6
    assert all(p.char_count <= 280 for p in social.threads_posts)


def test_acoustic_word_alignment_respects_silence_and_gaps(tmp_path: Path) -> None:
    sr = 16000
    dur = 8.2
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    rng = np.random.default_rng(42)
    sig = rng.normal(0.0, 0.0012, size=t.shape)
    for a, b, amp in [(1.0, 2.4, 0.08), (4.4, 5.2, 0.075), (6.5, 7.6, 0.075)]:
        m = (t >= a) & (t <= b)
        sig[m] += amp * (0.65 * np.sin(2 * np.pi * 240.0 * t[m]) + 0.35 * np.sin(2 * np.pi * 520.0 * t[m]))
    pcm = np.clip(sig * 32767.0, -32767, 32767).astype(np.int16)
    wav_path = tmp_path / "bursts.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    tr = TranscriptResult(
        engine="test",
        model="test",
        detected_language="ru",
        language_probability=1.0,
        duration_sec=dur,
        segments=[
            TranscriptSegment(
                segment_id="seg_b0",
                speaker_id="SPEAKER_00",
                start_sec=0.6,
                end_sec=7.9,
                text="Это видео я записываю себе после двух лет",
                language="ru",
                confidence=1.0,
            )
        ],
    )
    linear = apply_transcript_corrections(tr, {"seg_b0": "Это видео я записываю себе после двух лет"})
    aligned = align_transcript_to_audio(linear, wav_path, force_all=True)
    words = aligned.segments[0].words
    assert len(words) == 8
    assert words[0].start_sec >= 0.90
    # No word may sit inside the 2.0s pause [2.45..4.35] or 1.3s pause [5.25..6.45]
    for w in words:
        mid = 0.5 * (w.start_sec + w.end_sec)
        assert not (2.48 <= mid <= 4.32), f"Word {w} drifted into first silence"
        assert not (5.28 <= mid <= 6.42), f"Word {w} drifted into second silence"
        assert (w.end_sec - w.start_sec) <= 1.2


def test_karaoke_wait_tags_and_no_duplicate_subclip_caption(tmp_path: Path) -> None:
    cue = CaptionCue(
        cue_id="cap_k1",
        start_sec=10.0,
        end_sec=13.0,
        text="Привет там пока",
        words=[
            {"word": "Привет", "start_sec": 0.0, "end_sec": 0.4, "probability": 1.0},
            {"word": "там", "start_sec": 1.2, "end_sec": 1.6, "probability": 1.0},
            {"word": "пока", "start_sec": 2.4, "end_sec": 2.9, "probability": 1.0},
        ],
    )
    tl = RenderTimeline(
        timeline_id="tl_karaoke",
        project_id="proj_k",
        title="Karaoke Wait",
        width=1280,
        height=720,
        fps=30.0,
        caption_settings=CaptionSettings(enabled=True, highlight_word=True, max_words_per_cue=4),
        captions=[cue],
    )
    ass_path = write_ass(tl, tmp_path / "k.ass")
    ass_txt = ass_path.read_text(encoding="utf-8")
    # 0.8s wait between words must emit explicit {\k80} delay before {\kf40}там
    assert "{\\k80}{\\kf40}там" in ass_txt
    assert "{\\k80}{\\kf50}пока" in ass_txt

    seg = TranscriptSegment(
        segment_id="seg_dup",
        speaker_id="SPEAKER_00",
        start_sec=0.0,
        end_sec=8.0,
        text="Первое слово потом второе",
        language="ru",
        words=[
            WordTimestamp(word="Первое", start_sec=0.2, end_sec=0.8, probability=0.98),
            WordTimestamp(word="слово", start_sec=0.8, end_sec=1.3, probability=0.98),
            WordTimestamp(word="потом", start_sec=6.2, end_sec=6.8, probability=0.98),
            WordTimestamp(word="второе", start_sec=6.8, end_sec=7.5, probability=0.98),
        ],
    )
    # A breather subclip strictly inside [1.5, 5.9] between words must emit ZERO captions
    cues_gap = _split_segment_into_short_cues(
        seg=seg,
        clip_src_start=1.6,
        clip_src_end=5.8,
        clip_tl_start=0.0,
        speed=1.0,
        max_words=4,
        style_preset="zoomer_vlog",
    )
    assert cues_gap == []


def test_multi_clip_render_zero_concat_drift(tmp_path: Path) -> None:
    src = _make_speech_video(tmp_path / "src_drift.mp4", duration_sec=12.0)
    clips: list[VideoClipNode] = []
    cur = 0.0
    pairs = [(0.5, 1.85, 1.0), (2.2, 3.65, 1.08), (4.0, 5.25, 1.0), (5.5, 7.15, 1.05), (7.4, 8.9, 1.0), (9.2, 10.6, 1.0)]
    for i, (s, e, spd) in enumerate(pairs):
        dur_tl = (e - s) / spd
        clips.append(
            VideoClipNode(
                clip_id=f"clip_{i}",
                source_path=str(src),
                source_start_sec=s,
                source_end_sec=e,
                timeline_start_sec=round(cur, 4),
                timeline_end_sec=round(cur + dur_tl, 4),
                playback_speed=spd,
                zoom_scale=1.0 + (0.04 if i % 2 else 0.0),
            )
        )
        cur += dur_tl

    tl = RenderTimeline(
        timeline_id="tl_zero_drift",
        project_id="proj_drift",
        title="Zero Drift",
        width=640,
        height=360,
        fps=30.0,
        clips=clips,
        captions=[
            CaptionCue(
                cue_id="c_end",
                start_sec=round(cur - 1.0, 3),
                end_sec=round(cur - 0.1, 3),
                text="Финал синхронный",
            )
        ],
    )
    out_mp4 = tmp_path / "out_zero_drift.mp4"
    render_timeline(tl, out_mp4, tmp_path / "ws_drift")
    probed = probe_media(out_mp4)
    assert abs(probed.duration_sec - cur) <= 0.06, f"Expected {cur:.3f}s, got {probed.duration_sec:.3f}s"


def _make_pulsed_speech_video(
    out_path: Path,
    duration_sec: float,
    speech_windows: list[tuple[float, float]],
) -> Path:
    """Create test MP4 with real quiet background and pulsed speech-energy envelopes."""
    sr = 16000
    n_samples = int(round(duration_sec * sr))
    t = np.arange(n_samples, dtype=np.float32) / sr
    # Very quiet ambient room floor + bright AM-modulated harmonic signal during speech windows
    samples = 0.0008 * np.sin(2.0 * np.pi * 60.0 * t)
    for ws, we in speech_windows:
        i0 = max(0, min(n_samples, int(round(ws * sr))))
        i1 = max(0, min(n_samples, int(round(we * sr))))
        if i1 > i0:
            seg_t = t[i0:i1] - ws
            env = 0.38 * (0.55 + 0.45 * np.sin(2.0 * np.pi * 4.5 * seg_t))
            samples[i0:i1] += env * (
                0.7 * np.sin(2.0 * np.pi * 240.0 * seg_t) + 0.3 * np.sin(2.0 * np.pi * 520.0 * seg_t)
            )
    pcm = np.clip(samples * 32767.0, -32767, 32767).astype(np.int16)
    wav_path = out_path.with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

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
            f"testsrc=size=640x360:rate=24:d={duration_sec}",
            "-i",
            str(wav_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def test_subtitle_source_vs_output_mapping_reordered_cuts_and_meme_inserts(tmp_path: Path) -> None:
    vid = _make_speech_video(tmp_path / "src_reord.mp4", duration_sec=20.0)
    ov_mov = tmp_path / "meme_ov.mov"
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
            "color=c=orange@0.8:s=320x240:d=2.8:r=24",
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

    seg_a = TranscriptSegment(
        segment_id="seg_a",
        speaker_id="SPEAKER_00",
        start_sec=1.0,
        end_sec=4.4,
        text="Осы видеоға қарап мен өзімнің",
        language="kk",
        confidence=0.94,
        words=[
            WordTimestamp(word=w, start_sec=round(1.1 + i * 0.62, 3), end_sec=round(1.62 + i * 0.62, 3), probability=0.94)
            for i, w in enumerate("Осы видеоға қарап мен өзімнің".split())
        ],
    )
    seg_b = TranscriptSegment(
        segment_id="seg_b",
        speaker_id="SPEAKER_00",
        start_sec=8.0,
        end_sec=12.1,
        text="Өзінің major-ын тауып, Data Structure quiz",
        language="kk",
        confidence=0.95,
        words=[
            WordTimestamp(word=w, start_sec=round(8.1 + i * 0.64, 3), end_sec=round(8.66 + i * 0.64, 3), probability=0.95)
            for i, w in enumerate("Өзінің major-ын тауып, Data Structure quiz".split())
        ],
    )
    seg_c = TranscriptSegment(
        segment_id="seg_c",
        speaker_id="SPEAKER_00",
        start_sec=15.0,
        end_sec=18.6,
        text="Я надеюсь всё получится хорошо.",
        language="ru",
        confidence=0.95,
        words=[
            WordTimestamp(word=w, start_sec=round(15.1 + i * 0.65, 3), end_sec=round(15.65 + i * 0.65, 3), probability=0.95)
            for i, w in enumerate("Я надеюсь всё получится хорошо.".split())
        ],
    )
    bundle = MediaAnalysisBundle(
        project_id="proj_reord",
        probe=probe_media(vid),
        transcript=TranscriptResult(
            engine="whisper.cpp",
            model="test",
            detected_language="kk",
            language_probability=0.9,
            duration_sec=20.0,
            segments=[seg_a, seg_b, seg_c],
            languages_observed=["kk", "ru", "en"],
        ),
        silences=[
            SilenceInterval(start_sec=4.5, end_sec=7.9, duration_sec=3.4),
            SilenceInterval(start_sec=12.2, end_sec=14.9, duration_sec=2.7),
        ],
    )
    # Deliberately reorder beats: seg_b first, then seg_a (with meme overlay), then seg_c
    sb = Storyboard(
        storyboard_id="sb_reord",
        project_id="proj_reord",
        selected_strategy="Hook Swap + Meme Insert",
        target_duration_sec=11.1,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=8.0,
                source_end_sec=12.1,
                target_duration_sec=4.1,
                visual_treatment="jump_cut_and_zoom",
            ),
            StoryboardBeat(
                beat_index=1,
                role="proof",
                source_start_sec=1.0,
                source_end_sec=4.4,
                target_duration_sec=3.4,
                animation_template="meme_reaction_card",
                visual_treatment="jump_cut_and_zoom",
            ),
            StoryboardBeat(
                beat_index=2,
                role="payoff",
                source_start_sec=15.0,
                source_end_sec=18.6,
                target_duration_sec=3.6,
                visual_treatment="clean_frame",
            ),
        ],
    )
    meme_anim = AnimationRenderResult(
        animation_id="anim_meme_reord",
        scene_id="sc_reord",
        template_id="meme_reaction_card",
        adapter_used="chromium_canvas",
        scene_spec_path=str(tmp_path / "missing.json"),
        output_path=str(ov_mov),
        duration_sec=2.8,
        has_alpha=True,
        width=320,
        height=240,
        fps=24,
        render_time_sec=0.1,
        factuality_mode="artistic_illustration",
        beat_start_sec=1.0,
        beat_end_sec=4.4,
    )
    tl = build_timeline_from_storyboard(
        project_id="proj_reord",
        title="Reordered Multilingual Timeline",
        analysis=bundle,
        storyboard=sb,
        style_name="zoomer_vlog",
        animation_result=meme_anim,
    )
    assert tl.caption_settings.max_words_per_line <= 3
    assert all(len(c.text.split()) <= 3 for c in tl.captions)
    assert len(tl.overlays) >= 1
    assert tl.segment_mappings, "Expected explicit SegmentTimelineMapping records"
    mapped_segs = {m.source_segment_id for m in tl.segment_mappings}
    assert {"seg_a", "seg_b", "seg_c"}.issubset(mapped_segs)

    # First cue in output comes from seg_b (src >= 8.0s), NOT unadjusted 8.0s output timestamp
    first_cue = tl.captions[0]
    assert first_cue.source_segment_id == "seg_b"
    assert first_cue.source_start_sec is not None and first_cue.source_start_sec >= 8.0
    assert first_cue.start_sec < 1.0
    assert validate_subtitle_integrity(tl)["passed"] is True

    # Test reordering clips and iterative edits: must update cues, word timestamps, overlays, and mappings cleanly
    reversed_ids = [c.clip_id for c in reversed(tl.clips)]
    reordered_tl = reorder_timeline_clips(tl, reversed_ids)
    assert reordered_tl.captions[0].source_segment_id == "seg_c"
    assert reordered_tl.captions[0].start_sec < 1.0
    for c in reordered_tl.captions:
        for w in c.words:
            assert w["start_sec"] >= c.start_sec - 0.02
            assert w["end_sec"] <= c.end_sec + 0.02
    assert validate_subtitle_integrity(reordered_tl)["passed"] is True

    trimmed_tl, notes_trim = apply_iterative_edit(tl, "Trim 0.4s from the opening hook")
    assert notes_trim
    assert trimmed_tl.clips[0].source_start_sec > tl.clips[0].source_start_sec
    assert validate_subtitle_integrity(trimmed_tl)["passed"] is True

    faster_tl, _ = apply_iterative_edit(trimmed_tl, "Increase speed pace 1.15x faster")
    assert validate_subtitle_integrity(faster_tl)["passed"] is True
    assert faster_tl.clips[-1].timeline_end_sec < trimmed_tl.clips[-1].timeline_end_sec


def test_subtitle_integrity_catches_missing_duplicate_order_and_stale_source() -> None:
    clip = VideoClipNode(
        clip_id="clip_0",
        source_path="dummy.mp4",
        source_start_sec=15.0,
        source_end_sec=18.5,
        timeline_start_sec=0.0,
        timeline_end_sec=3.5,
        playback_speed=1.0,
    )
    bad_tl = RenderTimeline(
        timeline_id="tl_bad_sub",
        project_id="proj_bad_sub",
        title="Adversarial Subtitle Failures",
        clips=[clip],
        segment_mappings=[
            SegmentTimelineMapping(
                mapping_id="map_missing",
                source_segment_id="seg_missing",
                clip_id="clip_0",
                source_start_sec=17.4,
                source_end_sec=18.2,
                timeline_start_sec=2.4,
                timeline_end_sec=3.2,
                cue_ids=["nonexistent_cue"],
                text_excerpt="Skipped segment",
            )
        ],
        captions=[
            CaptionCue(
                cue_id="c1",
                clip_id="clip_0",
                source_segment_id="seg_1",
                source_start_sec=15.1,
                source_end_sec=16.0,
                start_sec=1.6,  # non-monotonic: starts after c2
                end_sec=2.1,
                text="Data Structure quiz",
            ),
            CaptionCue(
                cue_id="c2",
                clip_id="clip_0",
                source_segment_id="seg_1",
                source_start_sec=15.2,
                source_end_sec=16.2,
                start_sec=1.0,  # out of order vs c1
                end_sec=1.5,
                text="Data Structure quiz",  # duplicated consecutive text
            ),
            CaptionCue(
                cue_id="c3",
                clip_id="clip_0",
                source_segment_id="seg_1",
                source_start_sec=15.4,
                source_end_sec=16.5,
                start_sec=15.4,  # unadjusted raw source timestamp leaked outside clip [0.0, 3.5]!
                end_sec=16.5,
                text="Unadjusted timestamp",
            ),
        ],
    )
    report = validate_subtitle_integrity(bad_tl, expected_segment_ids={"seg_1", "seg_expected_missing"})
    assert report["passed"] is False
    missing_codes = {m["code"] for m in report["missing"]}
    dup_codes = {d["code"] for d in report["duplicated"]}
    order_codes = {o["code"] for o in report["out_of_order"]}
    mis_codes = {m["code"] for m in report["misaligned"]}
    assert "MISSING_CAPTION_FOR_MAPPED_SEGMENT" in missing_codes
    assert "EXPECTED_SEGMENT_UNCOVERED" in missing_codes
    assert "DUPLICATE_CAPTION_TEXT" in dup_codes
    assert "NON_MONOTONIC_CUE" in order_codes
    assert "OUTSIDE_HOST_CLIP_BOUNDS" in mis_codes
    assert "UNADJUSTED_SOURCE_TIMESTAMP_REUSE" in mis_codes


def test_final_edited_audio_alignment_after_cuts_pauses_and_meme_inserts(tmp_path: Path) -> None:
    speech_windows = [(1.25, 3.35), (6.50, 9.20), (10.75, 13.05)]
    src = _make_pulsed_speech_video(tmp_path / "src_pulsed.mp4", duration_sec=14.0, speech_windows=speech_windows)
    meme_ov = tmp_path / "meme_cut.mov"
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
            "color=c=purple@0.85:s=320x240:d=2.4:r=24",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4444",
            "-pix_fmt",
            "yuva444p10le",
            str(meme_ov),
        ],
        check=True,
    )

    seg0 = TranscriptSegment(
        segment_id="seg_p0",
        speaker_id="SPEAKER_00",
        start_sec=1.20,
        end_sec=3.40,
        text="Бірінші семестр қиын",
        language="kk",
        confidence=0.95,
        words=[
            WordTimestamp(word="Бірінші", start_sec=1.25, end_sec=1.90, probability=0.95),
            WordTimestamp(word="семестр", start_sec=1.92, end_sec=2.60, probability=0.95),
            WordTimestamp(word="қиын", start_sec=2.62, end_sec=3.32, probability=0.95),
        ],
    )
    seg1 = TranscriptSegment(
        segment_id="seg_p1",
        speaker_id="SPEAKER_00",
        start_sec=6.45,
        end_sec=9.25,
        text="Data Structure quiz ертең",
        language="en",
        confidence=0.94,
        words=[
            WordTimestamp(word="Data", start_sec=6.50, end_sec=7.05, probability=0.94),
            WordTimestamp(word="Structure", start_sec=7.10, end_sec=7.85, probability=0.94),
            WordTimestamp(word="quiz", start_sec=7.90, end_sec=8.45, probability=0.94),
            WordTimestamp(word="ертең", start_sec=8.50, end_sec=9.15, probability=0.94),
        ],
    )
    seg2 = TranscriptSegment(
        segment_id="seg_p2",
        speaker_id="SPEAKER_00",
        start_sec=10.70,
        end_sec=13.10,
        text="Хорошо, надеюсь всё.",
        language="ru",
        confidence=0.95,
        words=[
            WordTimestamp(word="Хорошо,", start_sec=10.75, end_sec=11.45, probability=0.95),
            WordTimestamp(word="надеюсь", start_sec=11.50, end_sec=12.25, probability=0.95),
            WordTimestamp(word="всё.", start_sec=12.30, end_sec=13.02, probability=0.95),
        ],
    )
    bundle = MediaAnalysisBundle(
        project_id="proj_final_align",
        probe=probe_media(src),
        transcript=TranscriptResult(
            engine="whisper.cpp",
            model="test",
            detected_language="kk",
            language_probability=0.9,
            duration_sec=14.0,
            segments=[seg0, seg1, seg2],
            languages_observed=["kk", "en", "ru"],
        ),
        silences=[
            SilenceInterval(start_sec=0.0, end_sec=1.15, duration_sec=1.15),
            SilenceInterval(start_sec=3.45, end_sec=6.40, duration_sec=2.95),
            SilenceInterval(start_sec=9.30, end_sec=10.65, duration_sec=1.35),
        ],
    )
    # Notice clip 0 starts at 0.35s (with leading silence before speech at 1.25s!)
    sb = Storyboard(
        storyboard_id="sb_final_align",
        project_id="proj_final_align",
        selected_strategy="Cut Pauses + Meme Overlay + Align Final Audio",
        target_duration_sec=9.4,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=0.35,
                source_end_sec=3.55,
                target_duration_sec=3.2,
                visual_treatment="jump_cut_and_zoom",
            ),
            StoryboardBeat(
                beat_index=1,
                role="proof",
                source_start_sec=6.20,
                source_end_sec=9.40,
                target_duration_sec=3.2,
                animation_template="meme_reaction_card",
                visual_treatment="punchy_contrast",
            ),
            StoryboardBeat(
                beat_index=2,
                role="payoff",
                source_start_sec=10.45,
                source_end_sec=13.30,
                target_duration_sec=2.85,
                visual_treatment="clean_frame",
            ),
        ],
    )
    anim = AnimationRenderResult(
        animation_id="anim_p",
        scene_id="sc_p",
        template_id="meme_reaction_card",
        adapter_used="chromium_canvas",
        scene_spec_path=str(tmp_path / "missing.json"),
        output_path=str(meme_ov),
        duration_sec=2.4,
        has_alpha=True,
        width=320,
        height=240,
        fps=24,
        render_time_sec=0.1,
        factuality_mode="artistic_illustration",
        beat_start_sec=6.20,
        beat_end_sec=9.40,
    )
    tl = build_timeline_from_storyboard(
        project_id="proj_final_align",
        title="Final Audio Alignment",
        analysis=bundle,
        storyboard=sb,
        style_name="zoomer_vlog",
        animation_result=anim,
    )
    # Intentionally inject mild drift onto initial cues before rendering to prove final-audio re-alignment fixes it
    tl.captions[0].start_sec = 0.05
    tl.captions[0].end_sec = 1.15  # right inside leading silence!

    out_mp4 = tmp_path / "aligned_render.mp4"
    res = render_timeline(tl, out_mp4, tmp_path / "ws_align")
    rendered_tl = res["rendered_timeline"]
    integrity = res["subtitle_integrity"]
    assert integrity["passed"] is True
    assert integrity["aligned_cue_count"] == len(rendered_tl.captions)
    # Cue 0 originally pushed back into leading silence [0.05s] -> must align forward onto active speech pulse (>=0.28s)
    assert rendered_tl.captions[0].start_sec >= 0.28
    assert rendered_tl.captions[0].source_start_sec == 1.25  # source timestamp stays preserved separately

    qa = run_quality_assurance("proj_final_align", out_mp4, rendered_tl, tmp_path / "qa_align", analysis=bundle)
    assert qa.passed is True
    assert qa.caption_quality.get("integrity", {}).get("passed") is True


def test_multilingual_asr_repetition_loop_and_code_switch_detection() -> None:
    clean_seg = TranscriptSegment(
        segment_id="s_clean",
        speaker_id="SPEAKER_00",
        start_sec=0.0,
        end_sec=4.2,
        text="Өзінің major-ын тауып, Data Structure quiz нормально.",
        language="kk",
        confidence=0.92,
        words=[
            WordTimestamp(word=w, start_sec=round(0.2 + i * 0.55, 3), end_sec=round(0.7 + i * 0.55, 3), probability=0.92)
            for i, w in enumerate("Өзінің major-ын тауып, Data Structure quiz нормально.".split())
        ],
    )
    loop_seg = TranscriptSegment(
        segment_id="s_loop",
        speaker_id="SPEAKER_00",
        start_sec=5.0,
        end_sec=19.0,
        text="Қазақстан, қазақстан. Қазақстан. Қазақстан. Қазақстан.",
        language="kk",
        confidence=0.80,
    )
    sparse_seg = TranscriptSegment(
        segment_id="s_sparse",
        speaker_id="SPEAKER_00",
        start_sec=20.0,
        end_sec=29.0,
        text="Сәб.",
        language="kk",
        confidence=0.72,
    )
    garble_seg = TranscriptSegment(
        segment_id="s_garble",
        speaker_id="SPEAKER_00",
        start_sec=30.0,
        end_sec=34.0,
        text="Бір аптадан менде структура құйыз бар.",
        language="kk",
        confidence=0.65,
    )
    ru_seg = TranscriptSegment(
        segment_id="s_ru",
        speaker_id="SPEAKER_00",
        start_sec=35.0,
        end_sec=38.2,
        text="Надеюсь, всё будет хорошо.",
        language="ru",
        confidence=0.94,
    )

    assert _chunk_quality_score("kk", [clean_seg]) > _chunk_quality_score("kk", [loop_seg]) + 0.40
    assert _chunk_quality_score("kk", [clean_seg]) > _chunk_quality_score("kk", [sparse_seg]) + 0.30

    assert "single_token_repeat_loop" in _detect_segment_anomalies(loop_seg)
    assert "sparse_words_over_long_window" in _detect_segment_anomalies(sparse_seg)
    assert "possible_code_switch_phonetic_garble" in _detect_segment_anomalies(garble_seg)

    tr = TranscriptResult(
        engine="whisper.cpp",
        model="test",
        detected_language="kk",
        language_probability=0.88,
        duration_sec=39.0,
        segments=[clean_seg, loop_seg, sparse_seg, garble_seg, ru_seg],
    )
    audited = audit_and_verify_transcript(tr)
    assert {"kk", "en", "ru"}.issubset(set(audited.languages_observed))
    assert audited.quality_report is not None
    rq_ids = {item.segment_id for item in audited.quality_report.review_queue}
    assert {"s_loop", "s_sparse", "s_garble"}.issubset(rq_ids)
    assert "Data Structure quiz" in audited.segments[0].text


def test_p0_word_level_asr_alignment_beyond_energy_envelopes(tmp_path: Path) -> None:
    """Verify align_captions_to_final_audio anchors directly to decoded ASR word timestamps rather than RMS alone."""
    wav_path = tmp_path / "speech_final.wav"
    sr = 16000
    dur_sec = 5.0
    n_samples = int(sr * dur_sec)
    t = np.linspace(0.0, dur_sec, n_samples, endpoint=False)
    sig = 0.002 * np.sin(2.0 * np.pi * 180.0 * t)
    # Create steady uniform modulation from 0.4s to 4.2s so RMS envelopes alone cannot tell words apart
    m = (t >= 0.4) & (t <= 4.2)
    sig[m] = 0.35 * np.sin(2.0 * np.pi * 240.0 * t[m])
    pcm = (np.clip(sig, -0.95, 0.95) * 32767.0).astype(np.int16)
    import wave

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    clip = VideoClipNode(
        clip_id="clip_0",
        source_path="dummy.mp4",
        source_start_sec=10.0,
        source_end_sec=15.0,
        timeline_start_sec=0.0,
        timeline_end_sec=5.0,
    )
    tl = RenderTimeline(
        timeline_id="tl_asr_sync",
        project_id="proj_asr_sync",
        title="ASR Word Sync Test",
        clips=[clip],
        segment_mappings=[
            SegmentTimelineMapping(
                mapping_id="m0",
                source_segment_id="seg_0",
                clip_id="clip_0",
                source_start_sec=10.3,
                source_end_sec=14.0,
                timeline_start_sec=0.3,
                timeline_end_sec=4.0,
                language="kk",
                text_excerpt="Бірінші тест нормально",
                cue_ids=["cue_0", "cue_1"],
            )
        ],
        captions=[
            CaptionCue(
                cue_id="cue_0",
                clip_id="clip_0",
                source_segment_id="seg_0",
                source_start_sec=10.3,
                source_end_sec=11.8,
                start_sec=0.3,
                end_sec=1.8,
                text="Бірінші тест",
                words=[
                    {"word": "Бірінші", "start_sec": 0.3, "end_sec": 1.0, "probability": 0.9},
                    {"word": "тест", "start_sec": 1.0, "end_sec": 1.8, "probability": 0.9},
                ],
            ),
            CaptionCue(
                cue_id="cue_1",
                clip_id="clip_0",
                source_segment_id="seg_0",
                source_start_sec=12.0,
                source_end_sec=13.5,
                start_sec=2.0,
                end_sec=3.5,
                text="нормально",
                words=[
                    {"word": "нормально", "start_sec": 2.0, "end_sec": 3.5, "probability": 0.9},
                ],
            ),
        ],
    )

    # Actual final audio pronounced "Бірінші" at 1.45-2.05, skipped/merged a middle token slightly, "нормально" at 3.25-3.95!
    asr_words = [
        {"word": "Бірінші", "start_sec": 1.45, "end_sec": 2.05, "probability": 0.94},
        {"word": "test", "start_sec": 2.20, "end_sec": 2.65, "probability": 0.91},  # cross-script match for тест!
        {"word": "нормально", "start_sec": 3.25, "end_sec": 3.95, "probability": 0.96},
    ]

    aligned = align_captions_to_final_audio(tl, wav_path, final_asr_words=asr_words, use_asr_alignment=False)
    c0, c1 = aligned.captions[0], aligned.captions[1]
    assert c0.aligned_to_final_audio is True
    assert abs(c0.words[0]["start_sec"] - 1.45) <= 0.02
    assert abs(c0.words[1]["start_sec"] - 2.20) <= 0.02
    assert abs(c1.words[0]["start_sec"] - 3.25) <= 0.02
    assert c0.source_start_sec == 10.3  # source timestamp stays untouched!
    assert c0.words[0]["alignment_source"] == "asr_word_match"

    integrity_ok = validate_subtitle_integrity(aligned, final_audio_wav=wav_path, final_asr_words=asr_words)
    assert integrity_ok["passed"] is True
    assert integrity_ok["asr_matched_word_count"] == 3
    assert integrity_ok["mean_word_onset_drift_sec"] is not None
    assert integrity_ok["mean_word_onset_drift_sec"] <= 0.05

    # Sabotage a matched word timestamp to verify ASR_WORD_DRIFT_EXCEEDED catches sync drift
    bad = aligned.model_copy(deep=True)
    bad.captions[1].words[0]["start_sec"] = 2.55
    bad.captions[1].words[0]["end_sec"] = 2.95
    integrity_bad = validate_subtitle_integrity(bad, final_audio_wav=wav_path, final_asr_words=asr_words)
    assert any(m["code"] == "ASR_WORD_DRIFT_EXCEEDED" for m in integrity_bad["misaligned"])


def test_p0_multilingual_asr_verification_parity_mlx_and_faster_whisper(tmp_path: Path) -> None:
    """Ensure mlx-whisper and faster-whisper execute identical chunked kk/ru/en alternate verification as whisper.cpp."""
    from types import SimpleNamespace
    from symposium_studio.engines.analysis.transcribe import (
        _transcribe_with_faster_whisper,
        _transcribe_with_mlx_whisper,
    )

    wav_path = tmp_path / "multilingual_32s.wav"
    sr = 16000
    dur = 32.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    # Speech bursts separated by pauses around 16s so _plan_wav_chunks yields >=2 chunks
    sig = np.zeros_like(t)
    sig[(t >= 1.0) & (t <= 14.0)] = 0.25 * np.sin(2.0 * np.pi * 220.0 * t[(t >= 1.0) & (t <= 14.0)])
    sig[(t >= 17.0) & (t <= 30.0)] = 0.25 * np.sin(2.0 * np.pi * 260.0 * t[(t >= 17.0) & (t <= 30.0)])
    import wave

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(sig, -0.95, 0.95) * 32767.0).astype(np.int16).tobytes())

    class FakeMLX:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def transcribe(self, path: str, **kwargs: Any) -> dict[str, Any]:
            lang = kwargs.get("language")
            self.calls.append((Path(path).name, lang))
            if "000" in Path(path).name:
                return {
                    "language": "kk",
                    "segments": [
                        {"start": 0.5, "end": 4.0, "text": "Бірінші курс өте қызық", "avg_logprob": -0.25, "no_speech_prob": 0.02}
                    ],
                }
            if lang is None or lang == "ko":
                # Outlier Korean misfire + repetition hallucination loop
                return {
                    "language": "ko",
                    "segments": [
                        {"start": 0.4, "end": 5.0, "text": "가나다 가나다 가나다 가나다", "avg_logprob": -1.25, "no_speech_prob": 0.35}
                    ],
                }
            if lang == "en":
                return {
                    "language": "en",
                    "segments": [
                        {"start": 0.4, "end": 5.0, "text": "Data Structure quiz tomorrow", "avg_logprob": -0.18, "no_speech_prob": 0.02}
                    ],
                }
            return {
                "language": lang,
                "segments": [
                    {"start": 0.4, "end": 5.0, "text": "непонятно что", "avg_logprob": -0.85, "no_speech_prob": 0.20}
                ],
            }

    fake_mlx = FakeMLX()
    res_mlx = _transcribe_with_mlx_whisper(wav_path, dur, "mlx-community/whisper-small-mlx", language=None, mlx_module=fake_mlx)
    assert {"kk", "en"}.issubset(set(res_mlx.languages_observed))
    assert any(s.verification_status == "alternate_hypothesis_selected" and "Data Structure" in s.text for s in res_mlx.segments)
    assert any(s.candidate_alternatives for s in res_mlx.segments)

    class FakeFWModel:
        def transcribe(self, path: str, language: str | None = None, word_timestamps: bool = True) -> tuple[list[Any], Any]:
            if "000" in Path(path).name:
                segs = [SimpleNamespace(start=0.5, end=4.0, text="Бүгін лаборатория бар", avg_logprob=-0.22, no_speech_prob=0.03, words=[])]
                return segs, SimpleNamespace(language="kk", language_probability=0.92)
            if language is None:
                segs = [SimpleNamespace(start=0.4, end=5.0, text="тест тест тест тест тест", avg_logprob=-1.35, no_speech_prob=0.3, words=[])]
                return segs, SimpleNamespace(language="tr", language_probability=0.41)
            if language == "ru":
                segs = [SimpleNamespace(start=0.4, end=5.0, text="И сейчас проверим генератор", avg_logprob=-0.16, no_speech_prob=0.02, words=[])]
                return segs, SimpleNamespace(language="ru", language_probability=0.96)
            segs = [SimpleNamespace(start=0.4, end=5.0, text="stub", avg_logprob=-0.95, no_speech_prob=0.35, words=[])]
            return segs, SimpleNamespace(language=language, language_probability=0.55)

    res_fw = _transcribe_with_faster_whisper(wav_path, dur, "small", language=None, whisper_model=FakeFWModel())
    assert {"kk", "ru"}.issubset(set(res_fw.languages_observed))
    assert any(s.verification_status == "alternate_hypothesis_selected" and "проверим генератор" in s.text for s in res_fw.segments)


def test_p0_qa_gate_blocks_ready_for_publish_and_approved_publish(tmp_path: Path) -> None:
    """Critical QA or subtitle integrity errors must set requires_review and block approved_publish / API publish."""
    import pytest
    from fastapi.testclient import TestClient
    from symposium_studio.api.app import create_app
    from symposium_studio.config import StudioConfig
    from symposium_studio.engines.distribution import execute_publish
    from symposium_studio.engines.production import evaluate_publish_readiness
    from symposium_studio.models.social import PublishingMode, PublishRequest, QualityAssuranceReport, QualityCheckIssue
    from symposium_studio.studio import SymposiumStudio

    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db")
    studio = SymposiumStudio(cfg)
    manifest = studio.create_project("QA Gate Project")
    pid = manifest.project.project_id
    studio.ingest_video(pid, _make_pulsed_speech_video(tmp_path / "src.mp4", duration_sec=6.0, speech_windows=[(0.5, 5.2)]))
    manifest = studio.workspaces.get_manifest(pid)
    ws_dir = Path(manifest.project.workspace_dir)

    bad_qa = QualityAssuranceReport(
        qa_id="qa_fail",
        project_id=pid,
        artifact_path=str(ws_dir / "renders" / "out.mp4"),
        passed=False,
        width=1920,
        height=1080,
        fps=30.0,
        duration_sec=5.0,
        has_video=True,
        has_audio=True,
        issues=[
            QualityCheckIssue(
                severity="error",
                code="SUBTITLE_INTEGRITY_ERROR",
                message="Subtitle integrity (misaligned): UNADJUSTED_SOURCE_TIMESTAMP_REUSE",
            )
        ],
    )
    (ws_dir / "reports").mkdir(parents=True, exist_ok=True)
    (ws_dir / "reports" / "qa_tl_bad.json").write_text(bad_qa.model_dump_json(indent=2), encoding="utf-8")

    # Simulate non-passing status after failed QA
    m = studio.workspaces.get_manifest(pid)
    readiness = studio.get_publish_readiness(pid)
    assert readiness["ready_for_publish"] is False
    assert readiness["recommended_status"] == "requires_review"
    m.project.status = readiness["recommended_status"]
    studio.workspaces.save_manifest(m)

    with pytest.raises(PermissionError, match="QA gate"):
        execute_publish(
            PublishRequest(
                project_id=pid,
                platform="telegram",
                mode=PublishingMode.APPROVED_PUBLISH,
                approval_confirmed=True,
                approval_reason="Verified by editor",
                artifact_path=str(ws_dir / "renders" / "out.mp4"),
                metadata={"text": "test", "chat_id": "@test"},
            ),
            ws_dir,
        )

    app = create_app(cfg)
    client = TestClient(app)
    r_gate = client.get(f"/api/projects/{pid}/publish-readiness")
    assert r_gate.status_code == 200
    assert r_gate.json()["ready_for_publish"] is False

    r_pub = client.post(
        f"/api/projects/{pid}/publish",
        json={
            "project_id": pid,
            "platform": "telegram",
            "mode": "approved_publish",
            "approval_confirmed": True,
            "approval_reason": "Try bypass",
            "artifact_path": str(ws_dir / "renders" / "out.mp4"),
            "metadata": {"text": "post"},
        },
    )
    assert r_pub.status_code == 409

    good_readiness = evaluate_publish_readiness(
        [
            QualityAssuranceReport(
                qa_id="qa_ok",
                project_id=pid,
                artifact_path=str(ws_dir / "renders" / "out.mp4"),
                passed=True,
                width=1920,
                height=1080,
                fps=30.0,
                duration_sec=5.0,
                has_video=True,
                has_audio=True,
            )
        ]
    )
    assert good_readiness["ready_for_publish"] is True
    assert good_readiness["recommended_status"] == "ready_for_publish"


def test_p0_monotonic_words_and_pause_localized_asr_alignment(tmp_path: Path) -> None:
    """Reproduce real-audio failure class: pause-smeared ASR tokens, far short-word distractors, and short-cue monotonic word clamping."""
    import wave

    wav_path = tmp_path / "paused_speech.wav"
    sr = 16000
    dur_sec = 14.0
    t = np.linspace(0.0, dur_sec, int(sr * dur_sec), endpoint=False)
    sig = 0.0015 * np.sin(2.0 * np.pi * 140.0 * t)
    # Two clear speech windows separated by a 4.0s pause [2.8, 6.8], plus burst at [9.5, 11.8]
    for s0, e0 in ((0.4, 2.8), (6.8, 8.6), (9.5, 11.8)):
        m = (t >= s0) & (t <= e0)
        sig[m] = 0.30 * np.sin(2.0 * np.pi * 230.0 * t[m])
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(sig, -0.95, 0.95) * 32767.0).astype(np.int16).tobytes())

    clip = VideoClipNode(
        clip_id="clip_00",
        source_path="dummy.mp4",
        source_start_sec=0.5,
        source_end_sec=14.5,
        timeline_start_sec=0.0,
        timeline_end_sec=14.0,
    )
    tl = RenderTimeline(
        timeline_id="tl_mono",
        project_id="proj_mono",
        title="Monotonic Pause Guard",
        clips=[clip],
        segment_mappings=[
            SegmentTimelineMapping(
                mapping_id="m0",
                source_segment_id="seg_0",
                clip_id="clip_00",
                source_start_sec=0.9,
                source_end_sec=12.3,
                timeline_start_sec=0.4,
                timeline_end_sec=11.8,
                language="en",
                text_excerpt="First burst words then second burst is fine",
                cue_ids=["c0", "c1", "c2", "c3"],
            )
        ],
        captions=[
            CaptionCue(
                cue_id="c0",
                clip_id="clip_00",
                source_segment_id="seg_0",
                source_start_sec=0.9,
                source_end_sec=1.5,
                start_sec=0.4,
                end_sec=1.0,
                text="First burst",
                words=[
                    {"word": "First", "start_sec": 0.4, "end_sec": 0.7},
                    {"word": "burst", "start_sec": 0.7, "end_sec": 1.0},
                ],
            ),
            CaptionCue(
                cue_id="c1",
                clip_id="clip_00",
                source_segment_id="seg_0",
                source_start_sec=1.55,
                source_end_sec=2.15,
                start_sec=1.05,
                end_sec=1.65,
                text="words then",
                words=[
                    {"word": "words", "start_sec": 1.05, "end_sec": 1.35},
                    {"word": "then", "start_sec": 1.35, "end_sec": 1.65},
                ],
            ),
            CaptionCue(
                cue_id="c2",
                clip_id="clip_00",
                source_segment_id="seg_0",
                source_start_sec=7.3,
                source_end_sec=8.9,
                start_sec=6.8,
                end_sec=8.4,
                text="second burst",
                words=[
                    {"word": "second", "start_sec": 6.8, "end_sec": 7.6},
                    {"word": "burst", "start_sec": 7.6, "end_sec": 8.4},
                ],
            ),
            CaptionCue(
                cue_id="c3",
                clip_id="clip_00",
                source_segment_id="seg_0",
                source_start_sec=10.0,
                source_end_sec=12.1,
                start_sec=9.5,
                end_sec=11.6,
                text="is fine",
                words=[
                    {"word": "is", "start_sec": 9.5, "end_sec": 10.2},
                    {"word": "fine", "start_sec": 10.2, "end_sec": 11.6},
                ],
            ),
        ],
    )

    # Inject noisy Whisper-like artifacts: smeared token across silence (0.05..6.2), and early false 2-letter distractor "is" at 2.3s!
    noisy_asr_words = [
        {"word": "First", "start_sec": 0.42, "end_sec": 0.85, "probability": 0.91},
        {"word": "is", "start_sec": 2.25, "end_sec": 2.35, "probability": 0.55},  # 2-char distractor right after c1
        {"word": "burst", "start_sec": 7.55, "end_sec": 8.35, "probability": 0.93},
        {"word": "fine", "start_sec": 10.25, "end_sec": 11.55, "probability": 0.96},
    ]
    aligned = align_captions_to_final_audio(tl, wav_path, final_asr_words=noisy_asr_words, use_asr_alignment=False)
    rep = validate_subtitle_integrity(aligned, final_audio_wav=wav_path, final_asr_words=noisy_asr_words)
    assert rep["passed"] is True, rep
    assert rep["out_of_order"] == []
    assert rep["misaligned"] == []
    # c1 ("words then") must stay inside the first speech island [0.4, 2.8] and NOT get dragged past 6.8s!
    assert aligned.captions[1].end_sec <= 3.1
    # Every word inside every cue must be strictly monotonic
    for cue in aligned.captions:
        prev_we = cue.start_sec - 0.001
        for w in cue.words:
            assert w["start_sec"] >= prev_we
            assert w["end_sec"] > w["start_sec"]
            prev_we = w["end_sec"]


