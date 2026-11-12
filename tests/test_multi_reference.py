from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from symposium_studio.api.app import create_app
from symposium_studio.cli import cli
from symposium_studio.config import StudioConfig
from symposium_studio.engines.analysis import apply_style_transfer_plan_to_timeline
from symposium_studio.engines.production import build_timeline_from_storyboard, render_timeline
from symposium_studio.engines.production.captions import validate_subtitle_integrity
from symposium_studio.models.media import (
    HighlightProposal,
    HighlightScoreBreakdown,
    MediaAnalysisBundle,
    MediaProbeResult,
    SilenceInterval,
    TopicCluster,
    TranscriptQualityReport,
    TranscriptResult,
    TranscriptSegment,
    WordTimestamp,
)
from symposium_studio.models.reference import ReferenceAnalysisStatus, SelectionMode
from symposium_studio.models.social import PublishingMode, QualityAssuranceReport, QualityCheckIssue
from symposium_studio.models.symposium import Storyboard, StoryboardBeat
from symposium_studio.studio import SymposiumStudio


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> Path:
    pcm = np.clip(samples * 32767.0, -32767, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def _make_tiny_distinct_mp4(out_path: Path, seed_idx: int) -> Path:
    colors = ["red", "blue", "green", "yellow", "purple", "cyan", "orange", "white", "magenta", "lime", "navy"]
    color = colors[seed_idx % len(colors)]
    freq = 220 + seed_idx * 45
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c={color}:s=160x120:d=0.8:r=15",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=16000:duration=0.8",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out_path),
        ],
        check=True,
    )
    return out_path


def _make_fast_reference_video(out_path: Path, duration_sec: float = 6.0) -> Path:
    """Create a 9:16 vertical fast-cut reference with frequent hard scene changes and HF SFX transients."""
    sr = 16000
    n = int(round(duration_sec * sr))
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.04 * np.sin(2.0 * np.pi * 240.0 * t)
    # Inject 8 sharp high-frequency click/transient spikes
    for k in range(8):
        center = 0.4 + k * 0.65
        if center >= duration_sec - 0.15:
            break
        i0 = int(center * sr)
        i1 = min(n, i0 + int(0.04 * sr))
        tt = np.arange(i1 - i0, dtype=np.float32) / sr
        env = np.exp(-tt * 85.0)
        sig[i0:i1] += 0.85 * env * np.sin(2.0 * np.pi * 4200.0 * tt)

    wav = _write_wav(out_path.with_suffix(".wav"), sig, sr=sr)
    # 6 hard color/scene blocks (~1.0s each) + bright caption-like lower-center bar
    vf = (
        "drawbox=x=0:y=0:w=iw:h=ih:color=navy@1:t=fill:enable='between(t,0,1.0)',"
        "drawbox=x=0:y=0:w=iw:h=ih:color=crimson@1:t=fill:enable='between(t,1.0,2.0)',"
        "drawbox=x=0:y=0:w=iw:h=ih:color=teal@1:t=fill:enable='between(t,2.0,3.0)',"
        "drawbox=x=0:y=0:w=iw:h=ih:color=orange@1:t=fill:enable='between(t,3.0,4.0)',"
        "drawbox=x=0:y=0:w=iw:h=ih:color=purple@1:t=fill:enable='between(t,4.0,5.0)',"
        "drawbox=x=0:y=0:w=iw:h=ih:color=green@1:t=fill:enable='between(t,5.0,6.0)',"
        "drawbox=x=40:y=470:w=280:h=55:color=white@0.95:t=fill"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s=360x640:d={duration_sec}:r=24",
            "-i", str(wav),
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out_path),
        ],
        check=True,
    )
    return out_path


def _make_calm_educational_reference(out_path: Path, duration_sec: float = 8.0) -> Path:
    """Create a 16:9 calm steady educational reference with low cut cadence and smooth audio."""
    sr = 16000
    n = int(round(duration_sec * sr))
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.08 * np.sin(2.0 * np.pi * 190.0 * t)
    wav = _write_wav(out_path.with_suffix(".wav"), sig, sr=sr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=0x1e293b:s=640x360:d={duration_sec}:r=24",
            "-i", str(wav),
            "-vf", "drawbox=x=160:y=80:w=320:h=180:color=0x38bdf8@0.5:t=4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out_path),
        ],
        check=True,
    )
    return out_path


def _make_pulsed_source_video(
    out_path: Path,
    duration_sec: float = 11.0,
    speech_windows: list[tuple[float, float]] | None = None,
) -> Path:
    windows = speech_windows or [(0.4, 2.8), (3.4, 6.2), (6.8, 10.2)]
    sr = 16000
    n = int(round(duration_sec * sr))
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.001 * np.sin(2.0 * np.pi * 60.0 * t)
    for ws, we in windows:
        i0 = max(0, min(n, int(round(ws * sr))))
        i1 = max(0, min(n, int(round(we * sr))))
        if i1 > i0:
            st = t[i0:i1] - ws
            env = 0.36 * (0.55 + 0.45 * np.sin(2.0 * np.pi * 4.2 * st))
            sig[i0:i1] += env * (0.7 * np.sin(2.0 * np.pi * 230.0 * st) + 0.3 * np.sin(2.0 * np.pi * 510.0 * st))
    wav = _write_wav(out_path.with_suffix(".wav"), sig, sr=sr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=24:d={duration_sec}",
            "-i", str(wav),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out_path),
        ],
        check=True,
    )
    return out_path


def test_a_multi_ref_ingest_dedup_and_artifact_cache_reuse(tmp_path: Path) -> None:
    """Test A: ingest >=10 local references, verify SHA256 dedup & cached StyleDNA reuse across projects."""
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db", whisper_model="base")
    studio = SymposiumStudio(cfg)
    p1 = studio.create_project("Ref Ingest One").project.project_id
    p2 = studio.create_project("Ref Ingest Two").project.project_id

    ref_files = [_make_tiny_distinct_mp4(tmp_path / f"ref_{i:02d}.mp4", i) for i in range(10)]
    res = studio.ingest_reference_files(p1, ref_files)
    assert res["ingested_count"] == 10
    assert len(studio.list_project_references(p1)) == 10

    # Re-ingest an exact duplicate inside the same project -> SHA256 cache hit & no extra project row created
    dup_copy = tmp_path / "dup_of_ref0.mp4"
    dup_copy.write_bytes(ref_files[0].read_bytes())
    dup_res = studio.ingest_reference_files(p1, [dup_copy])
    assert dup_res["cache_hits"] == 1
    assert len(studio.list_project_references(p1)) == 10

    # Analyze ref 0 in project 1, then reuse via SHA256 and personal library import in project 2
    ref0_id = res["references"][0]["reference_id"]
    an1 = studio.analyze_project_references(p1, reference_ids=[ref0_id], run_asr=False)
    assert an1["total_references"] == 10

    lib = studio.list_library_references()
    assert len(lib) >= 10

    # Same content ingested into p2 hits SHA256 StyleDNA artifact cache immediately
    reused = studio.ingest_reference_files(p2, [dup_copy])
    assert reused["cache_hits"] == 1
    assert reused["references"][0]["analysis_status"] == ReferenceAnalysisStatus.FULL_VIDEO_AUDIO.value
    assert reused["references"][0]["style_dna"] is not None

    an2 = studio.analyze_project_references(p2, run_asr=False)
    assert an2["cached_reused"] == 1

    # Import another reference explicitly from library by ID
    ref1_id = res["references"][1]["reference_id"]
    imp = studio.import_library_reference(p2, ref1_id)
    assert imp["reference"]["sha256"] == res["references"][1]["sha256"]
    assert len(studio.list_project_references(p2)) == 2


def test_b_url_ingest_honest_metadata_only_and_ranking_penalty(tmp_path: Path) -> None:
    """Test B: non-downloadable URLs stay metadata_only (no fake visual values) and lose to analyzed videos."""
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db", whisper_model="base")
    studio = SymposiumStudio(cfg)
    pid = studio.create_project("URL Honesty").project.project_id
    studio.ingest_video(pid, _make_pulsed_source_video(tmp_path / "src.mp4", duration_sec=8.0, speech_windows=[(0.5, 3.2), (4.0, 7.2)]))

    real_ref = _make_calm_educational_reference(tmp_path / "real_ref.mp4", duration_sec=6.0)
    studio.ingest_reference_files(pid, [real_ref])
    url_res = studio.ingest_reference_url(
        pid,
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Remote Unfetched URL",
        allow_network=False,
    )
    url_item = url_res["reference"]
    assert url_item["analysis_status"] == ReferenceAnalysisStatus.METADATA_ONLY.value
    assert url_item["style_dna"] is None
    assert url_item["edit_events_path"] is None
    lim_txt = " ".join(url_item["metadata"].get("honest_limitations", [])).lower()
    assert "metadata-only" in lim_txt and "not downloaded" in lim_txt

    studio.analyze_project_references(pid, run_asr=False)
    sel = studio.compare_and_select_references(pid, target_platform="youtube_long")
    ranked = sel["ranked_evaluations"]
    assert len(ranked) == 2

    url_eval = next(e for e in ranked if e["reference_id"] == url_item["reference_id"])
    real_eval = next(e for e in ranked if e["reference_id"] != url_item["reference_id"])
    assert url_eval["is_feasible"] is False
    assert url_eval["disqualification_reason"]
    assert real_eval["is_feasible"] is True
    assert real_eval["heuristic_compatibility_score"] > url_eval["heuristic_compatibility_score"]
    assert sel["selected_reference"]["reference_id"] == real_eval["reference_id"]


def test_c_real_cv_scenedetect_audio_analysis_artifacts(tmp_path: Path) -> None:
    """Test C: real CV/PySceneDetect/audio transient analyzer extracts distinct StyleDNA and artifacts."""
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db", whisper_model="base")
    studio = SymposiumStudio(cfg)
    pid = studio.create_project("Analyzer Test").project.project_id

    fast_mp4 = _make_fast_reference_video(tmp_path / "fast_ref.mp4", duration_sec=6.0)
    calm_mp4 = _make_calm_educational_reference(tmp_path / "calm_ref.mp4", duration_sec=8.0)
    studio.ingest_reference_files(pid, [fast_mp4, calm_mp4], titles=["Fast Cut Vlog", "Calm Edu"])

    res = studio.analyze_project_references(pid, run_asr=False)
    assert res["total_references"] == 2
    fast_item = next(r for r in res["references"] if r["title"] == "Fast Cut Vlog")
    calm_item = next(r for r in res["references"] if r["title"] == "Calm Edu")

    for item in (fast_item, calm_item):
        assert item["analysis_status"] == ReferenceAnalysisStatus.FULL_VIDEO_AUDIO.value
        assert item["style_dna"] is not None
        cs = Path(item["contact_sheet_path"])
        ev = Path(item["edit_events_path"])
        evid = Path(item["evidence_path"])
        assert cs.is_file() and cs.stat().st_size > 1500
        assert ev.is_file() and isinstance(json.loads(ev.read_text(encoding="utf-8")), list)
        assert evid.is_file()

    fast_events = json.loads(Path(fast_item["edit_events_path"]).read_text(encoding="utf-8"))
    assert len(fast_events) >= 3
    assert fast_item["style_dna"]["pacing"]["cuts_per_minute"] > calm_item["style_dna"]["pacing"]["cuts_per_minute"]
    assert fast_item["style_dna"]["audio"]["transient_sfx_per_min"] > calm_item["style_dna"]["audio"]["transient_sfx_per_min"]
    assert fast_item["style_dna"]["visual"]["aspect_ratio"] == "9:16"
    assert calm_item["style_dna"]["visual"]["aspect_ratio"] == "16:9"


def test_d_source_platform_aware_scoring_and_symposium_modes(tmp_path: Path) -> None:
    """Test D: 13 criteria respond to platform & source footage; Best Single, Hybrid, and Manual modes work."""
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db", whisper_model="base")
    studio = SymposiumStudio(cfg)
    pid = studio.create_project("Platform Modes").project.project_id
    studio.ingest_video(pid, _make_pulsed_source_video(tmp_path / "src.mp4", duration_sec=10.5))

    fast_mp4 = _make_fast_reference_video(tmp_path / "fast_9x16.mp4", duration_sec=6.0)
    calm_mp4 = _make_calm_educational_reference(tmp_path / "calm_16x9.mp4", duration_sec=8.0)
    studio.ingest_reference_files(pid, [fast_mp4, calm_mp4], titles=["Fast Vertical", "Calm 16x9"])
    studio.analyze_project_references(pid, run_asr=False)

    # Compare on youtube_long vs youtube_shorts
    long_sel = studio.compare_and_select_references(pid, target_platform="youtube_long", selection_mode=SelectionMode.BEST_SINGLE)
    short_sel = studio.compare_and_select_references(pid, target_platform="youtube_shorts", selection_mode=SelectionMode.BEST_SINGLE)

    long_ev_calm = next(e for e in long_sel["ranked_evaluations"] if e["title"] == "Calm 16x9")
    long_ev_fast = next(e for e in long_sel["ranked_evaluations"] if e["title"] == "Fast Vertical")
    short_ev_fast = next(e for e in short_sel["ranked_evaluations"] if e["title"] == "Fast Vertical")
    assert len(long_ev_calm["criteria_breakdown"]) == 13
    assert "not a forecast of audience views" in long_ev_calm["score_disclaimer"].lower()

    calm_aspect_long = next(c["score"] for c in long_ev_calm["criteria_breakdown"] if c["criterion"] == "target_platform_fit")
    fast_aspect_long = next(c["score"] for c in long_ev_fast["criteria_breakdown"] if c["criterion"] == "target_platform_fit")
    fast_aspect_short = next(c["score"] for c in short_ev_fast["criteria_breakdown"] if c["criterion"] == "target_platform_fit")
    assert calm_aspect_long > fast_aspect_long
    assert fast_aspect_short > fast_aspect_long

    # AI Symposium Reference Conference has 10 specialist statements + Moderator warnings
    conf = short_sel["conference"]
    assert len(conf["agent_statements"]) == 10
    assert len(conf["moderator_warnings"]) >= 1

    # Mode B: Intelligent Hybrid blends dimensions across references
    hyb_sel = studio.compare_and_select_references(pid, target_platform="youtube_shorts", selection_mode=SelectionMode.INTELLIGENT_HYBRID)
    hyb_guide = hyb_sel["conference"]["hybrid_guide"]
    assert hyb_guide is not None
    assert set(hyb_guide["dimension_sources"].keys()) == {
        "opening_hook", "pacing_rhythm", "subtitle_typography", "motion_graphics", "audio_sfx"
    }

    # Mode C: Manual Override explicitly chooses Calm 16x9 even on Shorts
    calm_ref_id = long_ev_calm["reference_id"]
    man_sel = studio.compare_and_select_references(
        pid,
        target_platform="youtube_shorts",
        selection_mode=SelectionMode.MANUAL_OVERRIDE,
        manual_reference_id=calm_ref_id,
    )
    assert man_sel["conference"]["selection_mode"] == SelectionMode.MANUAL_OVERRIDE.value
    assert man_sel["selected_reference"]["reference_id"] == calm_ref_id


def test_e_end_to_end_style_transfer_render_api_and_cli(monkeypatch: Any, tmp_path: Path) -> None:
    """Test E: end-to-end API upload -> analyze -> compare -> render -> similarity report + CLI check."""
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db", whisper_model="base")
    app = create_app(cfg)
    client = TestClient(app)

    pid = client.post("/api/projects", json={"name": "E2E Style Demo", "creator_niche": "science_engineering"}).json()["project"]["project_id"]
    src_mp4 = _make_pulsed_source_video(tmp_path / "src_talk.mp4", duration_sec=11.0)
    with src_mp4.open("rb") as f:
        up_src = client.post(f"/api/projects/{pid}/upload", files={"file": ("src_talk.mp4", f, "video/mp4")})
    assert up_src.status_code == 200

    # Multi-file reference upload via API
    r_fast = _make_fast_reference_video(tmp_path / "fast.mp4", duration_sec=6.0)
    r_calm = _make_calm_educational_reference(tmp_path / "calm.mp4", duration_sec=8.0)
    with r_fast.open("rb") as f1, r_calm.open("rb") as f2:
        up_refs = client.post(
            f"/api/projects/{pid}/references/upload",
            files=[
                ("files", ("fast_ref.mp4", f1, "video/mp4")),
                ("files", ("calm_ref.mp4", f2, "video/mp4")),
            ],
        )
    assert up_refs.status_code == 200
    assert up_refs.json()["ingested_count"] == 2

    an_resp = client.post(f"/api/projects/{pid}/references/analyze", json={"run_asr": False})
    assert an_resp.status_code == 200
    ref0_id = an_resp.json()["references"][0]["reference_id"]

    # Contact sheet & library API endpoints
    cs_resp = client.get(f"/api/references/{ref0_id}/contact-sheet")
    assert cs_resp.status_code == 200
    assert cs_resp.headers["content-type"].startswith("image/")
    assert len(cs_resp.content) > 1000
    lib_resp = client.get("/api/references")
    assert lib_resp.status_code == 200 and len(lib_resp.json()) >= 2

    cmp_resp = client.post(
        f"/api/projects/{pid}/references/compare",
        json={"target_platform": "youtube_long", "selection_mode": "intelligent_hybrid"},
    )
    assert cmp_resp.status_code == 200
    assert cmp_resp.json()["conference"]["selection_mode"] == "intelligent_hybrid"

    ren_resp = client.post(
        f"/api/projects/{pid}/references/render",
        json={"target_platform": "youtube_long", "include_shorts": True},
    )
    assert ren_resp.status_code == 200
    rdata = ren_resp.json()
    yt_out = Path(rdata["youtube"]["video_path"])
    assert yt_out.is_file() and yt_out.stat().st_size > 10_000
    sim = rdata["style_similarity"]
    assert 0.0 <= sim["overall_style_adherence_score"] <= 1.0
    assert len(sim["metrics"]) >= 2

    # CLI contract verification against the same workspace DB
    monkeypatch.setenv("SYMPOSIUM_ROOT", str(tmp_path))
    monkeypatch.setenv("SYMPOSIUM_WORKSPACES", str(tmp_path / "ws"))
    monkeypatch.setenv("SYMPOSIUM_DB", str(tmp_path / "ws" / "s.db"))
    runner = CliRunner()
    res_lib = runner.invoke(cli, ["reference", "library"])
    assert res_lib.exit_code == 0, res_lib.output
    assert len(json.loads(res_lib.output)) >= 2

    res_adj = runner.invoke(cli, ["reference", "adjust", pid, "--increase-pacing", "--disable-memes"])
    assert res_adj.exit_code == 0, res_adj.output
    adj_json = json.loads(res_adj.output)
    assert adj_json["style_transfer_plan"]["parameters"]["enable_memes"] is False
    assert adj_json["style_transfer_plan"]["parameters"]["pacing_multiplier"] > 1.0


def test_f_multilingual_kk_ru_en_subtitle_sync_under_style_transfer(tmp_path: Path) -> None:
    """Test F: multilingual kk/ru/en captions keep <=2w style transfer wrapping without breaking output sync."""
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db", whisper_model="base")
    studio = SymposiumStudio(cfg)
    pid = studio.create_project("Multilingual Sync").project.project_id
    src_mp4 = _make_pulsed_source_video(tmp_path / "kk_ru_en.mp4", duration_sec=9.6, speech_windows=[(0.4, 3.0), (3.6, 6.2), (6.6, 9.2)])
    probe = studio.workspaces.get_manifest(pid)
    studio.ingest_video(pid, src_mp4)
    ws_dir = Path(studio.workspaces.get_manifest(pid).project.workspace_dir)

    from symposium_studio.engines.analysis.probe import probe_media

    mp = probe_media(src_mp4)
    segs = [
        TranscriptSegment(
            segment_id="seg_0",
            start_sec=0.4,
            end_sec=3.0,
            text="Сәлем зертхана бастаймыз",
            language="kk",
            confidence=0.92,
            words=[
                WordTimestamp(word="Сәлем", start_sec=0.45, end_sec=1.15, probability=0.94),
                WordTimestamp(word="зертхана", start_sec=1.25, end_sec=2.05, probability=0.91),
                WordTimestamp(word="бастаймыз", start_sec=2.15, end_sec=2.90, probability=0.92),
            ],
        ),
        TranscriptSegment(
            segment_id="seg_1",
            start_sec=3.6,
            end_sec=6.2,
            text="Теперь проверим датчик",
            language="ru",
            confidence=0.94,
            words=[
                WordTimestamp(word="Теперь", start_sec=3.65, end_sec=4.35, probability=0.95),
                WordTimestamp(word="проверим", start_sec=4.45, end_sec=5.25, probability=0.94),
                WordTimestamp(word="датчик", start_sec=5.35, end_sec=6.10, probability=0.93),
            ],
        ),
        TranscriptSegment(
            segment_id="seg_2",
            start_sec=6.6,
            end_sec=9.2,
            text="voltage gain verified",
            language="en",
            confidence=0.95,
            words=[
                WordTimestamp(word="voltage", start_sec=6.65, end_sec=7.35, probability=0.96),
                WordTimestamp(word="gain", start_sec=7.45, end_sec=8.15, probability=0.95),
                WordTimestamp(word="verified", start_sec=8.25, end_sec=9.10, probability=0.95),
            ],
        ),
    ]
    analysis = MediaAnalysisBundle(
        project_id=pid,
        probe=mp,
        transcript=TranscriptResult(
            engine="whisper.cpp",
            model="base",
            detected_language="kk",
            language_probability=0.9,
            languages_observed=["kk", "ru", "en"],
            duration_sec=mp.duration_sec,
            segments=segs,
            quality_report=TranscriptQualityReport(
                total_segments=3,
                high_confidence_count=3,
                code_switching_detected=True,
                languages_observed=["kk", "ru", "en"],
            ),
        ),
        silences=[SilenceInterval(start_sec=3.0, end_sec=3.6, duration_sec=0.6)],
        topics=[TopicCluster(topic_id="t0", title="Circuit test", start_sec=0.4, end_sec=9.2, keywords=["voltage", "gain"])],
        highlights=[
            HighlightProposal(
                highlight_id="hl_0",
                start_sec=0.4,
                end_sec=9.2,
                duration_sec=8.8,
                role="core_explanation",
                title="Full Demo",
                transcript_excerpt="Сәлем зертхана бастаймыз",
                scores=HighlightScoreBreakdown(
                    information_density=0.8, novelty=0.7, story_relevance=0.9,
                    audience_curiosity=0.8, educational_value=0.9, emotional_intensity=0.6,
                    visual_opportunity=0.7, context_dependency=0.2, composite_score=0.84,
                ),
            )
        ],
    )
    sb = Storyboard(
        storyboard_id="sb_multi",
        project_id=pid,
        target_format="youtube_shorts",
        selected_strategy="Multilingual Reordered Style Transfer",
        target_duration_sec=8.1,
        beats=[
            StoryboardBeat(
                beat_index=0,
                role="hook",
                source_start_sec=3.55,
                source_end_sec=6.25,
                target_duration_sec=2.7,
                visual_treatment="face_tracked",
                caption_directive="kinetic_caption_pop",
                audio_directive="speech_loudness_polish",
                transcript_excerpt="Теперь проверим датчик",
            ),
            StoryboardBeat(
                beat_index=1,
                role="core",
                source_start_sec=0.35,
                source_end_sec=3.05,
                target_duration_sec=2.7,
                visual_treatment="face_tracked",
                caption_directive="kinetic_caption_pop",
                audio_directive="speech_loudness_polish",
                transcript_excerpt="Сәлем зертхана бастаймыз",
            ),
            StoryboardBeat(
                beat_index=2,
                role="payoff",
                source_start_sec=6.55,
                source_end_sec=9.25,
                target_duration_sec=2.7,
                visual_treatment="face_tracked",
                caption_directive="kinetic_caption_pop",
                audio_directive="speech_loudness_polish",
                transcript_excerpt="voltage gain verified",
            ),
        ],
    )
    ref_mp4 = _make_fast_reference_video(tmp_path / "fast_ref.mp4", duration_sec=6.0)
    studio.ingest_reference_files(pid, [ref_mp4])
    studio.analyze_project_references(pid, run_asr=False)
    # Persist analysis so compare uses it
    bundle_p = ws_dir / "transcripts" / "analysis_bundle.json"
    bundle_p.parent.mkdir(parents=True, exist_ok=True)
    bundle_p.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    sel = studio.compare_and_select_references(pid, target_platform="youtube_shorts")

    tl = build_timeline_from_storyboard(project_id=pid, title="Multi", analysis=analysis, storyboard=sb, style_name="zoomer_vlog")
    from symposium_studio.models.reference import StyleTransferPlan

    plan = StyleTransferPlan.model_validate(sel["style_transfer_plan"])
    plan.parameters.subtitle_max_words = 2
    styled_tl = apply_style_transfer_plan_to_timeline(tl, plan)
    assert styled_tl.caption_settings.max_words_per_line == 2

    out_mp4 = ws_dir / "renders" / "multi_sync.mp4"
    rf = render_timeline(styled_tl, out_mp4, ws_dir)
    rendered_tl = rf["rendered_timeline"]
    integrity = rendered_tl.subtitle_integrity or validate_subtitle_integrity(rendered_tl)
    assert integrity["passed"] is True, integrity
    srt_text = Path(rf["srt_path"]).read_text(encoding="utf-8")
    for w in ("Теперь", "проверим", "Сәлем", "зертхана", "voltage", "gain"):
        assert w in srt_text
    # Ensure reordered beat timestamps are monotonic output-relative (not leaked source order)
    starts = [c.start_sec for c in rendered_tl.captions]
    assert starts == sorted(starts)


def test_g_interactive_controls_modify_plan_and_timeline(tmp_path: Path) -> None:
    """Test G: interactive adjustment controls cumulatively alter StyleTransferPlan and concrete RenderTimeline."""
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db", whisper_model="base")
    studio = SymposiumStudio(cfg)
    pid = studio.create_project("Interactive Controls").project.project_id
    studio.ingest_video(pid, _make_pulsed_source_video(tmp_path / "src.mp4", duration_sec=8.0, speech_windows=[(0.5, 3.0), (3.8, 7.2)]))
    studio.ingest_reference_files(pid, [_make_fast_reference_video(tmp_path / "fast.mp4", duration_sec=6.0)])
    studio.analyze_project_references(pid, run_asr=False)
    studio.compare_and_select_references(pid, target_platform="youtube_long")

    # Stepwise cumulative controls
    r1 = studio.adjust_style_transfer_plan(pid, {"disable_memes": True})
    assert r1["style_transfer_plan"]["parameters"]["enable_memes"] is False

    r2 = studio.adjust_style_transfer_plan(pid, {"subtitle_minimal": True})
    p2 = r2["style_transfer_plan"]["parameters"]
    assert p2["enable_memes"] is False  # retained!
    assert p2["subtitle_max_words"] == 2
    assert p2["subtitle_scale_factor"] < 1.0

    r3 = studio.adjust_style_transfer_plan(pid, {"increase_pacing": True})
    p3 = r3["style_transfer_plan"]["parameters"]
    assert p3["pacing_multiplier"] > 1.15
    assert p3["target_shot_duration_sec"] < r1["style_transfer_plan"]["parameters"]["target_shot_duration_sec"]

    r4 = studio.adjust_style_transfer_plan(pid, {"keep_structure_apply_rhythm_only": True})
    assert r4["style_transfer_plan"]["parameters"]["keep_structure_apply_rhythm_only"] is True

    from symposium_studio.models.reference import StyleTransferPlan
    from symposium_studio.models.timeline import OverlayNode, RenderTimeline, VideoClipNode

    plan_obj = StyleTransferPlan.model_validate(r4["style_transfer_plan"])
    dummy_tl = RenderTimeline(
        timeline_id="tl_ctrl",
        project_id=pid,
        title="Controls TL",
        clips=[
            VideoClipNode(clip_id="c0", source_path="s.mp4", source_start_sec=0.0, source_end_sec=2.0, timeline_start_sec=0.0, timeline_end_sec=2.0),
            VideoClipNode(clip_id="c1", source_path="s.mp4", source_start_sec=2.0, source_end_sec=4.0, timeline_start_sec=2.0, timeline_end_sec=4.0),
        ],
        overlays=[
            OverlayNode(overlay_id="ov_meme", label="Zoomer Meme Reaction", asset_path="meme_card.mov", timeline_start_sec=0.5, timeline_end_sec=2.0),
            OverlayNode(overlay_id="ov_edu", label="Diagram Callout", asset_path="edu_card.mov", timeline_start_sec=2.1, timeline_end_sec=3.8),
        ],
    )
    out_tl = apply_style_transfer_plan_to_timeline(dummy_tl, plan_obj)
    assert len(out_tl.overlays) == 1 and out_tl.overlays[0].overlay_id == "ov_edu"
    assert out_tl.caption_settings.max_words_per_line == 2
    assert any(c.zoom_scale > 1.05 for c in out_tl.clips)


def test_h_qa_gate_and_publish_readiness_after_reference_render(tmp_path: Path) -> None:
    """Test H: strict publish gate permits ready_for_publish when QA passes and blocks 409 when QA fails."""
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db", whisper_model="base")
    app = create_app(cfg)
    client = TestClient(app)
    studio: SymposiumStudio = app.state.studio

    pid = client.post("/api/projects", json={"name": "Ref QA Gate"}).json()["project"]["project_id"]
    src_mp4 = _make_pulsed_source_video(tmp_path / "src.mp4", duration_sec=8.5, speech_windows=[(0.5, 4.0), (4.6, 8.0)])
    studio.ingest_video(pid, src_mp4)
    studio.ingest_reference_files(pid, [_make_calm_educational_reference(tmp_path / "calm.mp4", duration_sec=5.0)])
    studio.analyze_project_references(pid, run_asr=False)
    studio.compare_and_select_references(pid, target_platform="youtube_long")
    ren = studio.render_with_reference_style(pid, target_platform="youtube_long", include_shorts=False)
    assert ren["publish_readiness"]["ready_for_publish"] is True
    assert ren["project_status"] == "ready_for_publish"

    readiness_ok = client.get(f"/api/projects/{pid}/publish-readiness")
    assert readiness_ok.status_code == 200 and readiness_ok.json()["ready_for_publish"] is True

    # Write a failing critical QA report into workspace/reports to simulate failed final validation
    ws_dir = Path(studio.workspaces.get_manifest(pid).project.workspace_dir)
    bad_qa = QualityAssuranceReport(
        qa_id="qa_critical_fail",
        project_id=pid,
        artifact_path=ren["youtube"]["video_path"],
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
                message="Unadjusted source timestamps detected",
            )
        ],
    )
    (ws_dir / "reports" / "qa_critical_fail.json").write_text(bad_qa.model_dump_json(indent=2), encoding="utf-8")

    readiness_bad = client.get(f"/api/projects/{pid}/publish-readiness")
    assert readiness_bad.status_code == 200
    assert readiness_bad.json()["ready_for_publish"] is False
    assert readiness_bad.json()["recommended_status"] == "requires_review"

    pub_blocked = client.post(
        f"/api/projects/{pid}/publish",
        json={
            "project_id": pid,
            "platform": "youtube",
            "mode": PublishingMode.APPROVED_PUBLISH.value,
            "approval_confirmed": True,
            "approval_reason": "test_qa_gate",
            "artifact_path": ren["youtube"]["video_path"],
        },
    )
    assert pub_blocked.status_code == 409
