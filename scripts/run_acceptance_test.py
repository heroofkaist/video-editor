"""Execute all 15 Section-19 Acceptance Test milestones on a real 3-minute talking-head video."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_test_fixture import build_talking_head_fixture  # noqa: E402
from symposium_studio.config import StudioConfig  # noqa: E402
from symposium_studio.models.symposium import ResearchMode  # noqa: E402
from symposium_studio.studio import SymposiumStudio  # noqa: E402


def main() -> None:
    t_total = time.monotonic()
    cfg = StudioConfig.from_env(ROOT)
    fixture_mp4 = cfg.workspaces_root / "fixtures" / "electromagnetic_induction_talk_3min.mp4"

    t0 = time.monotonic()
    if not fixture_mp4.exists():
        print("[1/6] Generating real 3-minute spoken talking-head fixture...")
        build_talking_head_fixture(fixture_mp4, target_min_duration=180.0)
    else:
        print(f"[1/6] Using existing fixture: {fixture_mp4}")
    t_fixture = round(time.monotonic() - t0, 2)

    studio = SymposiumStudio(cfg)
    print("[2/6] Running complete Symposium Studio campaign pipeline...")
    t1 = time.monotonic()
    result = studio.run_campaign_pipeline(
        name="Electromagnetic Induction Explained",
        video_path=fixture_mp4,
        creator_niche="science_engineering",
        editing_style="educational_explainer",
        symposium_mode=ResearchMode.COMPREHENSIVE,
        telegram_channel="@symposium_physics_lab",
    )
    t_pipeline = round(time.monotonic() - t1, 2)

    pid = result["project_id"]
    # Also verify iterative natural-language instruction works end-to-end
    print("[3/6] Exercising iterative agent instruction ('Make the captions smaller')...")
    t2 = time.monotonic()
    iter_res = studio.apply_agent_instruction(pid, "Make the captions smaller")
    t_iter = round(time.monotonic() - t2, 2)

    # Refresh production report after iterative edit
    rep_res = studio.write_production_report(pid)

    yt_qa = result["renders"]["youtube"]["qa"]
    vertical = result["renders"]["vertical_variants"]
    print("[4/6] Validating all 15 Acceptance Test deliverables...")
    assert result["detected_language"] == "en", f"Expected 'en', got {result['detected_language']}"
    assert result["highlights_count"] >= 3
    assert yt_qa["passed"] is True
    assert 45.0 <= yt_qa["duration_sec"] <= 115.0, f"Unexpected YouTube edit duration: {yt_qa['duration_sec']}"
    assert yt_qa["caption_count"] >= 5
    assert yt_qa["overlay_count"] >= 1
    assert Path(result["renders"]["youtube"]["video_path"]).exists()
    assert Path(result["renders"]["youtube"]["otio_path"]).exists()
    assert Path(result["renders"]["youtube"]["thumbnail_path"]).exists()
    assert Path(result["animation"]["output_path"]).exists()
    assert len(vertical["youtube_shorts"]) >= 1 and vertical["youtube_shorts"][0]["qa"]["passed"]
    assert len(vertical["tiktok"]) >= 1 and vertical["tiktok"][0]["qa"]["passed"]
    assert len(vertical["instagram_reels"]) >= 1 and vertical["instagram_reels"][0]["qa"]["passed"]
    assert len(result["social"]["threads_posts"]) >= 6
    assert result["social"]["telegram_campaign"]["poll"] is not None
    assert Path(rep_res["markdown_path"]).exists()

    summary = {
        "project_id": pid,
        "workspace_dir": result["workspace_dir"],
        "timings_sec": {
            "fixture_generation": t_fixture,
            "full_campaign_pipeline": t_pipeline,
            "iterative_re_render": t_iter,
            "total": round(time.monotonic() - t_total, 2),
        },
        "detected_language": result["detected_language"],
        "highlights_count": result["highlights_count"],
        "animation_adapter_used": result["animation"]["adapter_used"],
        "animation_output": result["animation"]["output_path"],
        "youtube_video": result["renders"]["youtube"]["video_path"],
        "youtube_qa": iter_res["qa"],
        "shorts_video": vertical["youtube_shorts"][0]["video_path"],
        "tiktok_video": vertical["tiktok"][0]["video_path"],
        "reels_video": vertical["instagram_reels"][0]["video_path"],
        "threads_sample": result["social"]["threads_posts"][0]["text"],
        "telegram_channel": result["social"]["telegram_campaign"]["target_channel"],
        "report_markdown": rep_res["markdown_path"],
        "qa_frames": iter_res["qa"]["verification_frame_paths"],
    }
    print("[5/6] ACCEPTANCE TEST PASSED!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
