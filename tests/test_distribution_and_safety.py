"""Tests for Threads/Telegram repurposing, publishing authorization guardrails, and Growth Intelligence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from symposium_studio.config import StudioConfig
from symposium_studio.engines.distribution import (
    build_social_repurposing_bundle,
    build_telegram_campaign,
    execute_publish,
    generate_threads_posts,
)
from symposium_studio.engines.growth import GrowthIntelligenceEngine
from symposium_studio.engines.research import ResearchIntelligenceEngine
from symposium_studio.engines.symposium import SymposiumOrchestrator
from symposium_studio.models.social import PublishingMode, PublishRequest, ThreadsStyle
from symposium_studio.models.symposium import ResearchMode
from symposium_studio.models.timeline import RenderTimeline, VideoClipNode
from symposium_studio.storage import StudioDatabase
from tests.test_research_and_symposium import _sample_analysis


def test_threads_telegram_funnel_and_publish_guard(tmp_path: Path) -> None:
    db = StudioDatabase(tmp_path / "growth.db")
    analysis = _sample_analysis("proj_dist")
    tg = build_telegram_campaign("proj_dist", "Faraday Induction Lab", analysis, "@faraday_lab", db=db)

    assert len(tg.funnel_links) == 5
    assert any("?start=" in l.tagged_url for l in tg.funnel_links)
    assert tg.poll is not None and len(tg.poll.options) >= 3

    threads = generate_threads_posts(analysis, "Faraday induction", "@faraday_lab", max_chars=240)
    assert len(threads) == len(ThreadsStyle)
    for post in threads:
        assert post.char_count <= 240
        assert len(post.grounded_facts) >= 1

    tl = RenderTimeline(
        timeline_id="tl_1",
        project_id="proj_dist",
        title="Faraday Induction Lab",
        clips=[
            VideoClipNode(
                clip_id="c0",
                source_path="/tmp/fake.mp4",
                source_start_sec=0.0,
                source_end_sec=20.0,
                timeline_start_sec=0.0,
                timeline_end_sec=20.0,
                label="Hook: Stationary Magnet",
            )
        ],
    )
    bundle = build_social_repurposing_bundle("proj_dist", "Faraday Induction Lab", analysis, tl, tg)
    assert len(bundle.youtube.title_candidates) == 5
    assert bundle.youtube.chapters[0]["timestamp"] == "00:00"

    # Publishing safety: export_only & draft_prep succeed locally; approved_publish blocks without explicit reason/flag
    res_exp = execute_publish(
        PublishRequest(project_id="proj_dist", platform="threads", mode=PublishingMode.EXPORT_ONLY, artifact_path="", metadata={"text": threads[0].text}),
        tmp_path,
    )
    assert res_exp["external_posted"] is False

    with pytest.raises(PermissionError):
        execute_publish(
            PublishRequest(
                project_id="proj_dist",
                platform="telegram",
                mode=PublishingMode.APPROVED_PUBLISH,
                approval_confirmed=False,
                approval_reason="",
                artifact_path="",
                metadata={"text": tg.lead_post},
            ),
            tmp_path,
        )

    # Growth intelligence
    growth = GrowthIntelligenceEngine(db)
    growth.record_snapshot("proj_dist", "youtube", views_or_impressions=4200, likes_or_reactions=290, telegram_clicks=85, telegram_joins=34)
    summary = growth.summarize("proj_dist")
    assert summary["platform_summaries"][0]["click_to_join_pct"] == 40.0


def test_growth_cumulative_no_double_count_regression_and_telegram_sync(monkeypatch: Any, tmp_path: Path) -> None:
    db = StudioDatabase(tmp_path / "growth_cum.db")
    growth = GrowthIntelligenceEngine(db)

    # Record 3 cumulative snapshots for the same post -> MUST NOT sum to 1000+2400+5100=8500!
    growth.record_snapshot(
        "proj_c", "youtube", views_or_impressions=1000, likes_or_reactions=40,
        external_post_id="vid_1", measurement_mode="cumulative", recorded_at="2026-11-01T10:00:00+00:00",
    )
    growth.record_snapshot(
        "proj_c", "youtube", views_or_impressions=2400, likes_or_reactions=110,
        external_post_id="vid_1", measurement_mode="cumulative", recorded_at="2026-11-02T10:00:00+00:00",
    )
    growth.record_snapshot(
        "proj_c", "youtube", views_or_impressions=5100, likes_or_reactions=260,
        external_post_id="vid_1", measurement_mode="cumulative", recorded_at="2026-11-03T10:00:00+00:00",
    )
    # Also test incremental delta mode adds properly across snapshots
    growth.record_snapshot(
        "proj_c", "threads", views_or_impressions=120, likes_or_reactions=15,
        measurement_mode="delta", recorded_at="2026-11-01T11:00:00+00:00",
    )
    growth.record_snapshot(
        "proj_c", "threads", views_or_impressions=230, likes_or_reactions=25,
        measurement_mode="delta", recorded_at="2026-11-02T11:00:00+00:00",
    )

    s1 = growth.summarize("proj_c")
    by_plat = {p["platform"]: p for p in s1["platform_summaries"]}
    assert by_plat["youtube"]["total_views_or_impressions"] == 5100
    assert by_plat["youtube"]["latest_views_delta_since_prev"] == 2700
    assert by_plat["threads"]["total_views_or_impressions"] == 350

    # Counter regression warning when cumulative counter drops below previous
    growth.record_snapshot(
        "proj_c", "youtube", views_or_impressions=4900, likes_or_reactions=250,
        external_post_id="vid_1", measurement_mode="cumulative", recorded_at="2026-11-04T10:00:00+00:00",
    )
    s2 = growth.summarize("proj_c")
    assert any("regression" in w.lower() for w in s2["warnings"])

    # Mocked Telegram Bot API getChatMemberCount sync & bulk import
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot123:test_token")

    class _TgResp:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"ok": True, "result": 1842}

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _TgResp())
    snap = growth.sync_telegram_channel_stats("proj_c", "faraday_lab")
    assert snap.subscriber_or_member_count == 1842

    imported = growth.import_analytics_records(
        "proj_c",
        [{"platform": "instagram_reels", "views": 8200, "likes": 410, "external_post_id": "reel_9"}],
    )
    assert len(imported) == 1
    assert imported[0].views_or_impressions == 8200


def test_closed_loop_experiment_learning_and_target_tiers(tmp_path: Path) -> None:
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "learn.db")
    db = StudioDatabase(cfg.db_path)
    research = ResearchIntelligenceEngine(db)
    orch = SymposiumOrchestrator(cfg)
    growth = GrowthIntelligenceEngine(db)

    # Project 1: Baseline creator performance on YouTube (1,000 first-7d views)
    analysis_a = _sample_analysis("vid_a")
    report_a = research.generate_research_report("content_proposal", "science_engineering", ["faraday"])
    session_a = orch.run_symposium("vid_a", analysis_a, report_a, mode=ResearchMode.INEXPENSIVE)
    growth.record_experiment_from_production("vid_a", "Faraday Baseline Exp", "science_engineering", session_a, report_a)
    growth.record_snapshot(
        "vid_a",
        "youtube",
        views_or_impressions=700,
        first_7d_views=700,
        likes_or_reactions=30,
        external_post_id="yt_a",
        measurement_mode="cumulative",
        recorded_at="2026-11-01T10:00:00+00:00",
    )
    growth.record_snapshot(
        "vid_a",
        "youtube",
        views_or_impressions=1000,
        first_7d_views=1000,
        likes_or_reactions=55,
        authorized_ctr_pct=4.1,
        authorized_retention_pct=52.0,
        telegram_clicks=30,
        telegram_joins=10,
        external_post_id="yt_a",
        measurement_mode="cumulative",
        recorded_at="2026-11-02T10:00:00+00:00",
    )
    eval_a = {e.platform: e for e in growth.evaluate_project_experiments("vid_a")}
    assert eval_a["youtube"].first_7d_views == 1000
    assert eval_a["youtube"].growth_tier_reached == "initial_baseline_recorded"
    assert any("ctr modest" in h.lower() for h in eval_a["youtube"].learned_hypotheses)

    # Project 2: Same strategy achieves 1,650 first-7d views on YouTube (1.65x -> intermediate target)
    # and independently records YouTube Shorts (2,500 views, must NOT compare against 1,000 long-form baseline)
    analysis_b = _sample_analysis("vid_b")
    report_b = research.generate_research_report("content_proposal", "science_engineering", ["faraday"])
    session_b = orch.run_symposium("vid_b", analysis_b, report_b, mode=ResearchMode.INEXPENSIVE)
    growth.record_experiment_from_production("vid_b", "Faraday Spark Breakthrough", "science_engineering", session_b, report_b)
    growth.record_snapshot(
        "vid_b",
        "youtube",
        views_or_impressions=1650,
        first_7d_views=1650,
        likes_or_reactions=140,
        shares=25,
        authorized_ctr_pct=6.8,
        authorized_retention_pct=56.0,
        external_post_id="yt_b",
    )
    growth.record_snapshot(
        "vid_b",
        "youtube_shorts",
        views_or_impressions=2500,
        first_7d_views=2500,
        likes_or_reactions=210,
        external_post_id="sh_b",
    )
    eval_b = {e.platform: e for e in growth.evaluate_project_experiments("vid_b")}
    assert eval_b["youtube"].creator_baseline_median_views == 1000.0
    assert eval_b["youtube"].ratio_vs_creator_median == 1.65
    assert eval_b["youtube"].growth_tier_reached == "intermediate_target_1.5x"
    assert eval_b["youtube"].outcome_verdict == "outperformed"
    assert any("prediction confirmed" in h.lower() for h in eval_b["youtube"].learned_hypotheses)
    assert any("sample-size caution" in h.lower() for h in eval_b["youtube"].learned_hypotheses)
    # Separate platform baseline enforcement:
    assert eval_b["youtube_shorts"].creator_baseline_median_views is None
    assert eval_b["youtube_shorts"].growth_tier_reached == "initial_baseline_recorded"

    # Platform growth progress summary distinguishes YouTube vs Shorts & outlier caution
    summary_b = growth.summarize("vid_b")
    tp_by_plat = {p["platform"]: p for p in summary_b["target_progress"]}
    assert tp_by_plat["youtube"]["reached_1_5x_count"] == 1
    assert tp_by_plat["youtube"]["sample_confidence"] == "early_sample_avoid_outlier_bias"
    assert tp_by_plat["youtube_shorts"]["median_first_7d_views"] == 2500.0

    # Project 3: Next Symposium feeds prior experiments back into scoring & priors
    analysis_c = _sample_analysis("vid_c")
    session_c = orch.run_symposium(
        "vid_c",
        analysis_c,
        report_b,
        mode=ResearchMode.INEXPENSIVE,
        prior_experiments=db.list_experiments(limit=50),
    )
    assert session_c.prior_learnings_applied
    assert any(c.historical_lift_applied > 0.0 for c in session_c.candidate_evaluations)

    # Searchable FTS5 experiment ledger
    hits = db.search_all("Breakthrough")
    assert any(h["entity_type"] == "experiment" for h in hits)
