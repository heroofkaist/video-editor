"""Tests for Threads/Telegram repurposing, publishing authorization guardrails, and Growth Intelligence."""

from __future__ import annotations

from pathlib import Path

import pytest

from symposium_studio.engines.distribution import (
    build_social_repurposing_bundle,
    build_telegram_campaign,
    execute_publish,
    generate_threads_posts,
)
from symposium_studio.engines.growth import GrowthIntelligenceEngine
from symposium_studio.models.social import PublishingMode, PublishRequest, ThreadsStyle
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
