"""Tests for cross-niche Research Intelligence, FTS5 search, and AI Symposium deliberation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from symposium_studio.config import StudioConfig
from symposium_studio.engines.research import ResearchIntelligenceEngine, YouTubeOfficialConnector, import_explicit_item
from symposium_studio.engines.research.engine import recompute_platform_relative_scores
from symposium_studio.engines.symposium import AGENT_ROSTER, SymposiumOrchestrator, active_agents_for_mode
from symposium_studio.models.media import (
    HighlightProposal,
    HighlightScoreBreakdown,
    MediaAnalysisBundle,
    MediaProbeResult,
    TopicCluster,
    TranscriptResult,
    TranscriptSegment,
)
from symposium_studio.models.research import ContentCategory, ResearchPlatform
from symposium_studio.models.symposium import ResearchMode
from symposium_studio.storage import StudioDatabase


def _sample_analysis(proj_id: str) -> MediaAnalysisBundle:
    return MediaAnalysisBundle(
        project_id=proj_id,
        probe=MediaProbeResult(
            path="/tmp/fake.mp4",
            sha256="abc",
            size_bytes=1024,
            duration_sec=180.0,
            width=1920,
            height=1080,
            fps=30.0,
            video_codec="h264",
            audio_codec="aac",
        ),
        transcript=TranscriptResult(
            engine="mlx-whisper",
            model="mlx-community/whisper-small-mlx",
            detected_language="en",
            language_probability=0.96,
            languages_observed=["en"],
            duration_sec=180.0,
            segments=[
                TranscriptSegment(
                    segment_id="seg_0000",
                    start_sec=0.0,
                    end_sec=18.0,
                    text="Why does a magnet sitting inside a coil generate zero current until it moves?",
                    confidence=0.94,
                ),
                TranscriptSegment(
                    segment_id="seg_0001",
                    start_sec=20.0,
                    end_sec=65.0,
                    text="Faraday's law states that induced EMF equals minus N times the rate of change of magnetic flux.",
                    confidence=0.92,
                ),
                TranscriptSegment(
                    segment_id="seg_0002",
                    start_sec=70.0,
                    end_sec=120.0,
                    text="When I first wound the coil on my bench I measured zero volts because flux was constant.",
                    confidence=0.91,
                ),
            ],
        ),
        topics=[
            TopicCluster(
                topic_id="topic_00",
                title="Magnetic flux change and Faraday law",
                start_sec=0.0,
                end_sec=120.0,
                keywords=["magnet", "flux", "coil", "faraday", "voltage"],
                is_technical=True,
                domain="physics",
            )
        ],
        highlights=[
            HighlightProposal(
                highlight_id="hl_00",
                start_sec=0.0,
                end_sec=22.0,
                duration_sec=22.0,
                role="hook",
                title="Stationary magnet mystery",
                transcript_excerpt="Why does a magnet sitting inside a coil generate zero current until it moves?",
                scores=HighlightScoreBreakdown(
                    information_density=0.85,
                    novelty=0.80,
                    story_relevance=0.90,
                    audience_curiosity=0.92,
                    educational_value=0.88,
                    emotional_intensity=0.70,
                    visual_opportunity=0.85,
                    context_dependency=0.95,
                    composite_score=0.86,
                ),
                visual_opportunity_type="em_induction_faraday",
                suggested_animation_domain="physics",
            ),
            HighlightProposal(
                highlight_id="hl_01",
                start_sec=22.0,
                end_sec=65.0,
                duration_sec=43.0,
                role="core_explanation",
                title="Faraday's law explanation",
                transcript_excerpt="Faraday's law states that induced EMF equals minus N times the rate of change of magnetic flux.",
                scores=HighlightScoreBreakdown(
                    information_density=0.90,
                    novelty=0.82,
                    story_relevance=0.85,
                    audience_curiosity=0.75,
                    educational_value=0.96,
                    emotional_intensity=0.65,
                    visual_opportunity=0.94,
                    context_dependency=0.88,
                    composite_score=0.87,
                ),
                visual_opportunity_type="em_induction_faraday",
                suggested_animation_domain="physics",
            ),
        ],
        recommended_crop_center_x=0.38,
    )


def test_cross_niche_research_and_fts(tmp_path: Path) -> None:
    db = StudioDatabase(tmp_path / "test.db")
    research = ResearchIntelligenceEngine(db)

    all_items = research.search_patterns(limit=50)
    categories = {i.category.value for i in all_items}
    platforms = {i.platform.value for i in all_items}
    # Must span broad non-STEM categories as well as tech/science
    assert {"gaming", "comedy", "documentaries", "lifestyle", "sports", "entertainment"}.issubset(categories)
    assert {"youtube", "youtube_shorts", "tiktok", "instagram_reels", "threads"}.issubset(platforms)

    # Verify private metrics are never fabricated on structural studies
    for item in all_items:
        assert item.metrics.authorized_ctr is None
        assert item.metrics.authorized_avg_view_pct is None

    # Explicit import + within-platform scoring
    imported = import_explicit_item(
        platform=ResearchPlatform.YOUTUBE,
        creator="Creator Lab",
        original_url="https://www.youtube.com/watch?v=example123",
        title="Coil Benchmark Video",
        category=ContentCategory.ENGINEERING,
        topic="electromagnetic induction",
        opening_hook_structure="Shows spark at 0.5s",
        editing_style="fast visual proof",
        narrative_structure="demo -> law -> fix",
        views=125000,
        likes=6200,
    )
    saved = research.add_item(imported)
    assert 10.0 <= saved.platform_relative_score <= 99.0

    # FTS search finds gaming pattern and imported item
    hits = db.search_all("gaming")
    assert any(h["entity_type"] == "viral_item" for h in hits)

    report = research.generate_research_report("content_proposal", "science_engineering", ["induction", "coil"])
    assert len(report.cross_niche_transfers) >= 3
    assert len(report.source_item_ids) >= 3


def test_symposium_modes_and_traceability(tmp_path: Path) -> None:
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db")
    db = StudioDatabase(cfg.db_path)
    research = ResearchIntelligenceEngine(db)
    report = research.generate_research_report("content_proposal", "science_engineering", ["flux"])
    orch = SymposiumOrchestrator(cfg)
    analysis = _sample_analysis("proj_test")

    inexp = active_agents_for_mode(ResearchMode.INEXPENSIVE)
    std = active_agents_for_mode(ResearchMode.STANDARD)
    comp = active_agents_for_mode(ResearchMode.COMPREHENSIVE)
    assert len(inexp) < len(std) < len(comp) == len(AGENT_ROSTER) == 18

    session = orch.run_symposium("proj_test", analysis, report, mode=ResearchMode.COMPREHENSIVE)
    assert len(session.participating_agents) == 18
    assert len(session.challenges) >= 2
    assert len(session.decisions) >= 3
    assert any(b.animation_template == "em_induction_faraday" for b in session.storyboard.beats)


def test_unscored_benchmarks_zero_and_mocked_rss_audience_normalization(monkeypatch: Any, tmp_path: Path) -> None:
    db = StudioDatabase(tmp_path / "rss_test.db")
    research = ResearchIntelligenceEngine(db)
    seeded = research.search_patterns(limit=50)
    assert seeded
    for item in seeded:
        assert item.platform_relative_score == 0.0
        assert item.score_basis == "unscored_structural_benchmark"
        assert item.metrics.views is None

    sample_atom = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns:media="http://search.yahoo.com/mrss/"
      xmlns="http://www.w3.org/2005/Atom">
  <title>Physics Demo Channel</title>
  <entry>
    <yt:videoId>vid_baseline_1</yt:videoId>
    <title>Baseline Circuit Measurement</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=vid_baseline_1"/>
    <published>2026-10-01T12:00:00+00:00</published>
    <media:group>
      <media:description>Circuit lab notes.</media:description>
      <media:community>
        <media:starRating count="400"/>
        <media:statistics views="10000"/>
      </media:community>
    </media:group>
  </entry>
  <entry>
    <yt:videoId>vid_baseline_2</yt:videoId>
    <title>Capacitor Discharge Test</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=vid_baseline_2"/>
    <published>2026-10-05T12:00:00+00:00</published>
    <media:group>
      <media:description>Quick bench comparison.</media:description>
      <media:community>
        <media:starRating count="500"/>
        <media:statistics views="12000"/>
      </media:community>
    </media:group>
  </entry>
  <entry>
    <yt:videoId>vid_outlier</yt:videoId>
    <title>Why Did Zero Current Flow in This Closed Loop?</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=vid_outlier"/>
    <published>2026-10-10T12:00:00+00:00</published>
    <media:group>
      <media:description>Unexpected counter-EMF demonstration.</media:description>
      <media:community>
        <media:starRating count="9500"/>
        <media:statistics views="120000"/>
      </media:community>
    </media:group>
  </entry>
</feed>
"""

    class _FakeResp:
        status_code = 200
        text = sample_atom

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResp())
    conn = YouTubeOfficialConnector()
    fetched = conn.fetch_channel_rss("UCfake", limit=15, category=ContentCategory.SCIENCE)
    assert len(fetched) == 3
    by_id = {x.item_id: x for x in fetched}
    assert by_id["yt_rss_vid_outlier"].metrics.channel_baseline_views == 12000
    assert by_id["yt_rss_vid_outlier"].metrics.audience_normalized_ratio == 10.0

    scored = recompute_platform_relative_scores(fetched + seeded)
    scored_outlier = next(x for x in scored if x.item_id == "yt_rss_vid_outlier")
    scored_seed = next(x for x in scored if x.score_basis == "unscored_structural_benchmark")
    assert 10.0 <= scored_outlier.platform_relative_score <= 99.0
    assert scored_seed.platform_relative_score == 0.0


def test_symposium_asr_hazard_swap_and_mocked_ollama_grounding(monkeypatch: Any, tmp_path: Path) -> None:
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db")
    db = StudioDatabase(cfg.db_path)
    research = ResearchIntelligenceEngine(db)
    report = research.generate_research_report("content_proposal", "science_engineering", ["flux"])

    analysis = _sample_analysis("proj_asr")
    analysis.probe.duration_sec = 140.0
    # Add an extra highlight covering an uncertain ASR interval and a clean high-scoring backup highlight
    analysis.transcript.uncertain_segment_ids = ["seg_1"]
    analysis.transcript.segments[1].low_confidence = True
    analysis.highlights.append(
        HighlightProposal(
            highlight_id="hl_clean_backup",
            start_sec=80.0,
            end_sec=112.0,
            duration_sec=32.0,
            role="payoff",
            title="Clean Empirical Payoff",
            transcript_excerpt="When moving the magnet fast, the galvanometer pegs cleanly across the scale.",
            scores=HighlightScoreBreakdown(
                information_density=0.91,
                novelty=0.80,
                story_relevance=0.88,
                audience_curiosity=0.84,
                educational_value=0.92,
                emotional_intensity=0.72,
                visual_opportunity=0.80,
                context_dependency=0.86,
                composite_score=0.85,
            ),
        )
    )

    class _MockResp:
        def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("http error")

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _MockResp(200, {"models": [{"name": "qwen2.5:7b"}]}) if "/api/tags" in url else _MockResp(404, {}),
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: _MockResp(
            200,
            {
                "message": {
                    "content": json.dumps(
                        {
                            "summary": "Grounded LLM proposal prioritizing visual hook before derivation.",
                            "recommendations": ["Trim opening silence and open with coil spark proof."],
                            "cited_ids": ["hl_00", "hallucinated_id_999"],
                            "confidence": 0.93,
                        }
                    )
                }
            },
        ),
    )

    orch = SymposiumOrchestrator(cfg)
    session = orch.run_symposium(
        "proj_asr",
        analysis,
        report,
        mode=ResearchMode.STANDARD,
        target_min_sec=55.0,
        target_max_sec=70.0,
    )
    assert session.backend_used == "ollama:qwen2.5:7b"
    assert session.llm_calls_count > 0
    assert any(c.issue_type == "asr_confidence_hazard" for c in session.challenges)
    assert len(session.debate_turns) >= len(session.participating_agents)
    assert len(session.revision_history) >= 1
    for prop in session.proposals:
        assert "hallucinated_id_999" not in prop.evidence_refs


def test_growth_objective_11_factor_prediction_and_creative_director_gate(tmp_path: Path) -> None:
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "gate.db")
    db = StudioDatabase(cfg.db_path)
    research = ResearchIntelligenceEngine(db)
    research_items = research.search_patterns(limit=25)
    report = research.generate_research_report("content_proposal", "science_engineering", ["faraday", "induction"])
    orch = SymposiumOrchestrator(cfg)
    analysis = _sample_analysis("proj_growth_gate")

    session = orch.run_symposium(
        "proj_growth_gate",
        analysis,
        report,
        mode=ResearchMode.STANDARD,
        research_items=research_items,
    )

    # 1. Pre-edit relevance investigation & grounded non-fabricated justification
    assert session.relevance_assessment is not None
    assert session.relevance_assessment.topic_timeliness in {"timely", "timely_anchor_to_evergreen", "evergreen"}
    assert 0.0 <= session.relevance_assessment.momentum_score <= 1.0
    assert session.relevance_assessment.why_viewer_cares_today

    # 2. Content Performance Prediction Engine: 3 ranked candidates, all 11 scorecard factors, uncertainty interval
    assert len(session.candidate_evaluations) == 3
    scores = [c.composite_score for c in session.candidate_evaluations]
    assert scores == sorted(scores, reverse=True)
    assert session.candidate_evaluations[0].selected is True
    assert sum(1 for c in session.candidate_evaluations if c.selected) == 1

    expected_dims = (
        "hook_strength",
        "topic_relevance",
        "clarity",
        "information_density",
        "curiosity",
        "emotional_engagement",
        "narrative_progression",
        "visual_quality",
        "audience_compatibility",
        "shareability",
        "estimated_long_term_relevance",
    )
    for cand in session.candidate_evaluations:
        sc_dump = cand.scorecard.model_dump()
        assert set(sc_dump.keys()) == set(expected_dims)
        assert all(0.0 <= v <= 1.0 for v in sc_dump.values())
        assert 0.0 <= cand.confidence_interval["low"] <= cand.confidence_interval["high"] <= 1.0
        assert cand.confidence_interval["uncertainty"] > 0.0
        assert len(cand.evidence_rationale) >= 2

    # 3. Growth Objective Evaluator proposal + Creative Director quality gate
    assert session.growth_objective is not None
    assert "physics_enthusiasts" in session.growth_objective.target_audience
    assert "curious_ambitious_general_audience" in session.growth_objective.target_audience
    assert session.growth_objective.accessible_entry_hook
    assert session.growth_objective.retention_risk_analysis

    assert session.creative_director_review is not None
    assert session.creative_director_review.approved is True
    assert session.creative_director_review.scorecard_rank == 1
    assert session.creative_director_review.selected_candidate_id == session.candidate_evaluations[0].candidate_id
    assert len(session.creative_director_review.checked_criteria) >= 4
