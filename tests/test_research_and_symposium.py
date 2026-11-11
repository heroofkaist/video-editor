"""Tests for cross-niche Research Intelligence, FTS5 search, and AI Symposium deliberation."""

from __future__ import annotations

from pathlib import Path

from symposium_studio.config import StudioConfig
from symposium_studio.engines.research import ResearchIntelligenceEngine, import_explicit_item
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
