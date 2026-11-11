"""Growth objective evaluation, topic relevance, candidate performance prediction, and experiment learning models."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


CREATOR_CORE_AUDIENCE: tuple[str, ...] = (
    "physics_enthusiasts",
    "electrical_engineers",
    "programmers",
    "ai_researchers",
    "technology_enthusiasts",
    "students",
)


class RelevanceAssessment(BaseModel):
    """Pre-edit investigation of topic timeliness, momentum, and why viewers care today."""

    topic_timeliness: str = Field(
        description="timely | timely_anchor_to_evergreen | evergreen"
    )
    momentum_score: float = Field(ge=0.0, le=1.0)
    matched_trends: list[dict[str, Any]] = Field(default_factory=list)
    why_viewer_cares_today: str
    long_term_relevance_note: str
    limitations: list[str] = Field(default_factory=list)


class PerformancePredictionScorecard(BaseModel):
    """11-factor pre-render evaluation scorecard (0.0-1.0); never fabricates view counts or virality guarantees."""

    hook_strength: float = Field(ge=0.0, le=1.0)
    topic_relevance: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)
    information_density: float = Field(ge=0.0, le=1.0)
    curiosity: float = Field(ge=0.0, le=1.0)
    emotional_engagement: float = Field(ge=0.0, le=1.0)
    narrative_progression: float = Field(ge=0.0, le=1.0)
    visual_quality: float = Field(ge=0.0, le=1.0)
    audience_compatibility: float = Field(ge=0.0, le=1.0)
    shareability: float = Field(ge=0.0, le=1.0)
    estimated_long_term_relevance: float = Field(ge=0.0, le=1.0)


class CandidateEditEvaluation(BaseModel):
    """Ranked candidate edit strategy with explicit uncertainty bounds and evidence rationale."""

    candidate_id: str
    strategy_name: str
    editing_style: str
    beat_highlight_ids: list[str] = Field(default_factory=list)
    target_duration_sec: float
    scorecard: PerformancePredictionScorecard
    composite_score: float = Field(ge=0.0, le=1.0)
    historical_lift_applied: float = 0.0
    confidence_interval: dict[str, float] = Field(
        default_factory=lambda: {"low": 0.45, "high": 0.85, "uncertainty": 0.20}
    )
    evidence_rationale: list[str] = Field(default_factory=list)
    selected: bool = False


class GrowthObjectiveProposal(BaseModel):
    """Required Growth Objective Evaluator brief attached to every proposed production plan."""

    target_audience: list[str] = Field(default_factory=list)
    accessible_entry_hook: str
    content_hypothesis: str
    relevance_assessment: RelevanceAssessment
    expected_viewer_value: str
    selected_hook_strategy: str
    retention_risk_analysis: list[str] = Field(default_factory=list)
    distribution_strategy: dict[str, str] = Field(default_factory=dict)
    measurement_plan: dict[str, Any] = Field(default_factory=dict)


class CreativeDirectorGateReview(BaseModel):
    """Pre-approval audit ensuring Growth Objective criteria and candidate rank checks are satisfied."""

    approved: bool
    checked_criteria: list[str] = Field(default_factory=list)
    selected_candidate_id: str
    scorecard_rank: int = 1
    revision_notes: str = ""


class ContentExperimentRecord(BaseModel):
    """Searchable per-platform production & post-publication learning experiment."""

    experiment_id: str
    project_id: str
    title: str
    creator_niche: str = "science_engineering"
    platform: str  # youtube | youtube_shorts | tiktok | instagram_reels | threads
    editing_style: str
    strategy_name: str
    hook_strategy: str
    target_audience: list[str] = Field(default_factory=list)
    content_hypothesis: str
    predicted_composite_score: float = Field(ge=0.0, le=1.0)
    predicted_uncertainty: float = Field(ge=0.0, le=1.0)
    predicted_dimensions: dict[str, float] = Field(default_factory=dict)
    research_item_ids: list[str] = Field(default_factory=list)
    decision_summary: str = ""
    published_at: str
    first_7d_views: int | None = None
    engagement_rate_pct: float | None = None
    avg_view_duration_sec: float | None = None
    watch_time_minutes: float | None = None
    authorized_ctr_pct: float | None = None
    authorized_retention_pct: float | None = None
    shares: int = 0
    new_subscribers_or_followers: int = 0
    telegram_clicks: int = 0
    telegram_joins: int = 0
    creator_baseline_median_views: float | None = None
    ratio_vs_creator_median: float | None = None
    growth_tier_reached: str = "awaiting_post_publication_data"
    outcome_verdict: str = "pending"  # pending | outperformed | matched_baseline | underperformed
    learned_hypotheses: list[str] = Field(default_factory=list)
    updated_at: str
