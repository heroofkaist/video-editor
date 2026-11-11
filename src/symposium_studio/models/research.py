"""Viral research intelligence and cross-platform pattern models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ResearchPlatform(str, Enum):
    YOUTUBE = "youtube"
    YOUTUBE_SHORTS = "youtube_shorts"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    INSTAGRAM_POST = "instagram_post"
    THREADS = "threads"


class ContentCategory(str, Enum):
    ENTERTAINMENT = "entertainment"
    GAMING = "gaming"
    COMEDY = "comedy"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    ENGINEERING = "engineering"
    PROGRAMMING = "programming"
    EDUCATION = "education"
    BUSINESS = "business"
    STARTUPS = "startups"
    SPORTS = "sports"
    LIFESTYLE = "lifestyle"
    PERSONAL_STORYTELLING = "personal_storytelling"
    DOCUMENTARIES = "documentaries"
    POPULAR_CULTURE = "popular_culture"
    INTERNET_TRENDS = "internet_trends"


class PublicEngagementMetrics(BaseModel):
    """Strictly platform-scoped public metrics; private analytics are never fabricated."""

    platform: ResearchPlatform
    metric_source: str = Field(description="official_api | user_supplied | curated_structural_benchmark")
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares_or_reposts: int | None = None
    subscribers_or_followers_at_capture: int | None = None
    published_at: str | None = None
    captured_at: str | None = None
    # Note: retention & CTR intentionally omitted unless supplied from authorized analytics
    authorized_ctr: float | None = None
    authorized_avg_view_pct: float | None = None


class ReusableTechnique(BaseModel):
    technique_id: str
    name: str
    mechanism: str
    applicable_niches: list[str] = Field(default_factory=list)
    editing_translation: str


class ViralContentItem(BaseModel):
    item_id: str
    platform: ResearchPlatform
    creator: str
    original_url: str
    title: str
    published_at: str
    duration_sec: float | None = None
    metrics: PublicEngagementMetrics
    category: ContentCategory
    topic: str
    opening_hook_structure: str
    editing_style: str
    narrative_structure: str
    visual_effects: list[str] = Field(default_factory=list)
    audio_patterns: list[str] = Field(default_factory=list)
    caption_style: str = ""
    thumbnail_strategy: str = ""
    reusable_techniques: list[ReusableTechnique] = Field(default_factory=list)
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    platform_relative_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentile/momentum score strictly computed within the same platform and cohort.",
    )
    provenance_type: str = "curated_public_study"  # curated_public_study | official_api | explicit_import


class ResearchReport(BaseModel):
    report_id: str
    report_type: str  # weekly_trends | viral_formats | hook_library | editing_techniques | audience_interest | competitive_analysis | inspiration_board | content_proposal
    title: str
    target_niche: str
    summary: str
    source_item_ids: list[str] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    cross_niche_transfers: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    created_at: str
