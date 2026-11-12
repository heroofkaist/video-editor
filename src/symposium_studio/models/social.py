"""Social repurposing, Telegram community funnel, publishing, QA, and growth models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class PublishingMode(str, Enum):
    EXPORT_ONLY = "export_only"
    DRAFT_PREP = "draft_prep"
    APPROVED_PUBLISH = "approved_publish"


class ThreadsStyle(str, Enum):
    PROVOCATIVE = "provocative"
    EDUCATIONAL = "educational"
    PERSONAL = "personal"
    STORYTELLING = "storytelling"
    HUMOROUS = "humorous"
    TECHNICAL = "technical"


class YouTubePackage(BaseModel):
    title_candidates: list[str]
    description: str
    chapters: list[dict[str, str]]
    tags: list[str]
    thumbnail_concepts: list[dict[str, str]]
    thumbnail_path: str | None = None


class ThreadsPost(BaseModel):
    post_id: str
    style: ThreadsStyle
    text: str
    char_count: int
    max_chars: int = 500
    cta_to_telegram: str = ""
    grounded_facts: list[str] = Field(default_factory=list)


class ShortFormCopy(BaseModel):
    platform: str  # youtube_shorts | tiktok | instagram_reels
    variant_id: str
    hook_line: str
    caption: str
    hashtags: list[str] = Field(default_factory=list)
    cta_to_telegram: str = ""
    video_path: str | None = None


class FunnelLink(BaseModel):
    link_id: str
    source_platform: str
    campaign_slug: str
    target_telegram_channel: str
    tagged_url: str
    clicks: int = 0
    conversions: int = 0


class TelegramPoll(BaseModel):
    question: str
    options: list[str]
    is_anonymous: bool = True


class TelegramCampaign(BaseModel):
    campaign_id: str
    project_id: str
    target_channel: str
    lead_post: str
    technical_deep_dive_note: str
    discussion_questions: list[str] = Field(default_factory=list)
    resource_checklist: list[str] = Field(default_factory=list)
    mini_series_outline: list[str] = Field(default_factory=list)
    poll: TelegramPoll | None = None
    suggested_posting_windows_utc: list[str] = Field(default_factory=list)
    funnel_links: list[FunnelLink] = Field(default_factory=list)
    platform_ctas: dict[str, str] = Field(default_factory=dict)


class SocialRepurposingBundle(BaseModel):
    project_id: str
    youtube: YouTubePackage
    shorts_copy: list[ShortFormCopy] = Field(default_factory=list)
    threads_posts: list[ThreadsPost] = Field(default_factory=list)
    short_summary: str
    follow_up_ideas: list[str] = Field(default_factory=list)
    telegram_campaign: TelegramCampaign


class QualityCheckIssue(BaseModel):
    severity: str  # error | warning | info
    code: str
    message: str
    timestamp_sec: float | None = None


class QualityAssuranceReport(BaseModel):
    qa_id: str
    project_id: str
    artifact_path: str
    passed: bool
    width: int
    height: int
    fps: float
    duration_sec: float
    has_video: bool
    has_audio: bool
    audio_mean_db: float | None = None
    audio_max_db: float | None = None
    measured_lufs: float | None = None
    black_frame_intervals: list[dict[str, float]] = Field(default_factory=list)
    freeze_frame_intervals: list[dict[str, float]] = Field(default_factory=list)
    caption_count: int = 0
    overlay_count: int = 0
    caption_quality: dict[str, Any] = Field(default_factory=dict)
    editorial_checks: dict[str, Any] = Field(default_factory=dict)
    verification_frame_paths: list[str] = Field(default_factory=list)
    issues: list[QualityCheckIssue] = Field(default_factory=list)


class GrowthMetricSnapshot(BaseModel):
    snapshot_id: str
    project_id: str
    platform: str
    measurement_mode: str = "cumulative"  # cumulative | delta
    external_post_id: str | None = None
    views_or_impressions: int
    first_7d_views: int | None = None
    likes_or_reactions: int = 0
    comments: int = 0
    shares: int = 0
    new_subscribers_or_followers: int = 0
    avg_view_duration_sec: float | None = None
    watch_time_minutes: float | None = None
    authorized_ctr_pct: float | None = None
    authorized_retention_pct: float | None = None
    telegram_clicks: int = 0
    telegram_joins: int = 0
    subscriber_or_member_count: int | None = None
    data_provenance: str = "authorized_analytics_import"
    recorded_at: str
    notes: str = ""


class PublishRequest(BaseModel):
    project_id: str
    platform: str
    mode: PublishingMode = PublishingMode.EXPORT_ONLY
    approval_confirmed: bool = False
    approval_reason: str = ""
    artifact_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
