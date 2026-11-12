"""Project workspace, manifest, and provenance models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


WORKSPACE_SUBDIRS: tuple[str, ...] = (
    "source",
    "references",
    "transcripts",
    "research",
    "symposium",
    "storyboards",
    "timelines",
    "animations",
    "effects",
    "assets",
    "renders",
    "shorts",
    "thumbnails",
    "social",
    "telegram",
    "analytics",
    "reports",
)


class AssetRole(str, Enum):
    SOURCE_MEDIA = "source_media"
    REFERENCE_VIDEO = "reference_video"
    STYLE_DNA = "style_dna"
    STYLE_TRANSFER_REPORT = "style_transfer_report"
    TRANSCRIPT = "transcript"
    RESEARCH_REPORT = "research_report"
    SYMPOSIUM_LOG = "symposium_log"
    STORYBOARD = "storyboard"
    TIMELINE_JSON = "timeline_json"
    TIMELINE_OTIO = "timeline_otio"
    ANIMATION_SPEC = "animation_spec"
    ANIMATION_RENDER = "animation_render"
    SFX_ASSET = "sfx_asset"
    RENDER_YOUTUBE = "render_youtube"
    RENDER_SHORT = "render_short"
    RENDER_TIKTOK = "render_tiktok"
    RENDER_REEL = "render_reel"
    THUMBNAIL = "thumbnail"
    CAPTION_FILE = "caption_file"
    SOCIAL_PACKAGE = "social_package"
    TELEGRAM_CAMPAIGN = "telegram_campaign"
    QA_REPORT = "qa_report"
    PRODUCTION_REPORT = "production_report"


class AssetRecord(BaseModel):
    asset_id: str
    project_id: str
    run_id: str
    role: AssetRole
    relative_path: str
    absolute_path: str
    sha256: str
    size_bytes: int
    mime_type: str = "application/octet-stream"
    source_attribution: str = "original_or_generated"
    license_info: str = "Creator-owned / Apache-2.0 tooling"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class ToolVersionRecord(BaseModel):
    name: str
    version: str
    available: bool = True
    license: str = ""
    notes: str = ""


class DecisionTraceEntry(BaseModel):
    step: str
    actor: str
    decision: str
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str
    timestamp: str = Field(default_factory=utc_now_iso)


class RunRecord(BaseModel):
    run_id: str
    project_id: str
    command: str
    status: str = "running"
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str | None = None
    duration_sec: float | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    tool_versions: list[ToolVersionRecord] = Field(default_factory=list)
    decisions: list[DecisionTraceEntry] = Field(default_factory=list)
    output_asset_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class ProjectRecord(BaseModel):
    project_id: str
    name: str
    slug: str
    creator_niche: str = "science_engineering"
    editing_style: str = "educational_explainer"
    workspace_dir: str
    source_video_path: str | None = None
    source_sha256: str | None = None
    status: str = "initialized"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    tags: list[str] = Field(default_factory=list)


class ProjectManifest(BaseModel):
    schema_version: str = "1.0.0"
    project: ProjectRecord
    runs: list[RunRecord] = Field(default_factory=list)
    assets: list[AssetRecord] = Field(default_factory=list)
    decisions: list[DecisionTraceEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
