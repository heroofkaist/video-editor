"""AI Symposium multi-agent deliberation, critique, storyboard, and decision models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ResearchMode(str, Enum):
    INEXPENSIVE = "inexpensive"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class AgentTeam(str, Enum):
    RESEARCH = "research"
    CREATIVE = "creative"
    MODERATOR = "moderator"


class AgentProposal(BaseModel):
    proposal_id: str
    agent_id: str
    agent_title: str
    team: AgentTeam
    summary: str
    recommendations: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_actions: list[dict[str, Any]] = Field(default_factory=list)


class ModeratorChallenge(BaseModel):
    challenge_id: str
    target_proposal_id: str
    issue_type: str  # disagreement | unsupported_assumption | factuality_check | pacing_conflict
    description: str
    resolution: str


class StoryboardBeat(BaseModel):
    beat_index: int
    role: str  # hook | setup | core_insight | animation_explainer | proof | payoff | cta
    source_start_sec: float
    source_end_sec: float
    target_duration_sec: float
    visual_treatment: str
    animation_template: str | None = None
    effect_presets: list[str] = Field(default_factory=list)
    caption_directive: str = ""
    audio_directive: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class Storyboard(BaseModel):
    storyboard_id: str
    project_id: str
    target_format: str = "youtube_long"
    selected_strategy: str
    target_duration_sec: float
    beats: list[StoryboardBeat] = Field(default_factory=list)


class SymposiumDecision(BaseModel):
    decision_id: str
    topic: str
    chosen_option: str
    rejected_options: list[str] = Field(default_factory=list)
    supporting_agents: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str


class SymposiumSession(BaseModel):
    session_id: str
    project_id: str
    mode: ResearchMode
    backend_used: str = "deterministic_local"
    participating_agents: list[str] = Field(default_factory=list)
    proposals: list[AgentProposal] = Field(default_factory=list)
    challenges: list[ModeratorChallenge] = Field(default_factory=list)
    decisions: list[SymposiumDecision] = Field(default_factory=list)
    storyboard: Storyboard
    created_at: str
