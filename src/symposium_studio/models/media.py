"""Media analysis, multilingual transcription, diarization, vision, and highlight models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class MediaProbeResult(BaseModel):
    path: str
    sha256: str
    size_bytes: int
    duration_sec: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    has_audio: bool = True
    aspect_ratio: str = "16:9"
    mean_volume_db: float | None = None
    max_volume_db: float | None = None


class WordTimestamp(BaseModel):
    word: str
    start_sec: float
    end_sec: float
    probability: float = 0.9
    low_confidence: bool = False


class TranscriptSegment(BaseModel):
    segment_id: str
    start_sec: float
    end_sec: float
    text: str
    language: str = "en"
    confidence: float = 0.9
    low_confidence: bool = False
    low_confidence_reason: str | None = None
    speaker_id: str = "speaker_0"
    words: list[WordTimestamp] = Field(default_factory=list)


class TranscriptResult(BaseModel):
    engine: str
    model: str
    detected_language: str
    language_probability: float
    languages_observed: list[str] = Field(default_factory=list)
    duration_sec: float
    segments: list[TranscriptSegment] = Field(default_factory=list)
    uncertain_segment_ids: list[str] = Field(default_factory=list)
    verbatim_preserved: bool = True


class SpeakerTurn(BaseModel):
    speaker_id: str
    start_sec: float
    end_sec: float
    confidence: float = 0.75


class SilenceInterval(BaseModel):
    start_sec: float
    end_sec: float
    duration_sec: float


class SceneCut(BaseModel):
    scene_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    mean_motion: float = 0.0
    dominant_luma: float = 0.0


class FaceDetectionBox(BaseModel):
    time_sec: float
    x_norm: float
    y_norm: float
    w_norm: float
    h_norm: float
    center_x_norm: float
    center_y_norm: float


class VisualContextSample(BaseModel):
    time_sec: float
    faces: list[FaceDetectionBox] = Field(default_factory=list)
    ocr_text: str = ""
    motion_score: float = 0.0
    sharpness_score: float = 0.0
    suggested_crop_center_x: float = 0.5


class TopicCluster(BaseModel):
    topic_id: str
    title: str
    start_sec: float
    end_sec: float
    keywords: list[str] = Field(default_factory=list)
    summary: str = ""
    is_technical: bool = False
    domain: str = "general"


class HighlightScoreBreakdown(BaseModel):
    information_density: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    story_relevance: float = Field(ge=0.0, le=1.0)
    audience_curiosity: float = Field(ge=0.0, le=1.0)
    educational_value: float = Field(ge=0.0, le=1.0)
    emotional_intensity: float = Field(ge=0.0, le=1.0)
    visual_opportunity: float = Field(ge=0.0, le=1.0)
    context_dependency: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)


class HighlightProposal(BaseModel):
    highlight_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    role: str = Field(description="hook | core_explanation | turning_point | demo | payoff | short_candidate")
    title: str
    transcript_excerpt: str
    scores: HighlightScoreBreakdown
    visual_opportunity_type: str | None = None
    suggested_animation_domain: str | None = None
    rationale: str = ""
    kept: bool = True


class EditingStyleName(str, Enum):
    HIGH_ENERGY_YOUTUBE = "high_energy_youtube"
    CINEMATIC_STORYTELLING = "cinematic_storytelling"
    EDUCATIONAL_EXPLAINER = "educational_explainer"
    DOCUMENTARY = "documentary"
    TECHNICAL_TUTORIAL = "technical_tutorial"
    PODCAST_HIGHLIGHTS = "podcast_highlights"
    COMEDY = "comedy"
    PERSONAL_VLOG = "personal_vlog"
    SHORT_FORM_VIRAL = "short_form_viral"
    MINIMAL_PROFESSIONAL = "minimal_professional"


class EditingStyleProfile(BaseModel):
    name: EditingStyleName
    display_name: str
    target_avg_shot_sec: float
    max_silence_gap_sec: float
    transition_style: str
    caption_preset: str
    color_grade_preset: str
    zoom_punch_intensity: float = Field(ge=0.0, le=1.0)
    broll_animation_ratio: float = Field(ge=0.0, le=1.0)
    audio_duck_db: float = -12.0
    description: str = ""


class MediaAnalysisBundle(BaseModel):
    project_id: str
    probe: MediaProbeResult
    transcript: TranscriptResult
    speaker_turns: list[SpeakerTurn] = Field(default_factory=list)
    silences: list[SilenceInterval] = Field(default_factory=list)
    scene_cuts: list[SceneCut] = Field(default_factory=list)
    visual_samples: list[VisualContextSample] = Field(default_factory=list)
    topics: list[TopicCluster] = Field(default_factory=list)
    highlights: list[HighlightProposal] = Field(default_factory=list)
    repetitions_flagged: list[dict[str, Any]] = Field(default_factory=list)
    recommended_crop_center_x: float = 0.5
