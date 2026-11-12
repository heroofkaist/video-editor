"""Data models for Multi-Reference Video Intelligence, Style DNA, Selection, and Style Transfer."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReferenceSourceType(str, Enum):
    LOCAL_UPLOAD = "local_upload"
    WORKSPACE_FILE = "workspace_file"
    URL_VIDEO = "url_video"
    URL_METADATA_ONLY = "url_metadata_only"
    LIBRARY_CACHE = "library_cache"


class ReferenceAnalysisStatus(str, Enum):
    PENDING = "pending"
    FULL_VIDEO_AUDIO = "full_video_and_audio"
    METADATA_ONLY = "metadata_only"
    FAILED = "failed"


class SelectionMode(str, Enum):
    BEST_SINGLE = "best_single"
    INTELLIGENT_HYBRID = "intelligent_hybrid"
    MANUAL_OVERRIDE = "manual_override"


class ReferenceEditEvent(BaseModel):
    """Timestamped audiovisual edit event observed in a reference video."""

    event_id: str
    event_type: str  # cut | punch_in | jump_cut | motion_graphic | reaction_cutaway | sfx_transient | freeze_moment | static_hold
    start_sec: float
    end_sec: float
    duration_sec: float
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)
    description: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class SubtitleStyleObservation(BaseModel):
    """Observable caption styling & cadence measurements from reference frames and speech."""

    presence_detected: bool = False
    vertical_region: str = "bottom_safe"  # bottom_safe | lower_third | center_dynamic | upper_mid | none_detected
    estimated_words_per_cue: int = 3
    estimated_font_weight: str = "bold"  # bold | heavy | medium
    color_scheme_hint: str = "white_yellow_high_contrast"
    stroke_contrast_score: float = Field(ge=0.0, le=1.0, default=0.75)
    speech_wpm: float = 0.0
    speech_language: str = "unknown"
    evidence_note: str = "Estimated from frame edge/chroma band variance and local Whisper cadence."


class AudioStyleObservation(BaseModel):
    """Observable audio mixing, speech-pause structure, and SFX transient density."""

    has_audio: bool = True
    mean_db: float = -18.0
    max_db: float = -1.0
    measured_lufs: float | None = None
    loudness_range_lu: float | None = None
    speech_activity_ratio: float = Field(ge=0.0, le=1.0, default=0.75)
    silence_gap_count: int = 0
    median_pause_sec: float = 0.25
    transient_sfx_per_min: float = 0.0
    estimated_music_bed_present: bool = False
    music_bed_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    dialogue_ducking_ratio: float = Field(ge=0.0, le=1.0, default=0.0)


class VisualStyleObservation(BaseModel):
    """Color, contrast, framing, camera movement, and format metrics."""

    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    aspect_ratio: str = "16:9"
    is_vertical: bool = False
    has_blur_pillarbars: bool = False
    median_brightness: float = Field(ge=0.0, le=1.0, default=0.5)
    median_saturation: float = Field(ge=0.0, le=1.0, default=0.4)
    contrast_range: float = Field(ge=0.0, le=1.0, default=0.5)
    color_temperature_bias: str = "neutral"  # warm | cool | neutral
    face_presence_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    median_face_height_ratio: float = 0.0
    framing_style: str = "medium_talking_head"  # tight_closeup | medium_talking_head | wide_desk | b_roll_dominant | graphic_dominant
    camera_motion_mean: float = 0.0
    motion_graphics_ratio: float = Field(ge=0.0, le=1.0, default=0.0)


class PacingRhythmObservation(BaseModel):
    """Shot durations, cut density, punch-ins, and narrative structure cadence."""

    duration_sec: float
    total_cuts: int = 0
    cuts_per_minute: float = 0.0
    mean_shot_duration_sec: float = 2.0
    median_shot_duration_sec: float = 2.0
    p25_shot_duration_sec: float = 1.2
    p75_shot_duration_sec: float = 3.2
    punch_in_count: int = 0
    punch_ins_per_minute: float = 0.0
    hook_window_cuts_first_5s: int = 0
    hook_first_cut_sec: float | None = None
    hook_first_speech_sec: float | None = None
    climax_activity_time_ratio: float = Field(ge=0.0, le=1.0, default=0.5)
    energy_arc: str = "steady_build"  # explosive_frontloaded | steady_build | alternating_peaks | calm_deliberate


class StyleDNA(BaseModel):
    """Structured machine-readable style blueprint extracted from an analyzed reference."""

    dna_id: str
    reference_id: str
    schema_version: str = "1.0.0"
    analysis_status: ReferenceAnalysisStatus
    mapped_base_style: str = "educational_explainer"
    archetype_label: str = "educational_creator"
    pacing: PacingRhythmObservation
    subtitles: SubtitleStyleObservation
    audio: AudioStyleObservation
    visual: VisualStyleObservation
    recommended_animation_engine: str = "kinetic_callout"  # kinetic_callout | manim_factual_diagram | plotly_chart | meme_reaction_card
    recommended_sfx_intensity: float = Field(ge=0.0, le=1.0, default=0.35)
    recommended_meme_density: float = Field(ge=0.0, le=1.0, default=0.15)
    evidence_artifacts: dict[str, str] = Field(default_factory=dict)
    editorial_notes: list[str] = Field(default_factory=list)
    honest_limitations: list[str] = Field(default_factory=list)


class ReferenceItem(BaseModel):
    """A single reference video or metadata-only reference entry in library/project."""

    reference_id: str
    project_id: str | None = None
    title: str
    source_type: ReferenceSourceType
    source_path: str | None = None
    source_url: str | None = None
    sha256: str | None = None
    file_size_bytes: int = 0
    duration_sec: float = 0.0
    platform: str = "unknown"
    creator: str = ""
    analysis_status: ReferenceAnalysisStatus = ReferenceAnalysisStatus.PENDING
    style_dna: StyleDNA | None = None
    edit_events_path: str | None = None
    contact_sheet_path: str | None = None
    evidence_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)


class CriterionScore(BaseModel):
    """Explicit heuristic compatibility criterion score (never presented as view probability)."""

    criterion: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    rationale: str


class ReferenceCandidateEvaluation(BaseModel):
    """Compatibility ranking entry for one reference against a specific source footage & platform."""

    reference_id: str
    title: str
    analysis_status: ReferenceAnalysisStatus
    target_platform: str
    heuristic_compatibility_score: float = Field(ge=0.0, le=1.0)
    is_feasible: bool = True
    disqualification_reason: str | None = None
    criteria_breakdown: list[CriterionScore] = Field(default_factory=list)
    transferable_strengths: list[str] = Field(default_factory=list)
    identified_risks: list[str] = Field(default_factory=list)
    score_disclaimer: str = (
        "Heuristic editorial compatibility score computed from analyzed source footage and reference signals; "
        "not a forecast of audience views or algorithmic distribution."
    )


class HybridStyleGuide(BaseModel):
    """Multi-reference blend selecting distinct structural dimensions from compatible references."""

    primary_reference_id: str
    dimension_sources: dict[str, str] = Field(default_factory=dict)  # e.g. {"pacing": ref_a, "subtitles": ref_b}
    blended_notes: list[str] = Field(default_factory=list)


class ConferenceAgentStatement(BaseModel):
    """Individual specialist perspective during the AI Symposium Reference Conference."""

    agent_id: str
    agent_title: str
    stance_reference_id: str | None = None
    concerns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class ReferenceConferenceRecord(BaseModel):
    """Full debate and selection audit produced by the Symposium Reference Conference."""

    conference_id: str
    project_id: str
    target_platform: str
    selection_mode: SelectionMode
    backend_used: str = "deterministic_analytical"
    selected_reference_id: str
    rejected_reference_ids: list[str] = Field(default_factory=list)
    hybrid_guide: HybridStyleGuide | None = None
    agent_statements: list[ConferenceAgentStatement] = Field(default_factory=list)
    moderator_warnings: list[str] = Field(default_factory=list)
    final_rationale: str = ""
    created_at: str = Field(default_factory=_utc_now)


class StyleTransferParameters(BaseModel):
    """Executable editing parameters translated from StyleDNA/HybridStyleGuide with safety bounds."""

    base_style_profile: str = "educational_explainer"
    target_shot_duration_sec: float = 2.4
    cut_pause_threshold_sec: float = 0.25
    zoom_punch_step: float = 0.08
    max_hook_duration_sec: float = 3.5
    subtitle_max_words: int = 3
    subtitle_font_size: int = 58
    subtitle_scale_factor: float = 1.0
    subtitle_safe_margin_bottom_px: int = 100
    subtitle_position_hint: str = "bottom_safe"
    animation_engine_hint: str = "kinetic_callout"
    overlay_cadence_multiplier: float = 1.0
    sfx_click_min_spacing_sec: float = 2.5
    sfx_gain_db: float = -12.0
    music_duck_db: float = -22.0
    enable_memes: bool = True
    enable_educational_graphics: bool = True
    pacing_multiplier: float = 1.0
    color_saturation: float = 1.04
    color_contrast: float = 1.03
    color_brightness: float = 0.0
    keep_structure_apply_rhythm_only: bool = False


class StyleTransferPlan(BaseModel):
    """Explicit bridge from Selected Reference Style DNA -> Concrete Project Timeline Edits."""

    plan_id: str
    project_id: str
    target_platform: str
    selection_mode: SelectionMode
    primary_reference_id: str
    hybrid_guide: HybridStyleGuide | None = None
    parameters: StyleTransferParameters
    applied_adjustments: dict[str, Any] = Field(default_factory=dict)
    timeline_directives: list[str] = Field(default_factory=list)
    constraints_enforced: list[str] = Field(
        default_factory=lambda: [
            "Creator spoken narrative meaning preserved verbatim (no fabricated dialogue).",
            "Multilingual kk/ru/en subtitle integrity & word-level alignment strictly enforced.",
            "No copyrighted reference footage, music, or third-party channel assets copied.",
            "Visual safe zones protected from face/subtitle overlap.",
        ]
    )
    created_at: str = Field(default_factory=_utc_now)


class StyleSimilarityMetric(BaseModel):
    metric: str
    reference_value: float | str | None = None
    rendered_value: float | str | None = None
    delta: float | None = None
    alignment_status: str = "aligned"  # aligned | close | diverged_for_clarity


class StyleSimilarityReport(BaseModel):
    """Post-render comparison measuring rendered output against chosen StyleDNA targets."""

    report_id: str
    project_id: str
    timeline_id: str
    reference_id: str
    rendered_video_path: str
    overall_style_adherence_score: float = Field(ge=0.0, le=1.0)
    metrics: list[StyleSimilarityMetric] = Field(default_factory=list)
    intentional_divergences: list[str] = Field(default_factory=list)
    qa_passed: bool = False
    created_at: str = Field(default_factory=_utc_now)
