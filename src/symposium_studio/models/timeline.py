"""Unified editable timeline and render graph models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class TargetPlatformFormat(str, Enum):
    YOUTUBE_LONG = "youtube_long"
    YOUTUBE_SHORTS = "youtube_shorts"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"


class EffectCategory(str, Enum):
    TRANSITION = "transition"
    TYPOGRAPHY = "typography"
    VISUAL = "visual"
    SOUND = "sound"


class EffectSpec(BaseModel):
    effect_id: str
    preset_id: str
    category: EffectCategory
    start_sec: float
    duration_sec: float
    parameters: dict[str, Any] = Field(default_factory=dict)
    editorial_purpose: str = ""


class TransitionSpec(BaseModel):
    type: str = "cut"  # cut, cross_dissolve, whip_pan, zoom_push, dip_black
    duration_sec: float = 0.0
    sfx_preset_id: str | None = None


class VideoClipNode(BaseModel):
    clip_id: str
    source_path: str
    source_start_sec: float
    source_end_sec: float
    timeline_start_sec: float
    timeline_end_sec: float
    playback_speed: float = 1.0
    crop_center_x: float = 0.5
    zoom_scale: float = 1.0
    label: str = ""
    transition_in: TransitionSpec | None = None
    effects: list[EffectSpec] = Field(default_factory=list)


class OverlayNode(BaseModel):
    overlay_id: str
    asset_path: str
    scene_spec_path: str | None = None
    timeline_start_sec: float
    timeline_end_sec: float
    mode: str = "fullscreen_cutaway"  # fullscreen_cutaway | pip_top_right | split_screen | alpha_overlay
    opacity: float = 1.0
    x_norm: float = 0.0
    y_norm: float = 0.0
    w_norm: float = 1.0
    h_norm: float = 1.0
    label: str = ""


class AudioNode(BaseModel):
    audio_id: str
    source_path: str
    role: str = "sfx"  # dialogue | sfx | music | ambient
    timeline_start_sec: float
    duration_sec: float
    gain_db: float = 0.0
    duck_under_speech: bool = False


class CaptionCue(BaseModel):
    cue_id: str
    start_sec: float
    end_sec: float
    text: str
    words: list[dict[str, Any]] = Field(default_factory=list)
    low_confidence: bool = False
    style_preset: str = "kinetic_caption_pop"


class CaptionSettings(BaseModel):
    enabled: bool = True
    font_name: str = "Helvetica Neue"
    font_size: int = 58
    scale_factor: float = 1.0
    safe_margin_bottom_px: int = 90
    safe_margin_top_px: int = 80
    highlight_active_word: bool = True
    max_words_per_line: int = 5


class RenderTimeline(BaseModel):
    timeline_id: str
    project_id: str
    revision: int = 1
    title: str
    platform_format: TargetPlatformFormat = TargetPlatformFormat.YOUTUBE_LONG
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    style_profile: str = "educational_explainer"
    clips: list[VideoClipNode] = Field(default_factory=list)
    overlays: list[OverlayNode] = Field(default_factory=list)
    audio_events: list[AudioNode] = Field(default_factory=list)
    captions: list[CaptionCue] = Field(default_factory=list)
    caption_settings: CaptionSettings = Field(default_factory=CaptionSettings)
    global_effects: list[EffectSpec] = Field(default_factory=list)
    edit_notes: list[str] = Field(default_factory=list)

    @property
    def total_duration_sec(self) -> float:
        if not self.clips:
            return 0.0
        return max(c.timeline_end_sec for c in self.clips)
