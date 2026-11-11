"""Programmatic animation specification and render adapter models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class FactualityMode(str, Enum):
    FACTUAL_SCIENTIFIC = "factual_scientific"
    ARTISTIC_ILLUSTRATION = "artistic_illustration"


class AnimationDomain(str, Enum):
    PHYSICS = "physics"
    ELECTRICAL_ENGINEERING = "electrical_engineering"
    PROGRAMMING = "programming"
    STORYTELLING = "storytelling"
    DATA_CHART = "data_chart"


class AdapterLicenseTier(str, Enum):
    PERMISSIVE_OPEN = "permissive_open"
    COPYLEFT_ISOLATED = "copyleft_isolated"
    COMMERCIAL_RESTRICTED = "commercial_restricted"


class AnimationAdapterCapability(BaseModel):
    adapter_id: str
    display_name: str
    version: str
    available: bool
    license_name: str
    license_tier: AdapterLicenseTier
    supports_alpha: bool
    supported_outputs: list[str]
    best_domains: list[AnimationDomain]
    constraints: list[str] = Field(default_factory=list)
    relative_cost: str = "low"
    health_message: str = ""


class KeyframeValue(BaseModel):
    time_sec: float
    value: Any
    easing: str = "ease_in_out"


class SceneElementSpec(BaseModel):
    element_id: str
    kind: str  # coil | magnet | field_lines | circuit_loop | waveform | array_node | tree_node | timeline_card | badge | equation
    label: str = ""
    props: dict[str, Any] = Field(default_factory=dict)
    keyframes: dict[str, list[KeyframeValue]] = Field(default_factory=dict)


class SceneSpec(BaseModel):
    scene_id: str
    title: str
    domain: AnimationDomain
    template_id: str  # e.g. em_induction_faraday, rlc_signal_flow, binary_tree_trace, narrative_timeline
    factuality_mode: FactualityMode = FactualityMode.FACTUAL_SCIENTIFIC
    duration_sec: float = 8.0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    transparent_background: bool = True
    preferred_adapter: str | None = None
    theme_palette: dict[str, str] = Field(
        default_factory=lambda: {
            "bg": "#0B1020",
            "primary": "#38BDF8",
            "accent": "#F59E0B",
            "secondary": "#34D399",
            "text": "#F8FAFC",
        }
    )
    scientific_notes: list[str] = Field(default_factory=list)
    elements: list[SceneElementSpec] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class AnimationRenderResult(BaseModel):
    scene_id: str
    adapter_used: str
    output_path: str
    frame_sequence_dir: str | None = None
    has_alpha: bool
    width: int
    height: int
    fps: float
    duration_sec: float
    scene_spec_path: str
    render_time_sec: float
    factuality_mode: FactualityMode
