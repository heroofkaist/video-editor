"""Unified animation adapter registry and intelligent engine selection."""

from __future__ import annotations

from pathlib import Path

from symposium_studio.config import StudioConfig
from symposium_studio.engines.animation.adapters.chromium_canvas import ChromiumCanvasAdapter
from symposium_studio.engines.animation.adapters.external_adapters import (
    BlenderAdapter,
    ManimAdapter,
    MotionCanvasAdapter,
    RemotionAdapter,
)
from symposium_studio.engines.animation.adapters.matplotlib_vector import MatplotlibVectorAdapter
from symposium_studio.engines.animation.base import AnimationAdapter
from symposium_studio.models.animation import (
    AnimationAdapterCapability,
    AnimationRenderResult,
    SceneSpec,
)


class AnimationEngineRegistry:
    """Selects and executes the best available animation adapter per scene."""

    def __init__(self, config: StudioConfig) -> None:
        self.config = config
        self.adapters: dict[str, AnimationAdapter] = {
            "chromium_canvas": ChromiumCanvasAdapter(),
            "matplotlib_vector": MatplotlibVectorAdapter(),
            "motion_canvas": MotionCanvasAdapter(),
            "manim": ManimAdapter(),
            "blender": BlenderAdapter(),
            "remotion": RemotionAdapter(allow_remotion=config.safety.allow_remotion_adapter),
        }

    def list_capabilities(self) -> list[AnimationAdapterCapability]:
        return [a.capability() for a in self.adapters.values()]

    def select_adapter(self, scene_spec: SceneSpec) -> AnimationAdapter:
        if scene_spec.preferred_adapter and scene_spec.preferred_adapter in self.adapters:
            cand = self.adapters[scene_spec.preferred_adapter]
            if cand.capability().available:
                return cand
        # Prefer Chromium Canvas for rich multi-domain 2D/SVG motion graphics; fall back to Matplotlib
        for key in ("chromium_canvas", "matplotlib_vector"):
            cand = self.adapters[key]
            if cand.capability().available:
                return cand
        return self.adapters["matplotlib_vector"]

    def render_scene(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        adapter = self.select_adapter(scene_spec)
        try:
            return adapter.render(scene_spec, output_dir)
        except Exception:
            if adapter.capability().adapter_id != "matplotlib_vector":
                # Reliable pure-Python deterministic fallback if headless browser launch fails in restricted sandbox
                return self.adapters["matplotlib_vector"].render(scene_spec, output_dir)
            raise
