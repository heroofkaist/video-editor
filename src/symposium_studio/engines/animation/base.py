"""Base interface for interchangeable programmatic animation & 3D render adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from symposium_studio.models.animation import (
    AnimationAdapterCapability,
    AnimationRenderResult,
    SceneSpec,
)


class AnimationAdapter(ABC):
    """Contract implemented by every rendering engine (Chromium Canvas, Motion Canvas, Manim, Blender, Remotion, Matplotlib)."""

    @abstractmethod
    def capability(self) -> AnimationAdapterCapability:
        """Expose capabilities, constraints, licensing tier, and runtime health."""

    @abstractmethod
    def render(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        """Render the editable SceneSpec into a video/alpha-overlay asset."""
