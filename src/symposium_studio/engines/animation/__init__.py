"""Original animation generation engine & extensible render graph adapters."""

from symposium_studio.engines.animation.registry import AnimationEngineRegistry
from symposium_studio.engines.animation.scene_specs import build_scene_spec

__all__ = ["AnimationEngineRegistry", "build_scene_spec"]
