"""Effects catalog and procedural SFX synthesis."""

from symposium_studio.engines.effects.library import EFFECTS_CATALOG, EffectPreset, search_effects
from symposium_studio.engines.effects.sfx_synth import synthesize_sfx

__all__ = ["EFFECTS_CATALOG", "EffectPreset", "search_effects", "synthesize_sfx"]
