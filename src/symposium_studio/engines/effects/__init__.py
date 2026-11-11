"""Effects catalog, internet click SFX downloader, and procedural SFX synthesis."""

from symposium_studio.engines.effects.library import EFFECTS_CATALOG, EffectPreset, search_effects
from symposium_studio.engines.effects.sfx_synth import synthesize_ambient_bed, synthesize_sfx
from symposium_studio.engines.effects.sound_downloader import (
    DEFAULT_CLICK_SOUNDS,
    download_click_sounds,
    download_internet_sound,
    search_internet_clicks,
)

__all__ = [
    "DEFAULT_CLICK_SOUNDS",
    "EFFECTS_CATALOG",
    "EffectPreset",
    "download_click_sounds",
    "download_internet_sound",
    "search_effects",
    "search_internet_clicks",
    "synthesize_ambient_bed",
    "synthesize_sfx",
]
