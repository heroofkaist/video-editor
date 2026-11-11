"""Searchable Effects Library with configurable FFmpeg & ASS presets across all 4 categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from symposium_studio.models.timeline import EffectCategory


@dataclass(frozen=True)
class EffectPreset:
    preset_id: str
    name: str
    category: EffectCategory
    description: str
    editorial_purpose: str
    ffmpeg_filter_hint: str
    parameters: dict[str, Any] = field(default_factory=dict)


EFFECTS_CATALOG: dict[str, EffectPreset] = {
    # Transitions
    "whip_pan": EffectPreset(
        preset_id="whip_pan",
        name="Whip Pan Slide",
        category=EffectCategory.TRANSITION,
        description="High-velocity horizontal blur slide between narrative beats.",
        editorial_purpose="Signals sharp energy shift or comedic/hook pivot.",
        ffmpeg_filter_hint="xfade=transition=slideleft:duration=0.22",
        parameters={"duration_sec": 0.22, "sfx_preset_id": "whoosh_transition"},
    ),
    "zoom_push": EffectPreset(
        preset_id="zoom_push",
        name="Smooth Zoom Push",
        category=EffectCategory.TRANSITION,
        description="Dynamic 1.0x -> 1.12x punch-in across emphasis cut.",
        editorial_purpose="Highlights key claim or transition into animated explanation.",
        ffmpeg_filter_hint="scale=iw*1.10:ih*1.10,crop=w:h",
        parameters={"duration_sec": 0.20, "zoom_scale": 1.10, "sfx_preset_id": "whoosh_transition"},
    ),
    "cross_dissolve": EffectPreset(
        preset_id="cross_dissolve",
        name="Clean Cross Dissolve",
        category=EffectCategory.TRANSITION,
        description="Smooth optical dissolve between scenes.",
        editorial_purpose="Maintains calm documentary/cinematic continuity across time shifts.",
        ffmpeg_filter_hint="xfade=transition=fade:duration=0.35",
        parameters={"duration_sec": 0.35},
    ),
    "dip_to_black": EffectPreset(
        preset_id="dip_to_black",
        name="Chapter Dip to Black",
        category=EffectCategory.TRANSITION,
        description="Brief 0.3s fade-through-black separating major chapters.",
        editorial_purpose="Creates breathing room before new chapter topics.",
        ffmpeg_filter_hint="xfade=transition=fadeblack:duration=0.35",
        parameters={"duration_sec": 0.35, "sfx_preset_id": "sub_impact"},
    ),
    "match_motion_cut": EffectPreset(
        preset_id="match_motion_cut",
        name="Match Motion Cut",
        category=EffectCategory.TRANSITION,
        description="Zero-dissolve frame cut paired with subtle scale alignment.",
        editorial_purpose="Keeps technical tutorials crisp without visual clutter.",
        ffmpeg_filter_hint="cut",
        parameters={"duration_sec": 0.0, "sfx_preset_id": "ui_pop"},
    ),
    "speed_ramp": EffectPreset(
        preset_id="speed_ramp",
        name="Editorial Speed Ramp",
        category=EffectCategory.TRANSITION,
        description="1.25x-1.6x acceleration over transitional setup motion.",
        editorial_purpose="Compresses dead-time visual movement while preserving context.",
        ffmpeg_filter_hint="setpts=0.75*PTS",
        parameters={"playback_speed": 1.35},
    ),
    "mask_wipe": EffectPreset(
        preset_id="mask_wipe",
        name="Geometric Mask Reveal",
        category=EffectCategory.TRANSITION,
        description="Directional wipe reveal into animation or diagram cutaway.",
        editorial_purpose="Introduces original simulations cleanly over talking head.",
        ffmpeg_filter_hint="xfade=transition=wiperight:duration=0.28",
        parameters={"duration_sec": 0.28, "sfx_preset_id": "whoosh_transition"},
    ),
    # Typography
    "kinetic_caption_pop": EffectPreset(
        preset_id="kinetic_caption_pop",
        name="Kinetic Word Pop Captions",
        category=EffectCategory.TYPOGRAPHY,
        description="High-contrast bold subtitles with active spoken-word color highlight.",
        editorial_purpose="Maximizes silent-scroll comprehension on mobile feeds.",
        ffmpeg_filter_hint="ass=karaoke_pop",
        parameters={"highlight_active_word": True, "font_size": 62},
    ),
    "word_highlight_karaoke": EffectPreset(
        preset_id="word_highlight_karaoke",
        name="Educational Word Highlight",
        category=EffectCategory.TYPOGRAPHY,
        description="Clean legible bottom/safe-zone subtitles with soft word emphasis.",
        editorial_purpose="Keeps technical vocabulary readable without blocking diagrams.",
        ffmpeg_filter_hint="ass=word_highlight",
        parameters={"highlight_active_word": True, "font_size": 52},
    ),
    "lower_third_editorial": EffectPreset(
        preset_id="lower_third_editorial",
        name="Editorial Lower-Third Headline",
        category=EffectCategory.TYPOGRAPHY,
        description="Minimal bar chapter label and concise transcript subtitle.",
        editorial_purpose="Documentary and technical tutorial polish.",
        ffmpeg_filter_hint="ass=editorial",
        parameters={"highlight_active_word": False, "font_size": 46},
    ),
    "glitch_title": EffectPreset(
        preset_id="glitch_title",
        name="Chromatic Glitch Headline",
        category=EffectCategory.TYPOGRAPHY,
        description="Short RGB split burst for unexpected contrast reveals.",
        editorial_purpose="Used sparingly for plot twists or bug discoveries.",
        ffmpeg_filter_hint="rgbashift=rh=4:bh=-4",
        parameters={"duration_sec": 0.25},
    ),
    "slide_reveal": EffectPreset(
        preset_id="slide_reveal",
        name="Slide Reveal Typography",
        category=EffectCategory.TYPOGRAPHY,
        description="Smooth alpha slide-in captions.",
        editorial_purpose="Warm personal vlog and storytelling aesthetics.",
        ffmpeg_filter_hint="ass=slide_reveal",
        parameters={"highlight_active_word": False, "font_size": 48},
    ),
    "perspective_callout": EffectPreset(
        preset_id="perspective_callout",
        name="Tracked Callout Badge",
        category=EffectCategory.TYPOGRAPHY,
        description="Pinned floating label highlighting diagram or speaker quote.",
        editorial_purpose="Anchors complex concepts directly in frame.",
        ffmpeg_filter_hint="overlay=callout",
        parameters={"duration_sec": 3.0},
    ),
    # Visual Enhancements
    "dynamic_face_reframe_9_16": EffectPreset(
        preset_id="dynamic_face_reframe_9_16",
        name="Face-Tracked 9:16 Dynamic Reframe",
        category=EffectCategory.VISUAL,
        description="Centers vertical crop around tracked speaker coordinates instead of blind center crop.",
        editorial_purpose="Keeps presenter naturally framed in Shorts, TikTok, and Reels.",
        ffmpeg_filter_hint="crop=ih*9/16:ih:x_expr:0,scale=1080:1920",
        parameters={"target_aspect": "9:16"},
    ),
    "clean_clarity": EffectPreset(
        preset_id="clean_clarity",
        name="Clean Studio Color Clarity",
        category=EffectCategory.VISUAL,
        description="Gentle S-curve contrast + subtle sharpening + crisp white balance.",
        editorial_purpose="Improves talking-head legibility without artificial look.",
        ffmpeg_filter_hint="eq=contrast=1.06:saturation=1.08:brightness=0.01,unsharp=3:3:0.35",
        parameters={},
    ),
    "punchy_contrast": EffectPreset(
        preset_id="punchy_contrast",
        name="Punchy YouTube Contrast Grade",
        category=EffectCategory.VISUAL,
        description="Vibrant saturation and dynamic midtone pop.",
        editorial_purpose="High-energy thumbnail/feed visibility.",
        ffmpeg_filter_hint="eq=contrast=1.11:saturation=1.16:brightness=0.02,unsharp=3:3:0.45",
        parameters={},
    ),
    "cinematic_teal_amber": EffectPreset(
        preset_id="cinematic_teal_amber",
        name="Cinematic Warm/Cool Grade + Vignette",
        category=EffectCategory.VISUAL,
        description="Subtle filmic tone balance with gentle edge vignette.",
        editorial_purpose="Focuses viewer attention on presenter during narrative beats.",
        ffmpeg_filter_hint="eq=contrast=1.08:saturation=1.05,vignette=PI/6",
        parameters={},
    ),
    "selective_bg_blur": EffectPreset(
        preset_id="selective_bg_blur",
        name="Split/Overlay Depth Dimming",
        category=EffectCategory.VISUAL,
        description="Softens talking-head layer behind transparent/PIP animation overlays.",
        editorial_purpose="Preserves visual hierarchy when equations or circuits appear.",
        ffmpeg_filter_hint="boxblur=4:1",
        parameters={"opacity": 0.85},
    ),
    "pip_corner_overlay": EffectPreset(
        preset_id="pip_corner_overlay",
        name="Speaker PIP + Animation Canvas",
        category=EffectCategory.VISUAL,
        description="Displays original animated explanation prominently while preserving presenter presence.",
        editorial_purpose="Combines human connection with clear technical visualization.",
        ffmpeg_filter_hint="overlay=W-w-40:H-h-140",
        parameters={},
    ),
    # Sound Effects & Audio Polish
    "whoosh_transition": EffectPreset(
        preset_id="whoosh_transition",
        name="Clean Air Whoosh",
        category=EffectCategory.SOUND,
        description="Synthesized stereo-balanced bandpass air whoosh for visual transitions.",
        editorial_purpose="Cues viewer attention at diagram entry/exit.",
        ffmpeg_filter_hint="adelay+amix",
        parameters={"gain_db": -12.0, "duration_sec": 0.48},
    ),
    "sub_impact": EffectPreset(
        preset_id="sub_impact",
        name="Sub Bass Impact Hit",
        category=EffectCategory.SOUND,
        description="Warm low-end punch accent for chapter starts and major reveals.",
        editorial_purpose="Anchors hook payoffs without ear fatigue.",
        ffmpeg_filter_hint="adelay+amix",
        parameters={"gain_db": -10.0, "duration_sec": 0.55},
    ),
    "riser_tension": EffectPreset(
        preset_id="riser_tension",
        name="Tension Pitch Riser",
        category=EffectCategory.SOUND,
        description="Smooth rising harmonic sweep leading into a key turning point.",
        editorial_purpose="Builds curiosity right before revealing answer/result.",
        ffmpeg_filter_hint="adelay+amix",
        parameters={"gain_db": -15.0, "duration_sec": 1.3},
    ),
    "ui_pop": EffectPreset(
        preset_id="ui_pop",
        name="Crisp UI Pop",
        category=EffectCategory.SOUND,
        description="Short tactile pop for code highlights, annotations, and timeline ticks.",
        editorial_purpose="Micro-feedback when steps advance in explainer graphics.",
        ffmpeg_filter_hint="adelay+amix",
        parameters={"gain_db": -14.0, "duration_sec": 0.18},
    ),
    "speech_loudness_polish": EffectPreset(
        preset_id="speech_loudness_polish",
        name="Broadcast Voice Chain (HPF + Compression + Loudnorm -14 LUFS)",
        category=EffectCategory.SOUND,
        description="80Hz rumble filter, light vocal leveling, and -14 LUFS integrated loudness target.",
        editorial_purpose="Consistent YouTube/TikTok/Reels dialogue clarity.",
        ffmpeg_filter_hint="highpass=f=80,acompressor=threshold=-20dB:ratio=2.5,loudnorm=I=-14:TP=-1.5:LRA=11",
        parameters={"target_lufs": -14.0},
    ),
}


def search_effects(query: str | None = None, category: str | None = None) -> list[EffectPreset]:
    items = list(EFFECTS_CATALOG.values())
    if category:
        items = [e for e in items if e.category.value == category]
    if query:
        q = query.lower().strip()
        items = [
            e
            for e in items
            if q in e.preset_id.lower()
            or q in e.name.lower()
            or q in e.description.lower()
            or q in e.editorial_purpose.lower()
        ]
    return items
