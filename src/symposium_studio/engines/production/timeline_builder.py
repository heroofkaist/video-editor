"""Timeline builder, OpenTimelineIO (.otio) exporter, multi-platform planner, and iterative agent editor."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import opentimelineio as otio

from symposium_studio.engines.analysis.styles import get_style_profile
from symposium_studio.models.animation import AnimationRenderResult
from symposium_studio.models.media import MediaAnalysisBundle, SilenceInterval, TranscriptSegment
from symposium_studio.models.symposium import Storyboard
from symposium_studio.models.timeline import (
    AudioNode,
    CaptionCue,
    CaptionSettings,
    EffectCategory,
    EffectSpec,
    OverlayNode,
    RenderTimeline,
    TargetPlatformFormat,
    TransitionSpec,
    VideoClipNode,
)


def _split_segment_into_short_cues(
    seg: TranscriptSegment,
    clip_src_start: float,
    clip_src_end: float,
    clip_tl_start: float,
    speed: float,
    max_words: int,
    style_preset: str,
) -> list[CaptionCue]:
    """Map verbatim transcript words inside [clip_src_start, clip_src_end] onto output timeline timestamps."""
    words = [
        w
        for w in seg.words
        if clip_src_start <= (w.start_sec + w.end_sec) * 0.5 < clip_src_end
    ]
    if not words:
        if seg.words:
            # Avoid emitting a duplicate whole-segment caption when subword midpoints fell cleanly into neighboring subclips
            if seg.end_sec <= clip_src_start + 0.08 or seg.start_sec >= clip_src_end - 0.08:
                return []
            if any(clip_src_start <= (w.start_sec + w.end_sec) * 0.5 < clip_src_end for w in seg.words):
                return []
        if seg.end_sec <= clip_src_start or seg.start_sec >= clip_src_end:
            return []
        if min(seg.end_sec, clip_src_end) - max(seg.start_sec, clip_src_start) < 0.28:
            return []
        s_rel = max(0.0, seg.start_sec - clip_src_start) / speed
        e_rel = max(s_rel + 0.25, (min(clip_src_end, seg.end_sec) - clip_src_start) / speed)
        return [
            CaptionCue(
                cue_id=f"cue_{uuid.uuid4().hex[:8]}",
                start_sec=round(clip_tl_start + s_rel, 3),
                end_sec=round(clip_tl_start + e_rel, 3),
                text=seg.text,
                low_confidence=seg.low_confidence,
                style_preset=style_preset,
            )
        ]

    cues: list[CaptionCue] = []
    compact_burst = style_preset == "kinetic_caption_pop"
    chunk_size = max(2, max_words) if compact_burst else max(3, max_words * 2)
    cur_grp: list[Any] = []

    def _emit_grp(grp: list[Any]) -> None:
        if not grp:
            return
        s_rel = max(0.0, grp[0].start_sec - clip_src_start) / speed
        raw_end_rel = (min(clip_src_end, grp[-1].end_sec) - clip_src_start) / speed
        max_span = max(1.8, 0.58 * len(grp)) if compact_burst else max(2.8, 0.75 * len(grp))
        e_rel = max(s_rel + 0.22, min(raw_end_rel, s_rel + max_span))
        mapped_words = [
            {
                "word": w.word,
                "start_sec": round(clip_tl_start + max(0.0, w.start_sec - clip_src_start) / speed, 3),
                "end_sec": round(
                    clip_tl_start
                    + min(e_rel, max(0.0, min(clip_src_end, w.end_sec) - clip_src_start) / speed),
                    3,
                ),
            }
            for w in grp
        ]
        cues.append(
            CaptionCue(
                cue_id=f"cue_{uuid.uuid4().hex[:8]}",
                start_sec=round(clip_tl_start + s_rel, 3),
                end_sec=round(clip_tl_start + e_rel, 3),
                text=" ".join(w.word for w in grp),
                words=mapped_words,
                low_confidence=seg.low_confidence,
                style_preset=style_preset,
            )
        )

    for w in words:
        if cur_grp:
            gap_here = w.start_sec - cur_grp[-1].end_sec
            if len(cur_grp) >= chunk_size or (compact_burst and gap_here > 0.55):
                _emit_grp(cur_grp)
                cur_grp = []
        cur_grp.append(w)
    _emit_grp(cur_grp)
    return cues


def _crop_x_for_window(analysis: MediaAnalysisBundle, start_sec: float, end_sec: float) -> float:
    in_win = [
        s.suggested_crop_center_x
        for s in analysis.visual_samples
        if start_sec <= s.time_sec <= end_sec and s.faces
    ]
    if in_win:
        return round(sum(in_win) / len(in_win), 4)
    return analysis.recommended_crop_center_x


def _split_beat_around_silences(
    src_start: float,
    src_end: float,
    silences: list[SilenceInterval],
    segments: list[TranscriptSegment],
    min_gap_sec: float,
    target_shot_sec: float,
    tight_cuts: bool,
) -> list[tuple[float, float]]:
    """Strip dead silence pauses and gently subdivide long continuous takes on natural segment breaks."""
    if not tight_cuts or (src_end - src_start) <= 2.4:
        return [(src_start, src_end)]

    pad = 0.10
    raw_dead: list[tuple[float, float]] = []
    for sil in silences:
        if sil.duration_sec < min_gap_sec:
            continue
        ds = max(src_start, sil.start_sec + pad)
        de = min(src_end, sil.end_sec - pad)
        if de - ds >= 0.16:
            raw_dead.append((ds, de))

    # Also harvest explicit silent gaps between consecutive transcript segments on soft microphones
    seg_sorted = sorted(
        (s for s in segments if s.end_sec >= src_start and s.start_sec <= src_end),
        key=lambda s: s.start_sec,
    )
    for idx_s in range(len(seg_sorted) - 1):
        g0 = seg_sorted[idx_s].end_sec + pad
        g1 = seg_sorted[idx_s + 1].start_sec - pad
        if g1 - g0 >= max(0.32, min_gap_sec):
            ds = max(src_start, g0)
            de = min(src_end, g1)
            if de - ds >= 0.18:
                raw_dead.append((ds, de))

    raw_dead.sort()
    dead_spans: list[tuple[float, float]] = []
    for ds, de in raw_dead:
        if dead_spans and ds <= dead_spans[-1][1] + 0.05:
            dead_spans[-1] = (dead_spans[-1][0], max(dead_spans[-1][1], de))
        else:
            dead_spans.append((ds, de))

    spoken: list[tuple[float, float]] = []
    cur = src_start
    for ds, de in dead_spans:
        if ds > cur + 1.1:
            spoken.append((round(cur, 3), round(ds, 3)))
        cur = max(cur, de)
    if src_end - cur >= 1.1:
        spoken.append((round(cur, 3), round(src_end, 3)))
    if not spoken:
        return [(src_start, src_end)]

    # Gently break long takes ONLY at clean transcript segment boundaries for dynamic jump-cut rhythm
    max_take = max(4.6, target_shot_sec * 1.65)
    refined: list[tuple[float, float]] = []
    for st, en in spoken:
        if (en - st) <= max_take:
            refined.append((st, en))
            continue
        sub_cur = st
        cand_boundaries = sorted(
            round(seg.end_sec + 0.04, 3)
            for seg in segments
            if st + 1.8 < seg.end_sec < en - 1.4
        )
        for b in cand_boundaries:
            if b - sub_cur >= max(2.4, target_shot_sec * 0.9):
                refined.append((round(sub_cur, 3), b))
                sub_cur = b
        if en - sub_cur >= 1.1:
            refined.append((round(sub_cur, 3), en))
        elif refined:
            prev_s, _ = refined[-1]
            refined[-1] = (prev_s, en)
        else:
            refined.append((st, en))
    return refined or [(src_start, src_end)]


def _is_meme_overlay(anim: AnimationRenderResult) -> bool:
    if anim.adapter_used in {"meme_cutaway", "meme_reaction"}:
        return True
    hint = f"{anim.output_path} {anim.scene_spec_path}".lower()
    if "meme" in hint:
        return True
    try:
        spec_p = Path(anim.scene_spec_path)
        if spec_p.is_file():
            txt = spec_p.read_text(encoding="utf-8", errors="ignore")
            if "meme_reaction_card" in txt or "meme_cutaway" in txt:
                return True
    except Exception:
        pass
    return False


def _overlay_placement(
    is_vertical: bool,
    crop_x: float,
    is_meme: bool,
    slot_idx: int = 0,
    is_direct_meme_asset: bool = False,
) -> tuple[float, float, float, float, str]:
    if is_meme and (is_direct_meme_asset or is_vertical):
        # Full-frame blurred-backdrop reaction cutaway on Shorts (never covers face!) or alternating YouTube slots
        if is_vertical or slot_idx % 2 == 0:
            return 0.0, 0.0, 1.0, 1.0, "cutaway_blur_fill"
        if crop_x > 0.55:
            return 0.025, 0.04, 0.22, 0.29, "corner_sticker_clean"
        return 0.75, 0.04, 0.22, 0.29, "corner_sticker_clean"
    if is_vertical:
        if is_meme:
            return 0.0, 0.0, 1.0, 1.0, "cutaway_blur_fill"
        return 0.04, 0.055, 0.92, 0.34, "upper_stage_split"
    if crop_x > 0.55:
        if is_meme:
            return 0.03, 0.05, 0.31, 0.40, "pip_top_left"
        return 0.03, 0.06, 0.44, 0.54, "pip_top_left"
    if is_meme:
        return 0.66, 0.05, 0.31, 0.40, "pip_top_right"
    return 0.49, 0.07, 0.48, 0.58, "pip_top_right"


def build_timeline_from_storyboard(
    project_id: str,
    title: str,
    analysis: MediaAnalysisBundle,
    storyboard: Storyboard,
    style_name: str = "educational_explainer",
    animation_result: AnimationRenderResult | None = None,
    platform_format: TargetPlatformFormat = TargetPlatformFormat.YOUTUBE_LONG,
    sfx_paths: dict[str, str] | None = None,
    extra_overlays: list[AnimationRenderResult] | None = None,
) -> RenderTimeline:
    style = get_style_profile(style_name)
    is_vertical = platform_format != TargetPlatformFormat.YOUTUBE_LONG
    width, height = (1080, 1920) if is_vertical else (1920, 1080)
    fps = round(min(30.0, max(24.0, analysis.probe.fps or 30.0)), 3)
    tight_cuts = style.name.value in {"zoomer_vlog", "short_form_viral", "comedy"} or style.max_silence_gap_sec <= 0.35
    is_zoomer = style.name.value == "zoomer_vlog"

    anim_pool: list[AnimationRenderResult] = []
    seen_anim_paths: set[str] = set()
    for cand_anim in ([animation_result] if animation_result else []) + list(extra_overlays or []):
        if cand_anim is not None and cand_anim.output_path not in seen_anim_paths:
            seen_anim_paths.add(cand_anim.output_path)
            anim_pool.append(cand_anim)

    clips: list[VideoClipNode] = []
    overlays: list[OverlayNode] = []
    audio_events: list[AudioNode] = []
    captions: list[CaptionCue] = []
    cur_t = 0.0
    sfx_map = sfx_paths or {}
    subclip_counter = 0

    for idx, beat in enumerate(storyboard.beats):
        src_start = max(0.0, beat.source_start_sec)
        src_end = min(analysis.probe.duration_sec, max(src_start + 1.0, beat.source_end_sec))
        raw_dur = round(src_end - src_start, 3)
        spd = 1.0
        if beat.target_duration_sec > 0.5 and abs(raw_dur - beat.target_duration_sec) / max(1.0, raw_dur) > 0.03:
            spd = round(max(0.85, min(1.30, raw_dur / beat.target_duration_sec)), 3)
        if is_zoomer:
            spd = round(max(1.03, min(1.12, spd if spd > 1.0 else 1.06)), 3)

        sub_spans = _split_beat_around_silences(
            src_start=src_start,
            src_end=src_end,
            silences=analysis.silences,
            segments=analysis.transcript.segments,
            min_gap_sec=style.max_silence_gap_sec,
            target_shot_sec=style.target_avg_shot_sec,
            tight_cuts=tight_cuts,
        )

        for sub_i, (sub_s, sub_e) in enumerate(sub_spans):
            dur = round((sub_e - sub_s) / spd, 3)
            if dur < 0.55:
                continue
            tl_start = round(cur_t, 3)
            tl_end = round(cur_t + dur, 3)

            clip_effects: list[EffectSpec] = []
            if tight_cuts:
                phase = subclip_counter % 4
                if phase == 0:
                    zoom = 1.02
                    clip_effects.append(
                        EffectSpec(
                            effect_id=f"fx_{uuid.uuid4().hex[:6]}",
                            preset_id="smooth_push_in",
                            category=EffectCategory.VISUAL,
                            start_sec=0.0,
                            duration_sec=dur,
                            parameters={"zoom_delta": 0.07},
                            editorial_purpose="Subtle smooth push-in motion on wide talking-head beat",
                        )
                    )
                elif phase == 1:
                    zoom = 1.0 + 0.15 * min(1.0, style.zoom_punch_intensity)
                elif phase == 2:
                    zoom = 1.0
                    clip_effects.append(
                        EffectSpec(
                            effect_id=f"fx_{uuid.uuid4().hex[:6]}",
                            preset_id="smooth_push_in",
                            category=EffectCategory.VISUAL,
                            start_sec=0.0,
                            duration_sec=dur,
                            parameters={"zoom_delta": 0.055},
                            editorial_purpose="Subtle breathing push-in before next snap cut",
                        )
                    )
                else:
                    zoom = 1.0 + 0.11 * min(1.0, style.zoom_punch_intensity)
            else:
                zoom = 1.0 + (
                    0.07 * style.zoom_punch_intensity
                    if (idx == 0 or beat.role == "turning_point") and sub_i == 0
                    else 0.0
                )

            trans_type = "cut" if subclip_counter == 0 else style.transition_style
            trans_dur = 0.0 if trans_type == "cut" or subclip_counter == 0 else 0.18
            clip = VideoClipNode(
                clip_id=f"clip_{subclip_counter:02d}",
                source_path=analysis.probe.path,
                source_start_sec=sub_s,
                source_end_sec=sub_e,
                timeline_start_sec=tl_start,
                timeline_end_sec=tl_end,
                playback_speed=spd,
                crop_center_x=_crop_x_for_window(analysis, sub_s, sub_e),
                zoom_scale=round(zoom, 3),
                label=f"{beat.role}: {beat.visual_treatment}",
                transition_in=TransitionSpec(type=trans_type, duration_sec=trans_dur),
                effects=clip_effects,
            )
            clips.append(clip)

            # Map verbatim captions inside trimmed sub-clip onto output timeline
            for seg in analysis.transcript.segments:
                captions.extend(
                    _split_segment_into_short_cues(
                        seg=seg,
                        clip_src_start=sub_s,
                        clip_src_end=sub_e,
                        clip_tl_start=tl_start,
                        speed=spd,
                        max_words=3 if (is_vertical and is_zoomer) else (4 if is_vertical or is_zoomer else 6),
                        style_preset=style.caption_preset,
                    )
                )

            if subclip_counter == 0 and ("vine_boom" in sfx_map or "sub_impact" in sfx_map):
                hook_sfx = sfx_map["vine_boom"] if is_zoomer and "vine_boom" in sfx_map else sfx_map.get("sub_impact", sfx_map.get("vine_boom"))
                if hook_sfx:
                    audio_events.append(
                        AudioNode(
                            audio_id=f"sfx_{uuid.uuid4().hex[:6]}",
                            source_path=hook_sfx,
                            role="sfx",
                            timeline_start_sec=0.05,
                            duration_sec=0.65,
                            gain_db=-9.5 if is_zoomer else -11.0,
                        )
                    )
            cur_t = tl_end
            subclip_counter += 1

    # Place original animation(s) / meme pop-ups onto distinct clips with safe non-overlapping pacing
    if anim_pool and clips:
        multi_mode = len(anim_pool) > 1 or tight_cuts
        if not multi_mode:
            target_idx = next(
                (i for i, b in enumerate(storyboard.beats) if b.animation_template is not None),
                min(1, len(clips) - 1),
            )
            target_clip = clips[min(target_idx, len(clips) - 1)]
            anim = anim_pool[0]
            dur_c = target_clip.timeline_end_sec - target_clip.timeline_start_sec
            ov_start = round(target_clip.timeline_start_sec + 0.25, 3)
            ov_end = round(min(target_clip.timeline_end_sec - 0.1, ov_start + min(max(2.0, dur_c - 0.3), anim.duration_sec)), 3)
            if ov_end > ov_start + 0.8:
                is_meme = _is_meme_overlay(anim)
                is_direct = anim.adapter_used == "meme_cutaway" or anim.output_path.lower().endswith((".mp4", ".gif", ".png", ".jpg", ".webp"))
                ox, oy, ow, oh, omode = _overlay_placement(
                    is_vertical, target_clip.crop_center_x, is_meme, slot_idx=0, is_direct_meme_asset=is_direct
                )
                overlays.append(
                    OverlayNode(
                        overlay_id=f"ov_{uuid.uuid4().hex[:6]}",
                        asset_path=anim.output_path,
                        scene_spec_path=anim.scene_spec_path,
                        timeline_start_sec=ov_start,
                        timeline_end_sec=ov_end,
                        mode=omode,
                        opacity=1.0,
                        x_norm=ox,
                        y_norm=oy,
                        w_norm=ow,
                        h_norm=oh,
                        label=f"Meme Cutaway #1" if is_meme else f"Original Animation ({anim.adapter_used})",
                    )
                )
                sfx_key = "vine_boom" if is_meme and "vine_boom" in sfx_map else (
                    "ui_pop" if is_meme and "ui_pop" in sfx_map else "whoosh_transition"
                )
                if sfx_key in sfx_map:
                    audio_events.append(
                        AudioNode(
                            audio_id=f"sfx_{uuid.uuid4().hex[:6]}",
                            source_path=sfx_map[sfx_key],
                            role="sfx",
                            timeline_start_sec=ov_start,
                            duration_sec=0.45,
                            gain_db=-10.0 if is_meme else -12.0,
                        )
                    )
        else:
            total_tl = max(1.0, cur_t)
            n_items = min(4, len(anim_pool))
            if n_items >= 4:
                fracs = [0.08, 0.33, 0.58, 0.82]
            elif n_items == 3:
                fracs = [0.08, 0.42, 0.74]
            elif n_items == 2:
                fracs = [0.12, 0.56]
            else:
                fracs = [0.18]
            last_ov_end = -4.0
            for a_i, anim in enumerate(anim_pool[:4]):
                target_t = total_tl * fracs[a_i]
                is_meme = _is_meme_overlay(anim) or is_zoomer
                is_direct = anim.adapter_used == "meme_cutaway" or anim.output_path.lower().endswith((".mp4", ".gif", ".png", ".jpg", ".webp"))
                host_clip = next((c for c in clips if c.timeline_start_sec <= target_t <= c.timeline_end_sec), clips[-1])
                ox, oy, ow, oh, omode = _overlay_placement(
                    is_vertical, host_clip.crop_center_x, is_meme, slot_idx=a_i, is_direct_meme_asset=is_direct
                )
                # Keep full-screen meme cutaways snappy (1.55-1.85s) so pacing pops; corner stickers up to 2.6s
                desired_dur = 1.75 if omode == "cutaway_blur_fill" else min(2.7, max(1.6, anim.duration_sec))
                ov_start = round(max(0.85, last_ov_end + 3.0, target_t), 3)
                ov_end = round(min(total_tl - 0.55, ov_start + desired_dur), 3)
                if ov_end <= ov_start + 0.9:
                    continue
                overlays.append(
                    OverlayNode(
                        overlay_id=f"ov_{uuid.uuid4().hex[:6]}",
                        asset_path=anim.output_path,
                        scene_spec_path=anim.scene_spec_path,
                        timeline_start_sec=ov_start,
                        timeline_end_sec=ov_end,
                        mode=omode,
                        opacity=1.0,
                        x_norm=ox,
                        y_norm=oy,
                        w_norm=ow,
                        h_norm=oh,
                        label=f"Trending Meme Cutaway #{a_i + 1}" if is_meme else f"Original Animation ({anim.adapter_used})",
                    )
                )
                sfx_key = (
                    "vine_boom"
                    if (is_meme and a_i == 0 and "vine_boom" in sfx_map)
                    else ("ui_pop" if (is_meme and "ui_pop" in sfx_map) else "whoosh_transition")
                )
                if sfx_key in sfx_map:
                    audio_events.append(
                        AudioNode(
                            audio_id=f"sfx_{uuid.uuid4().hex[:6]}",
                            source_path=sfx_map[sfx_key],
                            role="sfx",
                            timeline_start_sec=ov_start,
                            duration_sec=0.55 if sfx_key == "vine_boom" else 0.35,
                            gain_db=-9.5 if is_meme else -12.0,
                        )
                    )
                last_ov_end = ov_end

    # Attach subtle ambient music bed ducked automatically under speech when synthesized
    if "ambient_bed" in sfx_map and cur_t > 1.0:
        audio_events.append(
            AudioNode(
                audio_id=f"mus_{uuid.uuid4().hex[:6]}",
                source_path=sfx_map["ambient_bed"],
                role="music",
                timeline_start_sec=0.0,
                duration_sec=round(cur_t, 3),
                gain_db=-19.0 if is_zoomer else -20.0,
                duck_under_speech=True,
            )
        )

    # Platform-specific safe caption settings
    cap_settings = CaptionSettings(
        enabled=True,
        font_name="Verdana" if is_zoomer else "Helvetica Neue",
        font_size=(64 if is_zoomer else 56) if is_vertical else (56 if is_zoomer else 48),
        scale_factor=1.0,
        safe_margin_bottom_px=290 if platform_format == TargetPlatformFormat.TIKTOK else (270 if is_vertical else 92),
        safe_margin_top_px=120 if is_vertical else 50,
        highlight_active_word=True,
        max_words_per_line=3 if (is_vertical and is_zoomer) else (4 if is_vertical or is_zoomer else 6),
    )

    tl = RenderTimeline(
        timeline_id=f"tl_{uuid.uuid4().hex[:8]}",
        project_id=project_id,
        revision=1,
        title=title,
        platform_format=platform_format,
        width=width,
        height=height,
        fps=fps,
        style_profile=style.name.value,
        clips=clips,
        overlays=overlays,
        audio_events=audio_events,
        captions=sorted(captions, key=lambda c: c.start_sec),
        caption_settings=cap_settings,
        edit_notes=[
            f"Built from Storyboard {storyboard.storyboard_id} using '{style.display_name}' profile.",
            f"Target platform: {platform_format.value} ({width}x{height} @ {fps}fps).",
        ],
    )
    if any(k in sfx_map for k in ("mouse_click", "mouse_double_click", "mechanical_click", "switch_click")):
        tl, _ = add_click_sounds_to_timeline(tl, sfx_map, style_name=style.name.value, replace_existing=True, bump_revision=False)
    return tl


CLICK_SFX_PRIORITY = ("mouse_click", "mechanical_click", "switch_click", "mouse_double_click")
_EXCLUDED_NON_CLICK_SFX = {"ambient_bed", "vine_boom", "whoosh_transition", "sub_impact", "riser_tension"}


def add_click_sounds_to_timeline(
    timeline: RenderTimeline,
    click_paths: dict[str, str],
    style_name: str | None = None,
    min_spacing_sec: float | None = None,
    replace_existing: bool = True,
    bump_revision: bool = True,
) -> tuple[RenderTimeline, list[dict[str, Any]]]:
    """Place internet-downloaded click SFX at punch cuts, overlay entries/exits, and kinetic caption hooks."""
    available: dict[str, str] = {}
    for k, p in (click_paths or {}).items():
        if k in _EXCLUDED_NON_CLICK_SFX:
            continue
        if Path(p).is_file():
            available[k] = str(Path(p).resolve())
    if not available:
        return timeline.model_copy(deep=True), []

    tl = timeline.model_copy(deep=True)
    if bump_revision:
        tl.revision += 1
    if replace_existing:
        tl.audio_events = [a for a in tl.audio_events if not str(a.audio_id).startswith("clk_")]

    ordered_keys = [k for k in CLICK_SFX_PRIORITY if k in available] + [
        k for k in sorted(available) if k not in CLICK_SFX_PRIORITY
    ]
    eff_style = style_name or tl.style_profile or "educational_explainer"
    fast_edit = eff_style in {"zoomer_vlog", "short_form_viral", "comedy"}
    spacing = float(min_spacing_sec) if min_spacing_sec is not None else (2.0 if fast_edit else 3.6)

    tl_end = max((c.timeline_end_sec for c in tl.clips), default=0.0)
    conflict_times = [
        round(float(a.timeline_start_sec), 3)
        for a in tl.audio_events
        if a.role == "sfx" and not str(a.audio_id).startswith("clk_")
    ]

    # Collect prioritized candidate timestamps: (priority, timestamp, reason, preferred_key, gain_db)
    candidates: list[tuple[int, float, str, str, float]] = []

    for o_i, ov in enumerate(tl.overlays):
        s0 = float(ov.timeline_start_sec)
        s1 = float(ov.timeline_end_sec)
        if not any(abs(s0 - ct) <= 0.40 for ct in conflict_times):
            candidates.append((1, round(s0, 3), "overlay_entry", "mouse_click", -14.5))
        if s1 - s0 >= 2.6:
            mid_t = round(s0 + min(1.5, (s1 - s0) * 0.45), 3)
            candidates.append((2, mid_t, "overlay_emphasis", "mouse_double_click", -15.5))
        candidates.append((1, round(max(0.5, s1 - 0.05), 3), "overlay_dismiss", "switch_click", -15.5))

    for c_i, clip in enumerate(tl.clips):
        if c_i == 0 or clip.timeline_start_sec < 0.65:
            continue
        t_cut = round(float(clip.timeline_start_sec), 3)
        if clip.zoom_scale >= 1.03:
            candidates.append((1, t_cut, "punch_zoom_cut", "mouse_click", -15.0))
        elif fast_edit or clip.transition_in.type == "cut":
            pref = "mechanical_click" if c_i % 2 == 1 else "mouse_click"
            candidates.append((2, t_cut, "jump_cut", pref, -16.0))

    for cap_i, cue in enumerate(tl.captions):
        t_cap = round(float(cue.start_sec), 3)
        if t_cap < 0.75:
            continue
        txt = cue.text.strip()
        is_emph = bool(re.search(r"[?!%#]|\d+", txt)) or (fast_edit and len(txt.split()) <= 2 and cap_i % 3 == 0)
        if is_emph:
            candidates.append((3, t_cap, "caption_emphasis", "mechanical_click", -16.5))

    # Greedily select highest-priority non-overlapping clicks separated cleanly from existing SFX
    candidates.sort(key=lambda item: (item[0], item[1]))
    chosen_times: list[float] = []
    chosen_entries: list[tuple[float, str, str, float]] = []

    for _prio, ts, reason, pref_key, gain_db in candidates:
        if ts < 0.40 or (tl_end > 1.2 and ts > tl_end - 0.18):
            continue
        if any(abs(ts - ct) < 0.48 for ct in conflict_times):
            continue
        if any(abs(ts - ch) < spacing for ch in chosen_times):
            continue
        key = pref_key if pref_key in available else ordered_keys[len(chosen_entries) % len(ordered_keys)]
        chosen_times.append(ts)
        chosen_entries.append((ts, reason, key, gain_db))

    chosen_entries.sort(key=lambda x: x[0])
    placements: list[dict[str, Any]] = []
    for idx, (ts, reason, key, gain_db) in enumerate(chosen_entries):
        src_path = available[key]
        au = AudioNode(
            audio_id=f"clk_{idx:02d}_{uuid.uuid4().hex[:5]}",
            source_path=src_path,
            role="sfx",
            timeline_start_sec=round(ts, 3),
            duration_sec=0.22,
            gain_db=gain_db,
            duck_under_speech=False,
        )
        tl.audio_events.append(au)
        placements.append(
            {
                "audio_id": au.audio_id,
                "preset": key,
                "reason": reason,
                "timeline_start_sec": au.timeline_start_sec,
                "gain_db": gain_db,
                "source_path": src_path,
            }
        )

    tl.audio_events.sort(key=lambda a: a.timeline_start_sec)
    if placements:
        tl.edit_notes.append(
            f"Added {len(placements)} click SFX cues across punch cuts, overlays, and emphasis beats."
        )
    return tl, placements


def build_vertical_short_timelines(
    project_id: str,
    base_title: str,
    analysis: MediaAnalysisBundle,
    animation_result: AnimationRenderResult | None = None,
    platform_format: TargetPlatformFormat = TargetPlatformFormat.YOUTUBE_SHORTS,
    max_variants: int = 3,
    sfx_paths: dict[str, str] | None = None,
    style_name: str = "short_form_viral",
    extra_overlays: list[AnimationRenderResult] | None = None,
) -> list[RenderTimeline]:
    """Create genuinely distinct vertical 9:16 edit plans (Hook / Mechanism / Payoff) across the video."""
    ranked = sorted(
        analysis.highlights,
        key=lambda h: (h.scores.audience_curiosity + h.scores.context_dependency + h.scores.visual_opportunity),
        reverse=True,
    )
    chosen = ranked[:max_variants] if ranked else []

    # If video is longer than ~45s and fewer than max_variants distinct highlights exist, partition clean non-overlapping slices
    total_dur = float(analysis.probe.duration_sec or 0.0)
    if len(chosen) < max_variants and total_dur >= 24.0:
        from symposium_studio.models.media import HighlightProposal, HighlightScoreBreakdown

        seg_len = min(35.0, max(12.0, total_dur / max_variants))
        existing_starts = [round(h.start_sec, 1) for h in chosen]
        for slot in range(max_variants):
            if len(chosen) >= max_variants:
                break
            st = round(slot * (total_dur / max_variants), 2)
            en = round(min(total_dur, st + seg_len), 2)
            if any(abs(st - es) < 4.0 for es in existing_starts) or en - st < 8.0:
                continue
            chosen.append(
                HighlightProposal(
                    highlight_id=f"hl_slice_{slot}",
                    start_sec=st,
                    end_sec=en,
                    duration_sec=round(en - st, 2),
                    role="payoff" if slot == max_variants - 1 else "short_candidate",
                    title=f"Short Segment {slot + 1}",
                    transcript_excerpt="",
                    scores=HighlightScoreBreakdown(
                        information_density=0.75,
                        novelty=0.72,
                        story_relevance=0.78,
                        audience_curiosity=0.72,
                        educational_value=0.78,
                        emotional_intensity=0.65,
                        visual_opportunity=0.75,
                        context_dependency=0.78,
                        composite_score=0.75,
                    ),
                )
            )
            existing_starts.append(round(st, 1))

    role_labels = ["Hook Cut", "Mechanism Breakdown", "Payoff & Takeaway"]
    anim_pool: list[AnimationRenderResult] = []
    seen_out: set[str] = set()
    for a in ([animation_result] if animation_result else []) + list(extra_overlays or []):
        if a is not None and a.output_path not in seen_out:
            seen_out.add(a.output_path)
            anim_pool.append(a)

    out: list[RenderTimeline] = []
    for v_idx, hl in enumerate(chosen[:max_variants]):
        from symposium_studio.models.symposium import StoryboardBeat

        tag = role_labels[v_idx] if v_idx < len(role_labels) else f"Variant {v_idx + 1}"
        # Composite distinct meme/visual per variant in zoomer mode, or standard placement otherwise
        if style_name == "zoomer_vlog" and anim_pool:
            chosen_anim = anim_pool[v_idx % len(anim_pool)]
            attach_anim = True
        else:
            chosen_anim = animation_result
            attach_anim = chosen_anim is not None and (
                (v_idx == 0 and hl.visual_opportunity_type is not None)
                or (v_idx == 1 and chosen_anim is not None)
            )
        sb = Storyboard(
            storyboard_id=f"sb_short_{v_idx}_{uuid.uuid4().hex[:5]}",
            project_id=project_id,
            target_format=platform_format.value,
            selected_strategy=f"Short-Form {tag} ({platform_format.value})",
            target_duration_sec=min(48.0, max(12.0, hl.duration_sec)),
            beats=[
                StoryboardBeat(
                    beat_index=0,
                    role="hook" if v_idx == 0 else "animation_explainer" if attach_anim else "payoff",
                    source_start_sec=hl.start_sec,
                    source_end_sec=min(hl.end_sec, hl.start_sec + 45.0),
                    target_duration_sec=min(45.0, hl.duration_sec),
                    visual_treatment="face_tracked_vertical_with_animation",
                    animation_template=hl.visual_opportunity_type if attach_anim else None,
                    caption_directive="kinetic_caption_pop",
                    audio_directive="speech_loudness_polish+ducked_ambient_bed",
                    evidence_ids=[hl.highlight_id],
                )
            ],
        )
        tl = build_timeline_from_storyboard(
            project_id=project_id,
            title=f"{base_title} [{platform_format.value} #{v_idx + 1}: {tag}]",
            analysis=analysis,
            storyboard=sb,
            style_name=style_name,
            animation_result=chosen_anim if attach_anim else None,
            platform_format=platform_format,
            sfx_paths=sfx_paths,
        )
        # If tight silence cuts shrank a vertical Short under ~12s on a longer source video, expand its window naturally
        tl_dur = tl.clips[-1].timeline_end_sec if tl.clips else 0.0
        if tl_dur < 12.0 and total_dur >= 18.0:
            cur_span = max(1.0, sb.beats[0].source_end_sec - sb.beats[0].source_start_sec)
            ext_span = min(40.0, max(24.0, cur_span * (15.0 / max(3.5, tl_dur))))
            exp_start = sb.beats[0].source_start_sec
            exp_end = min(total_dur, exp_start + ext_span)
            if exp_end - exp_start < ext_span:
                exp_start = max(0.0, exp_end - ext_span)
            sb.beats[0].source_start_sec = round(exp_start, 3)
            sb.beats[0].source_end_sec = round(exp_end, 3)
            sb.beats[0].target_duration_sec = round(min(42.0, max(14.0, (exp_end - exp_start) * 0.65)), 2)
            sb.target_duration_sec = sb.beats[0].target_duration_sec
            tl = build_timeline_from_storyboard(
                project_id=project_id,
                title=f"{base_title} [{platform_format.value} #{v_idx + 1}: {tag}]",
                analysis=analysis,
                storyboard=sb,
                style_name=style_name,
                animation_result=chosen_anim if attach_anim else None,
                platform_format=platform_format,
                sfx_paths=sfx_paths,
            )
        out.append(tl)
    return out


def export_otio(timeline: RenderTimeline, out_path: Path) -> Path:
    """Serialize RenderTimeline to a valid OpenTimelineIO (.otio) file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = timeline.fps or 30.0
    otio_tl = otio.schema.Timeline(
        name=timeline.title,
        metadata={
            "symposium_timeline_id": timeline.timeline_id,
            "platform_format": timeline.platform_format.value,
            "revision": timeline.revision,
            "style_profile": timeline.style_profile,
        },
    )

    video_track = otio.schema.Track(name="V1_Main", kind=otio.schema.TrackKind.Video)
    for c in timeline.clips:
        src_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(c.source_start_sec * fps, fps),
            duration=otio.opentime.RationalTime((c.source_end_sec - c.source_start_sec) * fps, fps),
        )
        otio_clip = otio.schema.Clip(
            name=c.label or c.clip_id,
            media_reference=otio.schema.ExternalReference(target_url=Path(c.source_path).resolve().as_uri()),
            source_range=src_range,
            metadata={"crop_center_x": c.crop_center_x, "zoom_scale": c.zoom_scale},
        )
        video_track.append(otio_clip)
    otio_tl.tracks.append(video_track)

    if timeline.overlays:
        ov_track = otio.schema.Track(name="V2_Animations", kind=otio.schema.TrackKind.Video)
        cursor = 0.0
        for ov in timeline.overlays:
            if ov.timeline_start_sec > cursor + 0.01:
                ov_track.append(
                    otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, fps),
                            duration=otio.opentime.RationalTime((ov.timeline_start_sec - cursor) * fps, fps),
                        )
                    )
                )
            dur = ov.timeline_end_sec - ov.timeline_start_sec
            ov_track.append(
                otio.schema.Clip(
                    name=ov.label or ov.overlay_id,
                    media_reference=otio.schema.ExternalReference(target_url=Path(ov.asset_path).resolve().as_uri()),
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, fps),
                        duration=otio.opentime.RationalTime(dur * fps, fps),
                    ),
                    metadata={"mode": ov.mode, "scene_spec_path": ov.scene_spec_path},
                )
            )
            cursor = ov.timeline_end_sec
        otio_tl.tracks.append(ov_track)

    if timeline.audio_events:
        au_track = otio.schema.Track(name="A2_SFX", kind=otio.schema.TrackKind.Audio)
        cursor = 0.0
        for au in sorted(timeline.audio_events, key=lambda x: x.timeline_start_sec):
            if au.timeline_start_sec > cursor + 0.01:
                au_track.append(
                    otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, fps),
                            duration=otio.opentime.RationalTime((au.timeline_start_sec - cursor) * fps, fps),
                        )
                    )
                )
            au_track.append(
                otio.schema.Clip(
                    name=au.role,
                    media_reference=otio.schema.ExternalReference(target_url=Path(au.source_path).resolve().as_uri()),
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, fps),
                        duration=otio.opentime.RationalTime(au.duration_sec * fps, fps),
                    ),
                )
            )
            cursor = au.timeline_start_sec + au.duration_sec
        otio_tl.tracks.append(au_track)

    otio.adapters.write_to_file(otio_tl, str(out_path))
    return out_path


def apply_iterative_edit(timeline: RenderTimeline, instruction: str) -> tuple[RenderTimeline, list[str]]:
    """Apply structured iterative agent/user editing instructions to a RenderTimeline."""
    low = instruction.lower()
    tl = timeline.model_copy(deep=True)
    tl.revision += 1
    changes: list[str] = []

    if "caption" in low and ("small" in low or "reduce" in low or "shrink" in low):
        tl.caption_settings.scale_factor = round(max(0.6, tl.caption_settings.scale_factor * 0.8), 2)
        changes.append(f"Scaled captions down to factor {tl.caption_settings.scale_factor}.")
    elif "caption" in low and ("big" in low or "large" in low or "increas" in low):
        tl.caption_settings.scale_factor = round(min(1.5, tl.caption_settings.scale_factor * 1.2), 2)
        changes.append(f"Scaled captions up to factor {tl.caption_settings.scale_factor}.")

    if "opening" in low and ("engag" in low or "hook" in low or "punch" in low):
        if tl.clips:
            tl.clips[0].zoom_scale = round(min(1.18, tl.clips[0].zoom_scale + 0.07), 2)
            tl.clips[0].label = f"[Hook Boosted] {tl.clips[0].label}"
            # Trim 0.35s leading slack if clip is long enough
            if tl.clips[0].source_end_sec - tl.clips[0].source_start_sec > 4.0:
                tl.clips[0].source_start_sec = round(tl.clips[0].source_start_sec + 0.35, 3)
            changes.append("Boosted opening clip punch-in zoom and tightened leading slack for stronger 0-2s hook.")

    if "disable caption" in low or "remove caption" in low:
        tl.caption_settings.enabled = False
        changes.append("Disabled burned-in captions.")

    if "speed up" in low or "faster" in low or "1.1x" in low or "slow down" in low or "slower" in low:
        factor = 0.94 if ("slow down" in low or "slower" in low) else 1.12
        cur = 0.0
        for clip in tl.clips:
            old_dur = max(0.2, clip.timeline_end_sec - clip.timeline_start_sec)
            clip.playback_speed = round(max(0.75, min(1.5, clip.playback_speed * factor)), 3)
            new_dur = round((clip.source_end_sec - clip.source_start_sec) / clip.playback_speed, 3)
            old_start = clip.timeline_start_sec
            clip.timeline_start_sec = round(cur, 3)
            clip.timeline_end_sec = round(cur + new_dur, 3)
            ratio = new_dur / old_dur
            for cap in tl.captions:
                if cap.start_sec >= old_start - 0.01 and cap.start_sec < old_start + old_dur + 0.01:
                    cap.start_sec = round(cur + (cap.start_sec - old_start) * ratio, 3)
                    cap.end_sec = round(max(cap.start_sec + 0.2, cur + (cap.end_sec - old_start) * ratio), 3)
            cur += new_dur
        changes.append(f"Adjusted clip playback speeds by {factor}x and rescaled caption sync timestamps.")

    if "cross dissolve" in low or "dissolve" in low or "dip to black" in low or "dip black" in low:
        t_type = "dip_black" if "dip" in low else "cross_dissolve"
        for c_i, clip in enumerate(tl.clips):
            if c_i > 0:
                clip.transition_in = TransitionSpec(type=t_type, duration_sec=0.22)
        changes.append(f"Set segment transition_in to `{t_type}`.")

    if "remove overlay" in low or "disable overlay" in low:
        tl.overlays = []
        changes.append("Cleared all animation/graphic overlays.")

    if "mute music" in low or "no music" in low or "remove music" in low:
        tl.audio_events = [a for a in tl.audio_events if a.role != "music"]
        changes.append("Removed background music bed tracks.")

    if not changes:
        changes.append(f"Recorded revision note: {instruction}")

    tl.edit_notes.extend(changes)
    return tl, changes
