"""Timeline builder, OpenTimelineIO (.otio) exporter, multi-platform planner, and iterative agent editor."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import opentimelineio as otio

from symposium_studio.engines.analysis.styles import get_style_profile
from symposium_studio.engines.production.directors import (
    _overlap_ratio,
    _tight_speech_start,
    select_best_hook,
    select_closing_payoff,
    snap_ending_boundary,
)
from symposium_studio.models.animation import AnimationRenderResult
from symposium_studio.models.media import HighlightProposal, HighlightScoreBreakdown, MediaAnalysisBundle, SilenceInterval, TranscriptSegment
from symposium_studio.models.symposium import Storyboard
from symposium_studio.models.timeline import (
    AudioNode,
    CaptionCue,
    CaptionSettings,
    EffectCategory,
    EffectSpec,
    OverlayNode,
    RenderTimeline,
    SegmentTimelineMapping,
    TargetPlatformFormat,
    TransitionSpec,
    VideoClipNode,
)

_SENTENCE_END_PUNCT = (".", "!", "?", "…")
_CLAUSE_PUNCT = (",", ";", ":", "—", "–")


def _split_segment_into_short_cues(
    seg: TranscriptSegment,
    clip_src_start: float,
    clip_src_end: float,
    clip_tl_start: float,
    speed: float,
    max_words: int,
    style_preset: str,
    clip_id: str | None = None,
) -> list[CaptionCue]:
    """Map verbatim transcript words inside [clip_src_start, clip_src_end] onto output timeline timestamps."""
    words = [
        w
        for w in seg.words
        if clip_src_start <= (w.start_sec + w.end_sec) * 0.5 < clip_src_end
    ]
    if not words:
        if seg.words:
            # Rescue edge words that genuinely overlap >= 45% (or >= 140ms) inside this subclip;
            # if none do, the subclip only caught an internal breath/pause inside seg -> emit nothing.
            straddlers = [
                w
                for w in seg.words
                if (min(w.end_sec, clip_src_end) - max(w.start_sec, clip_src_start))
                >= max(0.14, 0.45 * max(0.04, w.end_sec - w.start_sec))
            ]
            if not straddlers:
                return []
            words = straddlers
        if seg.end_sec <= clip_src_start or seg.start_sec >= clip_src_end:
            return []
        if min(seg.end_sec, clip_src_end) - max(seg.start_sec, clip_src_start) < 0.28:
            return []
        ov_s = max(clip_src_start, seg.start_sec)
        ov_e = min(clip_src_end, seg.end_sec)
        s_rel = max(0.0, ov_s - clip_src_start) / speed
        e_rel = max(s_rel + 0.25, (ov_e - clip_src_start) / speed)
        return [
            CaptionCue(
                cue_id=f"cue_{uuid.uuid4().hex[:8]}",
                start_sec=round(clip_tl_start + s_rel, 3),
                end_sec=round(clip_tl_start + e_rel, 3),
                text=seg.text,
                low_confidence=seg.low_confidence,
                style_preset=style_preset,
                clip_id=clip_id,
                source_segment_id=seg.segment_id,
                source_start_sec=round(ov_s, 3),
                source_end_sec=round(ov_e, 3),
            )
        ]

    cues: list[CaptionCue] = []
    compact_burst = style_preset == "kinetic_caption_pop"
    chunk_size = min(3, max(2, max_words)) if compact_burst else min(4, max(2, max_words))
    gap_limit = 0.28 if compact_burst else 0.42
    cur_grp: list[Any] = []

    def _emit_grp(grp: list[Any]) -> None:
        if not grp:
            return
        src_s = float(grp[0].start_sec)
        src_e = min(clip_src_end, float(grp[-1].end_sec))
        s_rel = max(0.0, src_s - clip_src_start) / speed
        raw_end_rel = (src_e - clip_src_start) / speed
        max_span = max(2.6, 0.85 * len(grp)) if compact_burst else max(3.6, 1.05 * len(grp))
        e_rel = max(s_rel + 0.22, min(raw_end_rel, s_rel + max_span))
        mapped_words = []
        for w in grp:
            w_src_s = float(w.start_sec)
            w_src_e = min(clip_src_end, float(w.end_sec))
            w_tl_s = round(clip_tl_start + max(0.0, w_src_s - clip_src_start) / speed, 3)
            w_tl_e = round(clip_tl_start + min(e_rel, max(0.0, w_src_e - clip_src_start) / speed), 3)
            w_tl_e = round(max(w_tl_s + 0.04, w_tl_e), 3)
            mapped_words.append(
                {
                    "word": w.word,
                    "start_sec": w_tl_s,
                    "end_sec": w_tl_e,
                    "source_start_sec": round(w_src_s, 3),
                    "source_end_sec": round(max(w_src_s + 0.02, w_src_e), 3),
                    "probability": round(float(getattr(w, "probability", 0.95)), 3),
                }
            )
        cues.append(
            CaptionCue(
                cue_id=f"cue_{uuid.uuid4().hex[:8]}",
                start_sec=round(clip_tl_start + s_rel, 3),
                end_sec=round(clip_tl_start + e_rel, 3),
                text=" ".join(w.word for w in grp),
                words=mapped_words,
                low_confidence=seg.low_confidence,
                style_preset=style_preset,
                clip_id=clip_id,
                source_segment_id=seg.segment_id,
                source_start_sec=round(src_s, 3),
                source_end_sec=round(max(src_s + 0.04, src_e), 3),
            )
        )

    def _run_len_ahead(start_idx: int) -> int:
        cnt = 1
        while start_idx + cnt < len(words):
            prev_w = words[start_idx + cnt - 1]
            nxt_w = words[start_idx + cnt]
            if (nxt_w.start_sec - prev_w.end_sec) > gap_limit:
                break
            if prev_w.word.rstrip().endswith(_SENTENCE_END_PUNCT):
                break
            if prev_w.word.rstrip().endswith(_CLAUSE_PUNCT):
                break
            cnt += 1
        return cnt

    for i_w, w in enumerate(words):
        if cur_grp:
            prev_w = cur_grp[-1]
            prev_txt = prev_w.word.rstrip()
            gap_here = w.start_sec - prev_w.end_sec
            hit_sentence_end = prev_txt.endswith(_SENTENCE_END_PUNCT)
            hit_clause_end = len(cur_grp) >= 2 and prev_txt.endswith(_CLAUSE_PUNCT)
            # Balance 4-word uninterrupted phrases into 2 + 2 instead of 3 + 1 orphan word
            rem_run = _run_len_ahead(i_w)
            balance_2_2 = chunk_size == 3 and len(cur_grp) == 2 and rem_run == 2
            if (
                len(cur_grp) >= chunk_size
                or gap_here > gap_limit
                or hit_sentence_end
                or hit_clause_end
                or balance_2_2
            ):
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


def _build_segment_mappings(
    segments: list[TranscriptSegment],
    clips: list[VideoClipNode],
    captions: list[CaptionCue],
) -> list[SegmentTimelineMapping]:
    """Build explicit source-segment <-> output-timeline mappings ordered by edited clips."""
    mappings: list[SegmentTimelineMapping] = []
    for c in clips:
        spd = max(0.25, float(c.playback_speed))
        for seg in segments:
            ov_s = max(c.source_start_sec, seg.start_sec)
            ov_e = min(c.source_end_sec, seg.end_sec)
            if ov_e - ov_s <= 0.12:
                continue
            tl_s = round(c.timeline_start_sec + (ov_s - c.source_start_sec) / spd, 3)
            tl_e = round(c.timeline_start_sec + (ov_e - c.source_start_sec) / spd, 3)
            linked_cues = [
                cue.cue_id
                for cue in captions
                if (cue.clip_id == c.clip_id and cue.source_segment_id == seg.segment_id)
                or (
                    cue.clip_id is None
                    and min(cue.end_sec, tl_e) - max(cue.start_sec, tl_s) > 0.05
                )
            ]
            # Only require/include excerpt words actually intersecting this clip slice
            slice_words = [
                w.word
                for w in (seg.words or [])
                if ov_s <= 0.5 * (w.start_sec + w.end_sec) < ov_e
            ]
            # When word timestamps exist, only map the slice if actual spoken words overlap (avoid flagging inter-word pauses)
            excerpt = (
                " ".join(slice_words)
                if seg.words
                else (seg.text if ov_e - ov_s >= 0.45 else "")
            )
            if not excerpt.strip() and not linked_cues:
                continue
            mappings.append(
                SegmentTimelineMapping(
                    mapping_id=f"map_{uuid.uuid4().hex[:8]}",
                    source_segment_id=seg.segment_id,
                    clip_id=c.clip_id,
                    source_start_sec=round(ov_s, 3),
                    source_end_sec=round(ov_e, 3),
                    timeline_start_sec=tl_s,
                    timeline_end_sec=tl_e,
                    playback_speed=spd,
                    language=seg.language,
                    text_excerpt=excerpt[:160],
                    cue_ids=linked_cues,
                )
            )
    return mappings


def _build_timeline_captions(
    segments: list[TranscriptSegment],
    clips: list[VideoClipNode],
    max_words: int,
    style_preset: str,
) -> list[CaptionCue]:
    """Map transcript segments onto timeline clips without dropping words in uniform segments."""
    captions: list[CaptionCue] = []
    # Iterate clips in final timeline order so reordered clips always yield chronological captions
    for c in clips:
        for seg in segments:
            if min(c.source_end_sec, seg.end_sec) - max(c.source_start_sec, seg.start_sec) <= 0.12:
                continue
            ov_for_seg = [
                (cc, max(cc.source_start_sec, seg.start_sec), min(cc.source_end_sec, seg.end_sec))
                for cc in clips
                if min(cc.source_end_sec, seg.end_sec) - max(cc.source_start_sec, seg.start_sec) > 0.12
            ]
            seg_use = seg
            words = seg.words or []
            if len(words) >= 2 and ov_for_seg:
                active_dur = sum(oe - os for _, os, oe in ov_for_seg)
                seg_dur = max(0.05, seg.end_sec - seg.start_sec)
                continuous_synthetic = all(
                    (words[idx_w + 1].start_sec - words[idx_w].end_sec) <= 0.08
                    for idx_w in range(len(words) - 1)
                )
                covers_bounds = (ov_for_seg[0][1] - seg.start_sec) <= 3.8 and (seg.end_sec - ov_for_seg[-1][2]) <= 1.2
                any_dropped = any(
                    not any(os <= (w.start_sec + w.end_sec) / 2.0 <= oe for _, os, oe in ov_for_seg)
                    for w in words
                )
                if continuous_synthetic and covers_bounds and any_dropped and active_dur >= seg_dur * 0.25:
                    cum_spans: list[tuple[float, float, float, float]] = []
                    acc = 0.0
                    for _, os, oe in ov_for_seg:
                        dur = max(0.01, oe - os)
                        cum_spans.append((acc, acc + dur, os, oe))
                        acc += dur
                    weights = [max(2.0, float(len(w.word)) + 1.0) for w in words]
                    total_w = sum(weights) or float(len(words))
                    adjusted_words = []
                    w_acc = 0.0
                    for w, weight in zip(words, weights):
                        a0 = (w_acc / total_w) * acc
                        w_acc += weight
                        a1 = (w_acc / total_w) * acc
                        mid = (a0 + a1) * 0.5
                        s0, s1, os, oe = next(
                            (sp for sp in cum_spans if sp[0] <= mid <= sp[1]),
                            cum_spans[-1],
                        )
                        span_w = max(0.01, s1 - s0)
                        f0 = max(0.0, min(1.0, (a0 - s0) / span_w))
                        f1 = max(f0 + 0.02, min(1.0, (a1 - s0) / span_w))
                        ws = round(os + f0 * (oe - os), 3)
                        we = round(min(oe, max(ws + 0.04, os + f1 * (oe - os))), 3)
                        adjusted_words.append(w.model_copy(update={"start_sec": ws, "end_sec": we}))
                    seg_use = seg.model_copy(update={"words": adjusted_words})

            captions.extend(
                _split_segment_into_short_cues(
                    seg=seg_use,
                    clip_src_start=c.source_start_sec,
                    clip_src_end=c.source_end_sec,
                    clip_tl_start=c.timeline_start_sec,
                    speed=c.playback_speed,
                    max_words=max_words,
                    style_preset=style_preset,
                    clip_id=c.clip_id,
                )
            )
    return captions


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

    fast_zoomer = min_gap_sec <= 0.25
    offset_pad = 0.16 if fast_zoomer else 0.20
    onset_pad = 0.26 if fast_zoomer else 0.28
    raw_dead: list[tuple[float, float]] = []
    for sil in silences:
        if sil.duration_sec < min_gap_sec:
            continue
        ds = max(src_start, sil.start_sec + offset_pad)
        de = min(src_end, sil.end_sec - onset_pad)
        if de - ds >= 0.16:
            raw_dead.append((ds, de))

    # Also harvest explicit silent gaps between consecutive transcript segments on soft microphones
    seg_sorted = sorted(
        (s for s in segments if s.end_sec >= src_start and s.start_sec <= src_end),
        key=lambda s: s.start_sec,
    )
    for idx_s in range(len(seg_sorted) - 1):
        g0 = seg_sorted[idx_s].end_sec + offset_pad
        g1 = seg_sorted[idx_s + 1].start_sec - onset_pad
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

    # Break long takes (>~4.2s continuous talk) at natural segment & word-pause boundaries for ADHD/jump-cut rhythm
    max_take = max(4.2, target_shot_sec * 1.55) if min_gap_sec <= 0.25 else max(4.6, target_shot_sec * 1.65)
    min_piece = max(2.2, target_shot_sec * 0.95) if min_gap_sec <= 0.25 else max(2.4, target_shot_sec * 0.9)
    refined: list[tuple[float, float]] = []
    for st, en in spoken:
        if (en - st) <= max_take:
            refined.append((st, en))
            continue
        sub_cur = st
        cand_boundaries = [
            round(seg.end_sec + 0.04, 3)
            for seg in segments
            if st + 1.4 < seg.end_sec < en - 1.2
        ]
        if min_gap_sec <= 0.25:
            for seg in segments:
                if seg.end_sec < st or seg.start_sec > en or len(seg.words) < 4:
                    continue
                for wi in range(len(seg.words) - 1):
                    w0, w1 = seg.words[wi], seg.words[wi + 1]
                    if (w1.start_sec - w0.end_sec) >= 0.06 and st + 1.4 < w0.end_sec < en - 1.2:
                        cand_boundaries.append(round((w0.end_sec + w1.start_sec) * 0.5, 3))
        cand_boundaries = sorted(set(cand_boundaries))
        for b in cand_boundaries:
            if b - sub_cur >= min_piece:
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


def _face_boxes_for_window(
    analysis: MediaAnalysisBundle,
    src_start: float,
    src_end: float,
) -> list[tuple[float, float, float, float]]:
    """Collect normalized face/head/cap-brim boxes active within a clip window."""
    boxes: list[tuple[float, float, float, float]] = []
    for sample in analysis.visual_samples:
        if sample.time_sec < src_start - 0.5 or sample.time_sec > src_end + 0.5:
            continue
        for f in sample.faces:
            # Expand slightly upward to protect hat/cap brim & forehead headroom
            x0 = max(0.0, f.x_norm - 0.02)
            y0 = max(0.0, f.y_norm - 0.07)
            x1 = min(1.0, f.x_norm + f.w_norm + 0.02)
            y1 = min(1.0, f.y_norm + f.h_norm + 0.02)
            boxes.append((x0, y0, x1, y1))
    return boxes


def _box_overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _is_meme_visual_acceptable(anim: AnimationRenderResult) -> bool:
    """Reject broken/tiny assets and narrow vertical portrait text strips in video overlay slots."""
    if anim.adapter_used != "meme_cutaway":
        return True
    w, h = int(anim.width or 0), int(anim.height or 0)
    if w > 0 and h > 0:
        if min(w, h) < 200:
            return False
        if h > w * 1.45 and w < 500:
            return False
    return True


def _anim_beat_metadata(anim: AnimationRenderResult) -> tuple[float | None, float | None]:
    """Read target source beat bounds attached in SceneSpec.parameters, if present."""
    sp = Path(anim.scene_spec_path) if anim.scene_spec_path else None
    if sp is None or not sp.is_file():
        return None, None
    try:
        raw = json.loads(sp.read_text(encoding="utf-8"))
        params = raw.get("parameters") or {}
        bs = params.get("beat_source_start")
        be = params.get("beat_source_end")
        if bs is not None and be is not None:
            return float(bs), float(be)
    except Exception:
        pass
    return None, None


def _pick_non_occluding_rect(
    candidates: list[tuple[float, float, float, float, str]],
    face_boxes: list[tuple[float, float, float, float]] | None,
    prefer_left: bool,
) -> tuple[float, float, float, float, str]:
    if not face_boxes:
        return candidates[0] if prefer_left else candidates[-1]
    best = candidates[0]
    best_pen = float("inf")
    for idx, (cx, cy, cw, ch, mode) in enumerate(candidates):
        rect = (cx, cy, cx + cw, cy + ch)
        overlap = sum(_box_overlap_area(rect, fb) for fb in face_boxes)
        tie_bias = 0.0 if ((prefer_left and idx == 0) or (not prefer_left and idx == len(candidates) - 1)) else 0.0005
        pen = overlap + tie_bias
        if pen < best_pen:
            best_pen = pen
            best = (cx, cy, cw, ch, mode)
    return best


def _overlay_placement(
    is_vertical: bool,
    crop_x: float,
    is_meme: bool,
    slot_idx: int = 0,
    is_direct_meme_asset: bool = False,
    anim: AnimationRenderResult | None = None,
    face_boxes: list[tuple[float, float, float, float]] | None = None,
) -> tuple[float, float, float, float, str]:
    if is_meme and (is_direct_meme_asset or is_vertical):
        tall_strip = bool(anim and anim.width > 0 and anim.height > anim.width * 1.45 and anim.width < 500)
        # Full-frame blurred-backdrop reaction cutaway on Shorts or alternating YouTube slots
        if (is_vertical or slot_idx % 2 == 0) and not tall_strip:
            return 0.0, 0.0, 1.0, 1.0, "cutaway_blur_fill"
        # Pick the upper/mid corner with lowest face+cap occlusion
        corner_cands = [
            (0.025, 0.04, 0.22, 0.28, "corner_sticker_clean"),
            (0.75, 0.04, 0.22, 0.28, "corner_sticker_clean"),
            (0.025, 0.27, 0.21, 0.27, "corner_sticker_clean"),
            (0.76, 0.27, 0.21, 0.27, "corner_sticker_clean"),
        ]
        # Keep index 0=left, last=right for default deterministic parity when no face boxes
        return _pick_non_occluding_rect(
            [corner_cands[0], corner_cands[2], corner_cands[3], corner_cands[1]],
            face_boxes,
            prefer_left=crop_x > 0.55,
        )
    if is_vertical:
        if is_meme:
            return 0.0, 0.0, 1.0, 1.0, "cutaway_blur_fill"
        return 0.04, 0.055, 0.92, 0.34, "upper_stage_split"
    if is_meme:
        pip_cands = [
            (0.03, 0.05, 0.30, 0.38, "pip_top_left"),
            (0.67, 0.05, 0.30, 0.38, "pip_top_right"),
        ]
        return _pick_non_occluding_rect(pip_cands, face_boxes, prefer_left=crop_x > 0.55)
    pip_std = [
        (0.03, 0.06, 0.44, 0.54, "pip_top_left"),
        (0.49, 0.07, 0.48, 0.58, "pip_top_right"),
    ]
    return _pick_non_occluding_rect(pip_std, face_boxes, prefer_left=crop_x > 0.55)


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
            if is_zoomer:
                phase = subclip_counter % 6
                if phase == 0:
                    zoom = 1.01
                    clip_effects.append(
                        EffectSpec(
                            effect_id=f"fx_{uuid.uuid4().hex[:6]}",
                            preset_id="smooth_push_in",
                            category=EffectCategory.VISUAL,
                            start_sec=0.0,
                            duration_sec=dur,
                            parameters={"zoom_delta": 0.09},
                            editorial_purpose="Fast push-in ADHD pacing opener",
                        )
                    )
                elif phase == 1:
                    zoom = 1.0 + 0.18 * min(1.0, style.zoom_punch_intensity)
                elif phase == 2:
                    zoom = 1.04
                    clip_effects.append(
                        EffectSpec(
                            effect_id=f"fx_{uuid.uuid4().hex[:6]}",
                            preset_id="smooth_push_in",
                            category=EffectCategory.VISUAL,
                            start_sec=0.0,
                            duration_sec=dur,
                            parameters={"zoom_delta": 0.08},
                            editorial_purpose="Mid-beat zoom punch build-up",
                        )
                    )
                elif phase == 3:
                    zoom = 1.0 + 0.22 * min(1.0, style.zoom_punch_intensity)
                elif phase == 4:
                    zoom = 1.08
                else:
                    zoom = 1.0 + 0.15 * min(1.0, style.zoom_punch_intensity)
            elif tight_cuts:
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

    # Place original animation(s) / meme pop-ups onto their matched beat clips (or clean evenly-spaced slots)
    usable_anims = [a for a in anim_pool if _is_meme_visual_acceptable(a)]
    if usable_anims and clips:
        multi_mode = len(usable_anims) > 1 or tight_cuts
        if not multi_mode:
            target_idx = next(
                (i for i, b in enumerate(storyboard.beats) if b.animation_template is not None),
                min(1, len(clips) - 1),
            )
            target_clip = clips[min(target_idx, len(clips) - 1)]
            anim = usable_anims[0]
            dur_c = target_clip.timeline_end_sec - target_clip.timeline_start_sec
            ov_start = round(target_clip.timeline_start_sec + 0.25, 3)
            ov_end = round(min(target_clip.timeline_end_sec - 0.1, ov_start + min(max(2.0, dur_c - 0.3), anim.duration_sec)), 3)
            if ov_end > ov_start + 0.8:
                is_meme = _is_meme_overlay(anim)
                is_direct = anim.adapter_used == "meme_cutaway" or anim.output_path.lower().endswith((".mp4", ".gif", ".png", ".jpg", ".webp"))
                faces = _face_boxes_for_window(analysis, target_clip.source_start_sec, target_clip.source_end_sec)
                ox, oy, ow, oh, omode = _overlay_placement(
                    is_vertical,
                    target_clip.crop_center_x,
                    is_meme,
                    slot_idx=0,
                    is_direct_meme_asset=is_direct,
                    anim=anim,
                    face_boxes=faces,
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
                        label="Meme Cutaway #1" if is_meme else f"Original Animation ({anim.adapter_used})",
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
            max_ov_count = 10 if is_zoomer else 4
            n_items = min(max_ov_count, len(usable_anims))
            if n_items == 4 and not is_zoomer:
                fracs = [0.08, 0.33, 0.58, 0.82]
            elif n_items == 3:
                fracs = [0.08, 0.42, 0.74]
            elif n_items == 2:
                fracs = [0.12, 0.56]
            elif n_items > 1:
                fracs = [round(0.05 + 0.88 * (i / (n_items - 1)), 3) for i in range(n_items)]
            else:
                fracs = [0.18]
            min_gap_ov = 1.65 if is_zoomer else 3.0
            last_ov_end = -min_gap_ov
            meme_sfx_cycle = [k for k in ("vine_boom", "ui_pop", "pop_click", "camera_shutter") if k in sfx_map]
            for a_i, anim in enumerate(usable_anims[:n_items]):
                b_src_s, b_src_e = _anim_beat_metadata(anim)
                matched_clip: VideoClipNode | None = None
                if b_src_s is not None and b_src_e is not None:
                    best_ov = 0.0
                    for c in clips:
                        ov_sec = max(0.0, min(c.source_end_sec, b_src_e) - max(c.source_start_sec, b_src_s))
                        if ov_sec > best_ov:
                            best_ov = ov_sec
                            matched_clip = c
                    if best_ov < 0.3:
                        matched_clip = None

                if matched_clip is not None:
                    host_clip = matched_clip
                    target_t = host_clip.timeline_start_sec + min(
                        0.85 if is_zoomer else 1.0,
                        max(0.20, (host_clip.timeline_end_sec - host_clip.timeline_start_sec) * 0.25),
                    )
                else:
                    target_t = total_tl * fracs[a_i]
                    host_clip = next((c for c in clips if c.timeline_start_sec <= target_t <= c.timeline_end_sec), clips[-1])

                is_meme = _is_meme_overlay(anim) or is_zoomer
                is_direct = anim.adapter_used == "meme_cutaway" or anim.output_path.lower().endswith((".mp4", ".gif", ".png", ".jpg", ".webp"))
                faces = _face_boxes_for_window(analysis, host_clip.source_start_sec, host_clip.source_end_sec)
                ox, oy, ow, oh, omode = _overlay_placement(
                    is_vertical,
                    host_clip.crop_center_x,
                    is_meme,
                    slot_idx=a_i,
                    is_direct_meme_asset=is_direct,
                    anim=anim,
                    face_boxes=faces,
                )
                desired_dur = (
                    (1.38 if is_zoomer else 1.75)
                    if omode == "cutaway_blur_fill"
                    else min(2.05 if is_zoomer else 2.7, max(1.35 if is_zoomer else 1.6, anim.duration_sec))
                )
                ov_start = round(max(0.75 if is_zoomer else 0.85, last_ov_end + min_gap_ov, target_t), 3)
                ov_end = round(min(total_tl - 0.55, ov_start + desired_dur), 3)
                if ov_end <= ov_start + 0.85:
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
                if is_meme and meme_sfx_cycle:
                    sfx_key = "vine_boom" if (a_i == 0 and "vine_boom" in sfx_map) else meme_sfx_cycle[a_i % len(meme_sfx_cycle)]
                else:
                    sfx_key = "whoosh_transition"
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

    # Platform-specific safe caption settings (keep concise 2-3 word natural bursts)
    cap_settings = CaptionSettings(
        enabled=True,
        font_name="Verdana" if is_zoomer else "Helvetica Neue",
        font_size=(64 if is_zoomer else 56) if is_vertical else (56 if is_zoomer else 48),
        scale_factor=1.0,
        safe_margin_bottom_px=290 if platform_format == TargetPlatformFormat.TIKTOK else (270 if is_vertical else 92),
        safe_margin_top_px=120 if is_vertical else 50,
        highlight_active_word=True,
        max_words_per_line=3,
    )
    captions = _build_timeline_captions(
        analysis.transcript.segments,
        clips,
        max_words=cap_settings.max_words_per_line,
        style_preset=style.caption_preset,
    )
    captions = sorted(captions, key=lambda c: (c.start_sec, c.end_sec))
    segment_mappings = _build_segment_mappings(analysis.transcript.segments, clips, captions)

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
        captions=captions,
        segment_mappings=segment_mappings,
        caption_settings=cap_settings,
        edit_notes=[
            f"Built from Storyboard {storyboard.storyboard_id} using '{style.display_name}' profile.",
            f"Target platform: {platform_format.value} ({width}x{height} @ {fps}fps).",
        ],
    )
    if any(k in sfx_map for k in CLICK_SFX_PRIORITY):
        tl, _ = add_click_sounds_to_timeline(tl, sfx_map, style_name=style.name.value, replace_existing=True, bump_revision=False)
    return tl


CLICK_SFX_PRIORITY = (
    "mouse_click",
    "mechanical_click",
    "camera_shutter",
    "pop_click",
    "switch_click",
    "typewriter_key",
    "mouse_double_click",
)
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
    spacing = float(min_spacing_sec) if min_spacing_sec is not None else (0.85 if fast_edit else 3.2)

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
            pref_in = "pop_click" if o_i % 2 == 0 else "mouse_click"
            candidates.append((1, round(s0, 3), "overlay_entry", pref_in, -11.5))
        if s1 - s0 >= 2.4:
            mid_t = round(s0 + min(1.4, (s1 - s0) * 0.45), 3)
            candidates.append((2, mid_t, "overlay_emphasis", "mouse_double_click", -12.5))
        pref_out = "switch_click" if o_i % 2 == 0 else "camera_shutter"
        candidates.append((1, round(max(0.5, s1 - 0.05), 3), "overlay_dismiss", pref_out, -12.0))

    zoom_keys = ("camera_shutter", "mouse_click", "mechanical_click")
    cut_keys = ("mechanical_click", "typewriter_key", "mouse_click")
    cap_keys = ("typewriter_key", "pop_click", "mechanical_click")

    for c_i, clip in enumerate(tl.clips):
        if c_i == 0 or clip.timeline_start_sec < 0.55:
            continue
        t_cut = round(float(clip.timeline_start_sec), 3)
        if clip.zoom_scale >= 1.03:
            candidates.append((1, t_cut, "punch_zoom_cut", zoom_keys[c_i % len(zoom_keys)], -12.0))
        elif fast_edit or clip.transition_in.type == "cut":
            candidates.append((2, t_cut, "jump_cut", cut_keys[c_i % len(cut_keys)], -13.0))

    for cap_i, cue in enumerate(tl.captions):
        t_cap = round(float(cue.start_sec), 3)
        if t_cap < 0.65:
            continue
        txt = cue.text.strip()
        is_emph = bool(re.search(r"[?!%#]|\d+", txt)) or (fast_edit and (len(txt.split()) <= 2 or cap_i % 2 == 0))
        if is_emph:
            candidates.append((3, t_cap, "caption_emphasis", cap_keys[cap_i % len(cap_keys)], -13.5))

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
    total_dur = float(analysis.probe.duration_sec or 0.0)
    chosen: list[HighlightProposal] = []
    used_spans: list[tuple[float, float]] = []

    def _is_distinct_window(cand_s: float, cand_e: float) -> bool:
        for us, ue in used_spans:
            if abs(cand_s - us) < 3.5:
                return False
            if _overlap_ratio(cand_s, cand_e, us, ue) >= 0.45:
                return False
        return True

    def _snapped_highlight(
        hl: HighlightProposal,
        max_len: float = 42.0,
        max_ext: float = 7.0,
    ) -> HighlightProposal:
        s0 = max(0.0, hl.start_sec)
        raw_end = min(total_dur if total_dur > 0 else hl.end_sec, s0 + max_len)
        e_target = min(raw_end, max(hl.end_sec, s0 + 14.0))
        ns, ne, closure = snap_ending_boundary(analysis, s0, e_target, max_extend_sec=max_ext)
        if total_dur > 0:
            ne = min(total_dur, min(s0 + 46.0, max(ne, hl.end_sec)))
        hl_copy = hl.model_copy(deep=True)
        hl_copy.start_sec = round(ns, 3)
        hl_copy.end_sec = round(max(ns + 4.0, ne), 3)
        hl_copy.duration_sec = round(hl_copy.end_sec - hl_copy.start_sec, 2)
        if not closure.get("closure_ok", True):
            hl_copy.rationale = (
                f"{hl_copy.rationale} [closure: {closure.get('closure_type')}]".strip()
            )
        return hl_copy

    # 1. Variant 0 -> Editorial Hook Director choice (strongest self-contained opening)
    if analysis.transcript.segments and max_variants >= 1:
        best_hook = select_best_hook(analysis)
        h_s = float(best_hook["start_sec"])
        h_e = float(best_hook["end_sec"])
        matched_hl = next(
            (
                h
                for h in ranked
                if h.highlight_id == best_hook.get("highlight_id")
                or _overlap_ratio(h.start_sec, h.end_sec, h_s, h_e) >= 0.6
            ),
            None,
        )
        if matched_hl is not None:
            cand_hl = matched_hl.model_copy(deep=True)
            cand_hl.start_sec = round(h_s, 3)
            cand_hl.end_sec = round(max(h_e, matched_hl.end_sec), 3)
        else:
            cand_hl = HighlightProposal(
                highlight_id=str(best_hook.get("highlight_id") or "hl_hook_00"),
                start_sec=round(h_s, 3),
                end_sec=round(min(total_dur if total_dur > 0 else h_e, max(h_e, h_s + 16.0)), 3),
                duration_sec=round(max(4.0, h_e - h_s), 2),
                role="hook",
                title="Opening Hook",
                transcript_excerpt=str(best_hook.get("excerpt") or ""),
                scores=HighlightScoreBreakdown(
                    information_density=0.78,
                    novelty=0.80,
                    story_relevance=0.90,
                    audience_curiosity=0.85,
                    educational_value=0.72,
                    emotional_intensity=0.70,
                    visual_opportunity=0.75,
                    context_dependency=0.92,
                    composite_score=0.82,
                ),
            )
        snapped_hook = _snapped_highlight(cand_hl, max_len=32.0, max_ext=8.0)
        chosen.append(snapped_hook)
        used_spans.append((snapped_hook.start_sec, snapped_hook.end_sec))

    # 2. Reserve Closing Payoff variant for final slot when max_variants >= 3
    payoff_hl: HighlightProposal | None = None
    if max_variants >= 3 and analysis.transcript.segments:
        payoff_hl = select_closing_payoff(analysis, used_spans)
        if payoff_hl is not None and _is_distinct_window(payoff_hl.start_sec, payoff_hl.end_sec):
            payoff_hl = _snapped_highlight(payoff_hl, max_len=36.0, max_ext=8.0)
            used_spans.append((payoff_hl.start_sec, payoff_hl.end_sec))
        else:
            payoff_hl = None

    # 3. Fill remaining middle slot(s) from non-overlapping top highlights
    needed_middle = max(0, max_variants - len(chosen) - (1 if payoff_hl is not None else 0))
    for h in ranked:
        if len([c for c in chosen]) >= max_variants - (1 if payoff_hl is not None else 0):
            break
        if needed_middle <= 0:
            break
        if _is_distinct_window(h.start_sec, h.end_sec):
            snapped = _snapped_highlight(h, max_len=38.0, max_ext=6.0)
            if _is_distinct_window(snapped.start_sec, snapped.end_sec):
                chosen.append(snapped)
                used_spans.append((snapped.start_sec, snapped.end_sec))
                needed_middle -= 1

    if payoff_hl is not None:
        chosen.append(payoff_hl)

    # Relaxed fill from remaining highlights if strict non-overlap left fewer than max_variants
    if len(chosen) < max_variants:
        existing_starts = [round(c.start_sec, 1) for c in chosen]
        for h in ranked:
            if len(chosen) >= max_variants:
                break
            if all(abs(round(h.start_sec, 1) - es) >= 2.0 for es in existing_starts):
                chosen.append(_snapped_highlight(h, max_len=36.0, max_ext=5.0))
                existing_starts.append(round(h.start_sec, 1))

    # If video is longer than ~24s and fewer than max_variants distinct highlights exist, partition clean non-overlapping slices
    if len(chosen) < max_variants and total_dur >= 24.0:
        seg_len = min(35.0, max(12.0, total_dur / max_variants))
        existing_starts = [round(h.start_sec, 1) for h in chosen]
        for slot in range(max_variants):
            if len(chosen) >= max_variants:
                break
            st = round(slot * (total_dur / max_variants), 2)
            en = round(min(total_dur, st + seg_len), 2)
            if any(abs(st - es) < 4.0 for es in existing_starts) or en - st < 8.0:
                continue
            slice_hl = HighlightProposal(
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
            chosen.append(_snapped_highlight(slice_hl, max_len=36.0, max_ext=6.0))
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
        # Composite 2-3 snappy memes per variant in zoomer mode, or standard single visual otherwise
        short_extras: list[AnimationRenderResult] | None = None
        if style_name == "zoomer_vlog" and anim_pool:
            chosen_anim = anim_pool[v_idx % len(anim_pool)]
            short_extras = [anim_pool[(v_idx + j) % len(anim_pool)] for j in range(1, min(3, len(anim_pool)))]
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
            extra_overlays=short_extras,
        )
        # If tight silence cuts shrank a vertical Short under ~12s on a longer source video, expand and re-snap cleanly
        tl_dur = tl.clips[-1].timeline_end_sec if tl.clips else 0.0
        if tl_dur < 12.0 and total_dur >= 18.0:
            cur_span = max(1.0, sb.beats[0].source_end_sec - sb.beats[0].source_start_sec)
            ext_span = min(40.0, max(24.0, cur_span * (15.0 / max(3.5, tl_dur))))
            exp_start = sb.beats[0].source_start_sec
            exp_end = min(total_dur, exp_start + ext_span)
            if exp_end - exp_start < ext_span:
                exp_start = max(0.0, exp_end - ext_span)
            exp_start, exp_end, _ = snap_ending_boundary(analysis, exp_start, exp_end, max_extend_sec=6.0)
            sb.beats[0].source_start_sec = round(exp_start, 3)
            sb.beats[0].source_end_sec = round(min(total_dur, exp_end), 3)
            sb.beats[0].target_duration_sec = round(min(42.0, max(14.0, (sb.beats[0].source_end_sec - sb.beats[0].source_start_sec) * 0.65)), 2)
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
                extra_overlays=short_extras,
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


def _find_clip_for_time(t: float, clips: list[VideoClipNode]) -> VideoClipNode | None:
    for c in clips:
        if c.timeline_start_sec - 0.05 <= t <= c.timeline_end_sec + 0.05:
            return c
    if not clips:
        return None
    return min(clips, key=lambda c: min(abs(t - c.timeline_start_sec), abs(t - c.timeline_end_sec)))


def remap_edited_timeline(
    timeline: RenderTimeline,
    previous_clips: list[VideoClipNode] | None = None,
) -> RenderTimeline:
    """Recompute output timeline bounds, captions, words, overlays, SFX, and segment_mappings after clip edits.

    Never reuses stale/unadjusted source or pre-edit output timestamps after cutting, trimming,
    speed changes, or reordering clips.
    """
    tl = timeline.model_copy(deep=True)
    ref_clips = [c.model_copy(deep=True) for c in (previous_clips if previous_clips is not None else tl.clips)]
    old_by_id = {c.clip_id: c for c in ref_clips}

    # 1. Recompute contiguous output positions from current clip order & source bounds
    cur = 0.0
    for clip in tl.clips:
        spd = max(0.25, float(clip.playback_speed))
        clip.timeline_start_sec = round(cur, 3)
        dur = round(max(0.08, (float(clip.source_end_sec) - float(clip.source_start_sec)) / spd), 3)
        clip.timeline_end_sec = round(cur + dur, 3)
        cur += dur
    new_by_id = {c.clip_id: c for c in tl.clips}

    def _map_old_time(t: float, host_old: VideoClipNode | None) -> float:
        if host_old is None:
            return float(t)
        host_new = new_by_id.get(host_old.clip_id)
        if host_new is None:
            return float(t)
        old_dur = max(0.05, float(host_old.timeline_end_sec) - float(host_old.timeline_start_sec))
        new_dur = max(0.05, float(host_new.timeline_end_sec) - float(host_new.timeline_start_sec))
        # Use exact source mapping if speed changed or head trimmed cleanly
        src_trim_shift = (float(host_new.source_start_sec) - float(host_old.source_start_sec)) / max(0.25, float(host_new.playback_speed))
        src_pos = float(host_old.source_start_sec) + max(0.0, float(t) - float(host_old.timeline_start_sec)) * float(host_old.playback_speed)
        if abs(src_trim_shift) > 0.01:
            rel_s = max(0.0, min(new_dur, (src_pos - float(host_new.source_start_sec)) / max(0.25, float(host_new.playback_speed))))
            return float(host_new.timeline_start_sec) + rel_s
        ratio = new_dur / old_dur
        return float(host_new.timeline_start_sec) + max(0.0, min(old_dur, float(t) - float(host_old.timeline_start_sec))) * ratio

    # 2. Remap all captions & word timestamps directly from surviving edited clips
    remapped_caps: list[CaptionCue] = []
    for cap in tl.captions:
        host_new = new_by_id.get(cap.clip_id or "")
        host_old = old_by_id.get(cap.clip_id or "")
        if host_new is None and host_old is None and ref_clips:
            host_old = _find_clip_for_time(0.5 * (cap.start_sec + cap.end_sec), ref_clips)
            if host_old is not None:
                host_new = new_by_id.get(host_old.clip_id)
        if host_new is None:
            continue

        spd = max(0.25, float(host_new.playback_speed))
        if cap.source_start_sec is not None and cap.source_end_sec is not None:
            # Anchor strictly to source-relative intervals clipped to edited host clip
            c_src_s = float(cap.source_start_sec)
            c_src_e = float(cap.source_end_sec)
            if c_src_e <= host_new.source_start_sec + 0.04 or c_src_s >= host_new.source_end_sec - 0.04:
                continue
            ov_s = max(float(host_new.source_start_sec), c_src_s)
            ov_e = min(float(host_new.source_end_sec), c_src_e)
            if ov_e - ov_s < 0.08:
                continue

            new_words: list[dict[str, Any]] = []
            for w in cap.words:
                if "source_start_sec" in w and "source_end_sec" in w:
                    ws0 = float(w["source_start_sec"])
                    we0 = float(w["source_end_sec"])
                    w_mid = 0.5 * (ws0 + we0)
                    ov_w = min(we0, float(host_new.source_end_sec)) - max(ws0, float(host_new.source_start_sec))
                    if not (host_new.source_start_sec <= w_mid <= host_new.source_end_sec or ov_w >= 0.45 * max(0.04, we0 - ws0)):
                        continue
                    ws_cl = max(float(host_new.source_start_sec), ws0)
                    we_cl = min(float(host_new.source_end_sec), max(ws_cl + 0.02, we0))
                    tl_ws = round(host_new.timeline_start_sec + (ws_cl - host_new.source_start_sec) / spd, 3)
                    tl_we = round(max(tl_ws + 0.04, host_new.timeline_start_sec + (we_cl - host_new.source_start_sec) / spd), 3)
                else:
                    tl_ws = round(_map_old_time(float(w.get("start_sec", cap.start_sec)), host_old), 3)
                    tl_we = round(max(tl_ws + 0.04, _map_old_time(float(w.get("end_sec", cap.end_sec)), host_old)), 3)
                wd = dict(w)
                wd["start_sec"] = tl_ws
                wd["end_sec"] = tl_we
                new_words.append(wd)

            if cap.words and not new_words:
                continue
            new_cap = cap.model_copy(deep=True)
            new_cap.clip_id = host_new.clip_id
            new_cap.source_start_sec = round(ov_s, 3)
            new_cap.source_end_sec = round(ov_e, 3)
            if new_words:
                new_cap.words = new_words
                new_cap.text = " ".join(str(w["word"]) for w in new_words)
                new_cap.start_sec = float(new_words[0]["start_sec"])
                new_cap.end_sec = round(max(new_cap.start_sec + 0.12, float(new_words[-1]["end_sec"])), 3)
            else:
                new_cap.start_sec = round(host_new.timeline_start_sec + (ov_s - host_new.source_start_sec) / spd, 3)
                new_cap.end_sec = round(max(new_cap.start_sec + 0.14, host_new.timeline_start_sec + (ov_e - host_new.source_start_sec) / spd), 3)
            new_cap.aligned_to_final_audio = False
            remapped_caps.append(new_cap)
        else:
            # Proportionally remap legacy/manual cues + their word dicts
            ns = round(_map_old_time(cap.start_sec, host_old), 3)
            ne = round(max(ns + 0.14, _map_old_time(cap.end_sec, host_old)), 3)
            new_words = []
            first_ws = float(cap.words[0].get("start_sec", 0.0)) if cap.words else 0.0
            last_we = float(cap.words[-1].get("end_sec", 0.0)) if cap.words else 0.0
            is_rel = cap.words and cap.start_sec > 0.35 and first_ws < 0.20 and last_we <= (cap.end_sec - cap.start_sec) + 0.20
            scale = (ne - ns) / max(1e-4, cap.end_sec - cap.start_sec)
            for w in cap.words:
                wd = dict(w)
                if is_rel:
                    ws = round(float(w.get("start_sec", 0.0)) * scale, 3)
                    we = round(max(ws + 0.04, float(w.get("end_sec", 0.1)) * scale), 3)
                else:
                    ws = round(_map_old_time(float(w.get("start_sec", cap.start_sec)), host_old), 3)
                    we = round(max(ws + 0.04, _map_old_time(float(w.get("end_sec", cap.end_sec)), host_old)), 3)
                wd["start_sec"] = ws
                wd["end_sec"] = we
                new_words.append(wd)
            new_cap = cap.model_copy(deep=True)
            new_cap.clip_id = host_new.clip_id
            new_cap.start_sec = ns
            new_cap.end_sec = ne
            new_cap.words = new_words
            new_cap.aligned_to_final_audio = False
            remapped_caps.append(new_cap)

    remapped_caps.sort(key=lambda c: (c.start_sec, c.end_sec))
    tl.captions = remapped_caps

    # 3. Remap overlays and audio events proportionally onto moved/edited clips
    total_dur = max(0.2, tl.total_duration_sec)
    if ref_clips:
        kept_overlays: list[OverlayNode] = []
        for ov in tl.overlays:
            host_old = _find_clip_for_time(0.5 * (ov.timeline_start_sec + ov.timeline_end_sec), ref_clips)
            if host_old is not None and host_old.clip_id in new_by_id:
                ov_dur = max(0.35, ov.timeline_end_sec - ov.timeline_start_sec)
                host_new = new_by_id[host_old.clip_id]
                ns = round(max(0.0, min(total_dur - 0.35, _map_old_time(ov.timeline_start_sec, host_old))), 3)
                ne = round(min(host_new.timeline_end_sec + 0.5, total_dur, ns + ov_dur), 3)
                if ne > ns + 0.25:
                    ov2 = ov.model_copy(deep=True)
                    ov2.timeline_start_sec = ns
                    ov2.timeline_end_sec = ne
                    kept_overlays.append(ov2)
        tl.overlays = kept_overlays

        kept_audio: list[AudioNode] = []
        for au in tl.audio_events:
            if au.role == "music":
                au2 = au.model_copy(deep=True)
                au2.duration_sec = round(total_dur, 3)
                kept_audio.append(au2)
            else:
                host_old = _find_clip_for_time(au.timeline_start_sec, ref_clips)
                if host_old is not None and host_old.clip_id in new_by_id:
                    au2 = au.model_copy(deep=True)
                    au2.timeline_start_sec = round(max(0.0, min(total_dur - 0.05, _map_old_time(au.timeline_start_sec, host_old))), 3)
                    kept_audio.append(au2)
        tl.audio_events = kept_audio

    # 4. Rebuild / remap explicit segment_mappings cleanly
    new_mappings: list[SegmentTimelineMapping] = []
    for m in tl.segment_mappings:
        host_new = new_by_id.get(m.clip_id)
        if host_new is None:
            continue
        ov_s = max(float(host_new.source_start_sec), float(m.source_start_sec))
        ov_e = min(float(host_new.source_end_sec), float(m.source_end_sec))
        if ov_e - ov_s <= 0.12:
            continue
        spd = max(0.25, float(host_new.playback_speed))
        tl_s = round(host_new.timeline_start_sec + (ov_s - host_new.source_start_sec) / spd, 3)
        tl_e = round(host_new.timeline_start_sec + (ov_e - host_new.source_start_sec) / spd, 3)
        linked_cues = [
            c.cue_id
            for c in tl.captions
            if (c.clip_id == host_new.clip_id and c.source_segment_id == m.source_segment_id)
            or (min(c.end_sec, tl_e) - max(c.start_sec, tl_s) > 0.05)
        ]
        m2 = m.model_copy(deep=True)
        m2.source_start_sec = round(ov_s, 3)
        m2.source_end_sec = round(ov_e, 3)
        m2.timeline_start_sec = tl_s
        m2.timeline_end_sec = tl_e
        m2.playback_speed = spd
        m2.cue_ids = linked_cues
        new_mappings.append(m2)
    # Keep mappings ordered cleanly by final timeline order
    new_mappings.sort(key=lambda m: (m.timeline_start_sec, m.timeline_end_sec))
    tl.segment_mappings = new_mappings
    return tl


def reorder_timeline_clips(
    timeline: RenderTimeline,
    ordered_clip_ids: list[str],
) -> RenderTimeline:
    """Reorder or subset clips cleanly while recomputing all output subtitle, overlay, and mapping positions."""
    old_clips = [c.model_copy(deep=True) for c in timeline.clips]
    clip_map = {c.clip_id: c.model_copy(deep=True) for c in timeline.clips}
    new_clips = [clip_map[cid] for cid in ordered_clip_ids if cid in clip_map]
    tl = timeline.model_copy(deep=True)
    tl.revision += 1
    tl.clips = new_clips
    return remap_edited_timeline(tl, previous_clips=old_clips)


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
            old_clips = [c.model_copy(deep=True) for c in tl.clips]
            tl.clips[0].zoom_scale = round(min(1.18, tl.clips[0].zoom_scale + 0.07), 2)
            tl.clips[0].label = f"[Hook Boosted] {tl.clips[0].label}"
            # Trim 0.35s leading slack and propagate shift through captions, words, overlays & mappings
            if tl.clips[0].source_end_sec - tl.clips[0].source_start_sec > 4.0:
                tl.clips[0].source_start_sec = round(tl.clips[0].source_start_sec + 0.35, 3)
                tl = remap_edited_timeline(tl, previous_clips=old_clips)
            changes.append("Boosted opening clip punch-in zoom, tightened leading slack, and resynced timeline captions.")

    if ("reverse" in low or "reorder" in low) and "clip" in low and len(tl.clips) >= 2:
        rev_ids = [c.clip_id for c in reversed(tl.clips)]
        tl = reorder_timeline_clips(tl, rev_ids)
        tl.revision -= 1  # Avoid double-increment inside apply_iterative_edit
        changes.append("Reordered clips cleanly and recomputed output timestamps and mappings.")

    if "disable caption" in low or "remove caption" in low:
        tl.caption_settings.enabled = False
        changes.append("Disabled burned-in captions.")

    if "speed up" in low or "faster" in low or "1.1x" in low or "slow down" in low or "slower" in low:
        factor = 0.94 if ("slow down" in low or "slower" in low) else 1.12
        old_clips = [c.model_copy(deep=True) for c in tl.clips]
        for clip in tl.clips:
            clip.playback_speed = round(max(0.75, min(1.5, clip.playback_speed * factor)), 3)
        tl = remap_edited_timeline(tl, previous_clips=old_clips)
        changes.append(f"Adjusted clip playback speeds by {factor}x and resynced captions, words, overlays, and mappings.")

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
