"""Timeline builder, OpenTimelineIO (.otio) exporter, multi-platform planner, and iterative agent editor."""

from __future__ import annotations

import uuid
from pathlib import Path

import opentimelineio as otio

from symposium_studio.engines.analysis.styles import get_style_profile
from symposium_studio.models.animation import AnimationRenderResult
from symposium_studio.models.media import MediaAnalysisBundle, TranscriptSegment
from symposium_studio.models.symposium import Storyboard
from symposium_studio.models.timeline import (
    AudioNode,
    CaptionCue,
    CaptionSettings,
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
        if w.end_sec > clip_src_start + 0.02 and w.start_sec < clip_src_end - 0.02
    ]
    if not words:
        if seg.end_sec <= clip_src_start or seg.start_sec >= clip_src_end:
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
    chunk_size = max(3, max_words * 2)
    for i in range(0, len(words), chunk_size):
        grp = words[i : i + chunk_size]
        s_rel = max(0.0, grp[0].start_sec - clip_src_start) / speed
        e_rel = max(s_rel + 0.22, (min(clip_src_end, grp[-1].end_sec) - clip_src_start) / speed)
        mapped_words = [
            {
                "word": w.word,
                "start_sec": round(clip_tl_start + max(0.0, w.start_sec - clip_src_start) / speed, 3),
                "end_sec": round(clip_tl_start + max(0.0, min(clip_src_end, w.end_sec) - clip_src_start) / speed, 3),
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
                low_confidence=seg.low_confidence or any(w.low_confidence for w in grp),
                style_preset=style_preset,
            )
        )
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


def build_timeline_from_storyboard(
    project_id: str,
    title: str,
    analysis: MediaAnalysisBundle,
    storyboard: Storyboard,
    style_name: str = "educational_explainer",
    animation_result: AnimationRenderResult | None = None,
    platform_format: TargetPlatformFormat = TargetPlatformFormat.YOUTUBE_LONG,
    sfx_paths: dict[str, str] | None = None,
) -> RenderTimeline:
    style = get_style_profile(style_name)
    is_vertical = platform_format != TargetPlatformFormat.YOUTUBE_LONG
    width, height = (1080, 1920) if is_vertical else (1920, 1080)
    fps = round(min(30.0, max(24.0, analysis.probe.fps or 30.0)), 3)

    clips: list[VideoClipNode] = []
    overlays: list[OverlayNode] = []
    audio_events: list[AudioNode] = []
    captions: list[CaptionCue] = []
    cur_t = 0.0
    sfx_map = sfx_paths or {}

    for idx, beat in enumerate(storyboard.beats):
        src_start = max(0.0, beat.source_start_sec)
        src_end = min(analysis.probe.duration_sec, max(src_start + 1.0, beat.source_end_sec))
        raw_dur = round(src_end - src_start, 3)
        spd = 1.0
        if beat.target_duration_sec > 0.5 and abs(raw_dur - beat.target_duration_sec) / max(1.0, raw_dur) > 0.03:
            spd = round(max(0.85, min(1.30, raw_dur / beat.target_duration_sec)), 3)
        dur = round(raw_dur / spd, 3)
        tl_start = round(cur_t, 3)
        tl_end = round(cur_t + dur, 3)

        zoom = 1.0 + (0.07 * style.zoom_punch_intensity if idx == 0 or beat.role == "turning_point" else 0.0)
        trans_type = "cut" if idx == 0 else style.transition_style
        trans_dur = 0.0 if trans_type == "cut" or idx == 0 else 0.18
        clip = VideoClipNode(
            clip_id=f"clip_{idx:02d}",
            source_path=analysis.probe.path,
            source_start_sec=src_start,
            source_end_sec=src_end,
            timeline_start_sec=tl_start,
            timeline_end_sec=tl_end,
            playback_speed=spd,
            crop_center_x=_crop_x_for_window(analysis, src_start, src_end),
            zoom_scale=round(zoom, 3),
            label=f"{beat.role}: {beat.visual_treatment}",
            transition_in=TransitionSpec(type=trans_type, duration_sec=trans_dur),
        )
        clips.append(clip)

        # Map verbatim captions into output timeline (synced to playback_speed)
        for seg in analysis.transcript.segments:
            captions.extend(
                _split_segment_into_short_cues(
                    seg=seg,
                    clip_src_start=src_start,
                    clip_src_end=src_end,
                    clip_tl_start=tl_start,
                    speed=spd,
                    max_words=4 if is_vertical else 6,
                    style_preset=style.caption_preset,
                )
            )

        # Attach original generated animation onto designated explainer beat (or beat 1 fallback)
        if animation_result is not None and (beat.animation_template is not None or (idx == 1 and not overlays)):
            ov_dur = min(dur - 0.3, animation_result.duration_sec) if dur > 2.0 else dur
            ov_start = round(tl_start + 0.25, 3)
            ov_end = round(min(tl_end - 0.1, ov_start + max(2.0, ov_dur)), 3)
            if ov_end > ov_start + 0.8:
                if is_vertical:
                    ov_x, ov_y, ov_w, ov_h = 0.04, 0.055, 0.92, 0.34
                elif clip.crop_center_x > 0.55:
                    ov_x, ov_y, ov_w, ov_h = 0.03, 0.07, 0.47, 0.58
                else:
                    ov_x, ov_y, ov_w, ov_h = 0.49, 0.07, 0.48, 0.58
                overlays.append(
                    OverlayNode(
                        overlay_id=f"ov_{uuid.uuid4().hex[:6]}",
                        asset_path=animation_result.output_path,
                        scene_spec_path=animation_result.scene_spec_path,
                        timeline_start_sec=ov_start,
                        timeline_end_sec=ov_end,
                        mode="upper_stage_split" if is_vertical else "pip_top_right",
                        opacity=1.0,
                        x_norm=ov_x,
                        y_norm=ov_y,
                        w_norm=ov_w,
                        h_norm=ov_h,
                        label=f"Original Animation ({animation_result.adapter_used})",
                    )
                )
                if "whoosh_transition" in sfx_map:
                    audio_events.append(
                        AudioNode(
                            audio_id=f"sfx_{uuid.uuid4().hex[:6]}",
                            source_path=sfx_map["whoosh_transition"],
                            role="sfx",
                            timeline_start_sec=ov_start,
                            duration_sec=0.5,
                            gain_db=-12.0,
                        )
                    )

        # Hook sub impact at opening
        if idx == 0 and "sub_impact" in sfx_map:
            audio_events.append(
                AudioNode(
                    audio_id=f"sfx_{uuid.uuid4().hex[:6]}",
                    source_path=sfx_map["sub_impact"],
                    role="sfx",
                    timeline_start_sec=0.05,
                    duration_sec=0.55,
                    gain_db=-11.0,
                )
            )
        cur_t = tl_end

    # Attach subtle ambient music bed ducked automatically under speech when synthesized
    if "ambient_bed" in sfx_map and cur_t > 1.0:
        audio_events.append(
            AudioNode(
                audio_id=f"mus_{uuid.uuid4().hex[:6]}",
                source_path=sfx_map["ambient_bed"],
                role="music",
                timeline_start_sec=0.0,
                duration_sec=round(cur_t, 3),
                gain_db=-20.0,
                duck_under_speech=True,
            )
        )

    # Platform-specific safe caption settings
    cap_settings = CaptionSettings(
        enabled=True,
        font_name="Helvetica Neue",
        font_size=56 if is_vertical else 48,
        scale_factor=1.0,
        safe_margin_bottom_px=280 if platform_format == TargetPlatformFormat.TIKTOK else (250 if is_vertical else 75),
        safe_margin_top_px=120 if is_vertical else 50,
        highlight_active_word=True,
        max_words_per_line=4 if is_vertical else 6,
    )

    return RenderTimeline(
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


def build_vertical_short_timelines(
    project_id: str,
    base_title: str,
    analysis: MediaAnalysisBundle,
    animation_result: AnimationRenderResult | None = None,
    platform_format: TargetPlatformFormat = TargetPlatformFormat.YOUTUBE_SHORTS,
    max_variants: int = 3,
    sfx_paths: dict[str, str] | None = None,
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
    out: list[RenderTimeline] = []
    for v_idx, hl in enumerate(chosen[:max_variants]):
        from symposium_studio.models.symposium import StoryboardBeat

        tag = role_labels[v_idx] if v_idx < len(role_labels) else f"Variant {v_idx + 1}"
        # Composite animation on either explicit visual highlight or Variant #1 (#0 is hook, #1 is mechanism)
        attach_anim = animation_result is not None and (
            (v_idx == 0 and hl.visual_opportunity_type is not None)
            or (v_idx == 1 and animation_result is not None)
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
            style_name="short_form_viral",
            animation_result=animation_result if attach_anim else None,
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
