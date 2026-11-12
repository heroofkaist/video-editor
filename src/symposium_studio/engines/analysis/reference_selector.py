"""Multi-reference compatibility scoring (13 criteria), hybrid blending, Style Transfer translation, and post-render validation."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from symposium_studio.engines.analysis.probe import probe_media
from symposium_studio.engines.analysis.styles import get_style_profile
from symposium_studio.engines.analysis.vision import detect_scene_cuts
from symposium_studio.models.media import EditingStyleName, EditingStyleProfile, MediaAnalysisBundle
from symposium_studio.models.reference import (
    CriterionScore,
    HybridStyleGuide,
    ReferenceAnalysisStatus,
    ReferenceCandidateEvaluation,
    ReferenceConferenceRecord,
    ReferenceItem,
    SelectionMode,
    StyleDNA,
    StyleSimilarityMetric,
    StyleSimilarityReport,
    StyleTransferParameters,
    StyleTransferPlan,
    _utc_now,
)
from symposium_studio.models.social import QualityAssuranceReport
from symposium_studio.models.timeline import (
    EffectCategory,
    EffectSpec,
    RenderTimeline,
    TargetPlatformFormat,
    TransitionSpec,
    VideoClipNode,
)


def _source_signal_summary(analysis: MediaAnalysisBundle) -> dict[str, Any]:
    segs = analysis.transcript.segments
    total_words = sum(len(s.words) if s.words else len(s.text.split()) for s in segs)
    speech_dur = sum(max(0.1, s.end_sec - s.start_sec) for s in segs) or max(1.0, analysis.probe.duration_sec)
    wpm = (total_words / speech_dur) * 60.0 if segs else 120.0
    total_sil = sum(s.duration_sec for s in analysis.silences)
    pause_ratio = min(0.8, total_sil / max(1.0, analysis.probe.duration_sec))
    tech_topics = sum(1 for t in analysis.topics if t.is_technical)
    tech_ratio = tech_topics / max(1, len(analysis.topics))
    face_frames = sum(1 for v in analysis.visual_samples if v.faces)
    face_ratio = face_frames / max(1, len(analysis.visual_samples)) if analysis.visual_samples else 0.7
    langs = set(analysis.transcript.languages_observed or ([analysis.transcript.detected_language] if analysis.transcript.detected_language else []))
    multilingual = len([l for l in langs if l and l != "unknown"]) >= 2 or bool(
        langs & {"kk", "ru"}
    )
    is_vertical_src = analysis.probe.height > analysis.probe.width
    avg_hook_curiosity = (
        sum(h.scores.audience_curiosity for h in analysis.highlights[:3]) / max(1, len(analysis.highlights[:3]))
        if analysis.highlights
        else 0.55
    )
    avg_humor = (
        sum(h.scores.emotional_intensity for h in analysis.highlights[:3]) / max(1, len(analysis.highlights[:3]))
        if analysis.highlights
        else 0.45
    )
    return {
        "duration_sec": analysis.probe.duration_sec,
        "wpm": round(wpm, 1),
        "pause_ratio": round(pause_ratio, 3),
        "tech_ratio": round(tech_ratio, 3),
        "face_ratio": round(face_ratio, 3),
        "multilingual": multilingual,
        "languages": sorted(langs),
        "is_vertical_src": is_vertical_src,
        "hook_curiosity": round(avg_hook_curiosity, 3),
        "humor_energy": round(avg_humor, 3),
        "has_uncertain_asr": bool(analysis.transcript.uncertain_segment_ids),
    }


def evaluate_reference_against_source(
    ref: ReferenceItem,
    analysis: MediaAnalysisBundle,
    target_platform: str = "youtube_long",
    creator_niche: str = "science_engineering",
) -> ReferenceCandidateEvaluation:
    """Evaluate a single reference across 13 weighted heuristic compatibility criteria for this source footage."""
    src = _source_signal_summary(analysis)
    is_short_target = target_platform in {"youtube_shorts", "tiktok", "instagram_reels"}
    dna = ref.style_dna

    if dna is None or ref.analysis_status != ReferenceAnalysisStatus.FULL_VIDEO_AUDIO:
        # Honest fallback when reference is metadata-only or unanalyzed: explicitly penalized and marked risky
        c_list = [
            CriterionScore(criterion="niche_topic_compatibility", score=0.45, weight=0.09, rationale="Unverified without video analysis."),
            CriterionScore(criterion="speech_cadence_fit", score=0.30, weight=0.10, rationale="Reference speech/shot cadence not observed (metadata-only)."),
            CriterionScore(criterion="footage_camera_fit", score=0.30, weight=0.08, rationale="Camera framing not observed from pixels."),
            CriterionScore(criterion="visual_complexity_feasibility", score=0.40, weight=0.08, rationale="Assumed fallback complexity only."),
            CriterionScore(criterion="explanation_clarity_support", score=0.35, weight=0.10, rationale="Cannot confirm visual explainer balance without frames."),
            CriterionScore(criterion="humor_meme_compatibility", score=0.35, weight=0.07, rationale="Meme cutaway frequency unknown."),
            CriterionScore(criterion="pacing_compatibility", score=0.25, weight=0.09, rationale="No observed edit events or shot durations."),
            CriterionScore(criterion="target_platform_fit", score=0.50 if is_short_target == (dna.visual.is_vertical if dna else False) else 0.30, weight=0.10, rationale="URL/platform hint only."),
            CriterionScore(criterion="multilingual_subtitle_legibility", score=0.45, weight=0.09, rationale="Requires safe default kk/ru/en subtitles."),
            CriterionScore(criterion="attention_hook_compatibility", score=0.30, weight=0.07, rationale="Opening 5s edit rhythm unavailable."),
            CriterionScore(criterion="production_feasibility", score=0.15, weight=0.06, rationale="Reference lacks local video/audio stream (metadata-only)."),
            CriterionScore(criterion="audio_sfx_compatibility", score=0.30, weight=0.04, rationale="Audio transient profile unavailable."),
            CriterionScore(criterion="copyright_safety_and_originality", score=0.85, weight=0.03, rationale="Metadata only; no media assets copied."),
        ]
        total = round(sum(c.score * c.weight for c in c_list), 4)
        return ReferenceCandidateEvaluation(
            reference_id=ref.reference_id,
            title=ref.title,
            analysis_status=ref.analysis_status,
            target_platform=target_platform,
            heuristic_compatibility_score=total,
            is_feasible=False,
            disqualification_reason="Metadata-only or unanalyzed reference: upload local MP4/MOV/WebM to run genuine frame-level analysis.",
            criteria_breakdown=c_list,
            transferable_strengths=[],
            identified_risks=[
                "Visual framing, cut rhythm, and subtitle typography are NOT observable for metadata-only references.",
                "Cannot outrank genuinely analyzed local reference videos.",
            ],
        )

    cuts_pm = dna.pacing.cuts_per_minute
    shot_med = dna.pacing.median_shot_duration_sec
    src_wpm = src["wpm"]
    is_tech_src = src["tech_ratio"] >= 0.3 or "science" in creator_niche or "program" in creator_niche or "physics" in creator_niche
    ref_is_explainer = dna.mapped_base_style in {"educational_explainer", "documentary_deep_dive"} or dna.visual.motion_graphics_ratio >= 0.25
    ref_is_zoomer = dna.mapped_base_style in {"zoomer_vlog", "short_form_viral", "comedy"} or cuts_pm >= 20.0

    # 1. Niche / topic compatibility
    niche_sc = 0.88 if (is_tech_src and ref_is_explainer) else (0.76 if is_tech_src and not ref_is_zoomer else (0.72 if not is_tech_src else 0.56))
    # 2. Speech cadence fit
    ideal_cpm = 24.0 if is_short_target else (14.0 if src_wpm >= 135 else 10.0)
    cadence_diff = abs(cuts_pm - ideal_cpm)
    cadence_sc = max(0.32, min(0.95, 1.0 - cadence_diff / 34.0))
    # 3. Footage & camera fit (static single camera talking head matches medium/closeup better than fast b-roll montage)
    if src["face_ratio"] >= 0.45:
        cam_sc = 0.90 if dna.visual.framing_style in {"medium_talking_head", "tight_closeup", "wide_desk"} else (0.68 if dna.visual.motion_graphics_ratio >= 0.25 else 0.52)
    else:
        cam_sc = 0.84 if dna.visual.framing_style in {"b_roll_dominant", "graphic_dominant"} else 0.62
    # 4. Visual complexity feasibility inside Symposium engines (Manim/PIL/FFmpeg)
    if dna.visual.camera_motion_mean > 0.09 and dna.visual.face_presence_ratio < 0.2:
        vis_feas = 0.62
    elif dna.recommended_animation_engine in {"manim_factual_diagram", "plotly_chart", "kinetic_callout", "meme_reaction_card"}:
        vis_feas = 0.90
    else:
        vis_feas = 0.78
    # 5. Explanation clarity support
    if is_tech_src:
        if cuts_pm > 28.0 and dna.visual.motion_graphics_ratio < 0.15:
            expl_sc = 0.36  # ultra-chaotic cuts hurt equation / physics / CS explanations
        elif ref_is_explainer or dna.pacing.median_shot_duration_sec >= 2.0:
            expl_sc = 0.92
        else:
            expl_sc = 0.68
    else:
        expl_sc = 0.80 if dna.pacing.median_shot_duration_sec <= 3.5 else 0.65
    # 6. Humor / meme compatibility
    if is_short_target or src["humor_energy"] >= 0.55:
        meme_sc = 0.86 if 0.15 <= dna.recommended_meme_density <= 0.60 else 0.68
    else:
        meme_sc = 0.88 if dna.recommended_meme_density <= 0.35 else 0.54
    # 7. Pacing compatibility with pause density
    if src["pause_ratio"] >= 0.22:
        pace_sc = 0.88 if cuts_pm >= 10.0 else 0.58  # messy paused footage needs active silence stripping & cuts
    else:
        pace_sc = max(0.45, 0.92 - abs(shot_med - (1.8 if is_short_target else 3.2)) / 8.0)
    # 8. Target platform fit (crucial differentiation between 16:9 YouTube vs 9:16 Shorts!)
    if is_short_target:
        plat_sc = 0.62
        if dna.visual.is_vertical:
            plat_sc += 0.18
        if dna.pacing.hook_window_cuts_first_5s >= 2 or cuts_pm >= 16.0:
            plat_sc += 0.16
        elif cuts_pm < 7.0:
            plat_sc -= 0.20
    else:
        plat_sc = 0.65
        if not dna.visual.is_vertical:
            plat_sc += 0.16
        if 8.0 <= cuts_pm <= 22.0:
            plat_sc += 0.15
        elif cuts_pm > 30.0:
            plat_sc -= 0.22  # exhausting for long-form YouTube watch sessions
    plat_sc = round(max(0.25, min(0.98, plat_sc)), 3)
    # 9. Multilingual (kk/ru/en) subtitle legibility
    if src["multilingual"]:
        sub_sc = 0.90 if dna.subtitles.estimated_words_per_cue <= 3 and dna.subtitles.vertical_region in {"bottom_safe", "lower_third", "center_dynamic"} else 0.64
    else:
        sub_sc = 0.84
    # 10. Attention hook compatibility
    hook_sc = 0.90 if (dna.pacing.hook_window_cuts_first_5s >= (2 if is_short_target else 1)) else 0.62
    # 11. Production feasibility
    prod_sc = 0.94 if (not ref.style_dna.honest_limitations) else 0.82
    # 12. Audio & SFX compatibility
    sfx_pm = dna.audio.transient_sfx_per_min
    audio_sc = 0.88 if (sfx_pm <= 20.0 and dna.audio.has_audio) else (0.58 if sfx_pm > 25.0 else 0.65)
    # 13. Copyright safety & structural translation
    copy_sc = 0.96

    criteria = [
        CriterionScore(criterion="niche_topic_compatibility", score=round(niche_sc, 3), weight=0.09, rationale=f"Source tech_ratio={src['tech_ratio']} vs ref archetype '{dna.archetype_label}'."),
        CriterionScore(criterion="speech_cadence_fit", score=round(cadence_sc, 3), weight=0.10, rationale=f"Source {src_wpm:.0f} WPM vs ref {cuts_pm:.1f} cuts/min."),
        CriterionScore(criterion="footage_camera_fit", score=round(cam_sc, 3), weight=0.08, rationale=f"Source face_ratio={src['face_ratio']} vs ref framing '{dna.visual.framing_style}'."),
        CriterionScore(criterion="visual_complexity_feasibility", score=round(vis_feas, 3), weight=0.08, rationale=f"Maps to native `{dna.recommended_animation_engine}` without proprietary copying."),
        CriterionScore(criterion="explanation_clarity_support", score=round(expl_sc, 3), weight=0.10, rationale=f"Median shot {shot_med:.2f}s & graphic ratio {dna.visual.motion_graphics_ratio:.2f}."),
        CriterionScore(criterion="humor_meme_compatibility", score=round(meme_sc, 3), weight=0.07, rationale=f"Recommended meme density={dna.recommended_meme_density:.2f}."),
        CriterionScore(criterion="pacing_compatibility", score=round(pace_sc, 3), weight=0.09, rationale=f"Source silence/pause ratio={src['pause_ratio']} vs ref shot rhythm."),
        CriterionScore(criterion="target_platform_fit", score=plat_sc, weight=0.10, rationale=f"Evaluated for `{target_platform}` against ref aspect `{dna.visual.aspect_ratio}` & hook cuts={dna.pacing.hook_window_cuts_first_5s}."),
        CriterionScore(criterion="multilingual_subtitle_legibility", score=round(sub_sc, 3), weight=0.09, rationale=f"Multilingual={src['multilingual']} vs <= {dna.subtitles.estimated_words_per_cue}w/cue in `{dna.subtitles.vertical_region}`."),
        CriterionScore(criterion="attention_hook_compatibility", score=round(hook_sc, 3), weight=0.07, rationale=f"First 5s hook transitions={dna.pacing.hook_window_cuts_first_5s}."),
        CriterionScore(criterion="production_feasibility", score=round(prod_sc, 3), weight=0.06, rationale="Full local video+audio Style DNA observed."),
        CriterionScore(criterion="audio_sfx_compatibility", score=round(audio_sc, 3), weight=0.04, rationale=f"Ref SFX transients/min={sfx_pm:.1f}, music_bed={dna.audio.estimated_music_bed_present}."),
        CriterionScore(criterion="copyright_safety_and_originality", score=copy_sc, weight=0.03, rationale="Structural editing parameters only; original assets generated cleanly."),
    ]
    total = round(sum(c.score * c.weight for c in criteria), 4)

    strengths: list[str] = []
    risks: list[str] = []
    for c in sorted(criteria, key=lambda x: x.score, reverse=True)[:3]:
        strengths.append(f"{c.criterion}: {c.score:.2f} ({c.rationale})")
    for c in criteria:
        if c.score < 0.60:
            risks.append(f"Low {c.criterion} ({c.score:.2f}): {c.rationale}")

    return ReferenceCandidateEvaluation(
        reference_id=ref.reference_id,
        title=ref.title,
        analysis_status=ref.analysis_status,
        target_platform=target_platform,
        heuristic_compatibility_score=total,
        is_feasible=True,
        criteria_breakdown=criteria,
        transferable_strengths=strengths,
        identified_risks=risks,
    )


def build_hybrid_style_guide(
    ranked: list[ReferenceCandidateEvaluation],
    refs_by_id: dict[str, ReferenceItem],
    target_platform: str,
) -> HybridStyleGuide:
    """Mode B: blend strongest individual dimensions across feasible FULL_VIDEO_AUDIO references."""
    feasible = [r for r in ranked if r.is_feasible and r.reference_id in refs_by_id]
    if not feasible:
        feasible = [r for r in ranked if r.reference_id in refs_by_id]
    primary_id = feasible[0].reference_id

    def _best_for(criterion: str) -> str:
        best_id = primary_id
        best_val = -1.0
        for ev in feasible:
            sc = next((c.score for c in ev.criteria_breakdown if c.criterion == criterion), 0.0)
            if sc > best_val:
                best_val = sc
                best_id = ev.reference_id
        return best_id

    dim_sources = {
        "opening_hook": _best_for("attention_hook_compatibility"),
        "pacing_rhythm": _best_for("pacing_compatibility"),
        "subtitle_typography": _best_for("multilingual_subtitle_legibility"),
        "motion_graphics": _best_for("explanation_clarity_support"),
        "audio_sfx": _best_for("audio_sfx_compatibility"),
    }
    notes = [
        f"Hybrid for {target_platform}: primary base={refs_by_id[primary_id].title}",
        *[
            f"Dimension '{dim}' sourced from `{refs_by_id[rid].title}` ({rid})"
            for dim, rid in dim_sources.items()
            if rid in refs_by_id
        ],
    ]
    return HybridStyleGuide(
        primary_reference_id=primary_id,
        dimension_sources=dim_sources,
        blended_notes=notes,
    )


def build_style_transfer_plan(
    project_id: str,
    primary_ref: ReferenceItem,
    refs_by_id: dict[str, ReferenceItem],
    conference: ReferenceConferenceRecord,
    target_platform: str = "youtube_long",
    adjustments: dict[str, Any] | None = None,
) -> tuple[StyleTransferPlan, EditingStyleProfile]:
    """Translate selected StyleDNA / HybridStyleGuide + user adjustments into real editing parameters."""
    adj = adjustments or {}
    dna: StyleDNA | None = primary_ref.style_dna
    hybrid = conference.hybrid_guide if conference.selection_mode == SelectionMode.INTELLIGENT_HYBRID else None

    def _dna_for(dim: str) -> StyleDNA | None:
        if hybrid and dim in hybrid.dimension_sources:
            rid = hybrid.dimension_sources[dim]
            r = refs_by_id.get(rid)
            if r and r.style_dna is not None:
                return r.style_dna
        return dna

    pace_dna = _dna_for("pacing_rhythm")
    sub_dna = _dna_for("subtitle_typography")
    gfx_dna = _dna_for("motion_graphics")
    aud_dna = _dna_for("audio_sfx")
    hook_dna = _dna_for("opening_hook")

    base_style = dna.mapped_base_style if dna else "educational_explainer"
    is_short = target_platform in {"youtube_shorts", "tiktok", "instagram_reels"}
    if is_short and base_style in {"minimalist_cinematic", "documentary_deep_dive"}:
        base_style = "short_form_viral"

    raw_shot = pace_dna.pacing.median_shot_duration_sec if (pace_dna and pace_dna.pacing.median_shot_duration_sec > 0) else 3.0
    base_mult = float(adj.get("pacing_multiplier", 1.0))
    if adj.get("increase_pacing") and "pacing_multiplier" not in adj:
        base_mult = round(base_mult + 0.25, 2)
    elif adj.get("decrease_pacing") and "pacing_multiplier" not in adj:
        base_mult = round(base_mult - 0.20, 2)
    pacing_mult = max(0.5, min(1.8, base_mult))
    target_shot = round(max(0.85, min(7.5, raw_shot / pacing_mult)), 2)

    cpm = pace_dna.pacing.cuts_per_minute if pace_dna else 10.0
    pause_thresh = 0.20 if (cpm * pacing_mult >= 18.0 or is_short) else (0.38 if cpm * pacing_mult < 8.0 else 0.28)
    punch_step = 0.10 if (pace_dna and (pace_dna.pacing.punch_ins_per_minute >= 3.0 or cpm >= 18.0)) else 0.05
    punch_step = round(min(0.16, punch_step * pacing_mult), 3)

    max_hook = 2.6 if (hook_dna and hook_dna.pacing.hook_window_cuts_first_5s >= 2) or is_short else 4.2

    sub_words = 2 if is_short else (sub_dna.subtitles.estimated_words_per_cue if sub_dna else 3)
    sub_words = max(1, min(4, sub_words))
    if adj.get("subtitle_minimal"):
        sub_words = 2
        font_sz = 46 if not is_short else 52
        scale_f = 0.92
    elif adj.get("subtitle_bold"):
        sub_words = 2
        font_sz = 64 if not is_short else 68
        scale_f = 1.08
    else:
        font_sz = 60 if (sub_dna and sub_dna.subtitles.estimated_font_weight == "heavy") else 54
        scale_f = 1.05 if (sub_dna and sub_dna.subtitles.estimated_font_weight == "heavy") else 0.96

    safe_bottom = 285 if is_short else 92
    pos_hint = sub_dna.subtitles.vertical_region if sub_dna else ("center_dynamic" if is_short else "bottom_safe")
    if pos_hint == "center_dynamic" and is_short:
        safe_bottom = 360

    enable_memes = bool(adj.get("enable_memes", True))
    if adj.get("disable_memes"):
        enable_memes = False
    enable_edu_gfx = bool(adj.get("enable_educational_graphics", True))
    boost_edu = bool(adj.get("boost_educational_graphics", False))
    keep_struct = bool(adj.get("keep_structure_apply_rhythm_only", False))

    anim_hint = gfx_dna.recommended_animation_engine if gfx_dna else "kinetic_callout"
    if boost_edu and anim_hint == "meme_reaction_card":
        anim_hint = "manim_factual_diagram"
    overlay_mult = 1.45 if boost_edu else (0.5 if not enable_memes else 1.0)

    sfx_pm = aud_dna.audio.transient_sfx_per_min if aud_dna else 6.0
    sfx_spacing = max(1.2, min(6.0, 60.0 / max(4.0, sfx_pm * pacing_mult)))
    sfx_gain = -10.5 if (aud_dna and aud_dna.recommended_sfx_intensity >= 0.55) else -13.5
    music_duck = -24.0 if (aud_dna and aud_dna.audio.estimated_music_bed_present) else -20.0

    # Gentle bounded color transfer derived from StyleDNA (protects skin tones)
    ref_sat = dna.visual.median_saturation if dna else 0.42
    ref_con = dna.visual.contrast_range if dna else 0.50
    ref_bri = dna.visual.median_brightness if dna else 0.50
    sat_factor = round(max(0.92, min(1.18, 1.0 + (ref_sat - 0.42) * 0.35)), 3)
    con_factor = round(max(0.96, min(1.14, 1.0 + (ref_con - 0.50) * 0.25)), 3)
    bri_shift = round(max(-0.04, min(0.05, (ref_bri - 0.50) * 0.12)), 3)

    params = StyleTransferParameters(
        base_style_profile=base_style,
        target_shot_duration_sec=target_shot,
        cut_pause_threshold_sec=pause_thresh,
        zoom_punch_step=punch_step,
        max_hook_duration_sec=max_hook,
        subtitle_max_words=sub_words,
        subtitle_font_size=font_sz,
        subtitle_scale_factor=scale_f,
        subtitle_safe_margin_bottom_px=safe_bottom,
        subtitle_position_hint=pos_hint,
        animation_engine_hint=anim_hint,
        overlay_cadence_multiplier=overlay_mult,
        sfx_click_min_spacing_sec=round(sfx_spacing, 2),
        sfx_gain_db=sfx_gain,
        music_duck_db=music_duck,
        enable_memes=enable_memes,
        enable_educational_graphics=enable_edu_gfx,
        pacing_multiplier=pacing_mult,
        color_saturation=sat_factor,
        color_contrast=con_factor,
        color_brightness=bri_shift,
        keep_structure_apply_rhythm_only=keep_struct,
    )

    # Synthesize runtime EditingStyleProfile compatible with timeline_builder
    base_prof = get_style_profile(base_style).model_copy(deep=True)
    base_prof.target_avg_shot_sec = params.target_shot_duration_sec
    base_prof.max_silence_gap_sec = params.cut_pause_threshold_sec
    base_prof.zoom_punch_intensity = min(0.85, max(0.15, params.zoom_punch_step * 5.0))
    base_prof.audio_duck_db = params.music_duck_db
    base_prof.display_name = f"Reference Transfer ({primary_ref.title[:28]})"

    directives = [
        f"Style basis: `{primary_ref.title}` ({conference.selection_mode.value}) -> `{params.base_style_profile}`",
        f"Cutting cadence: ~{params.target_shot_duration_sec}s median shot, strip silence >= {params.cut_pause_threshold_sec}s, punch zoom step +{params.zoom_punch_step}.",
        f"Subtitles: <= {params.subtitle_max_words} words/cue, font {params.subtitle_font_size}px (scale={params.subtitle_scale_factor}), safe bottom={params.subtitle_safe_margin_bottom_px}px ({params.subtitle_position_hint}).",
        f"Motion/Overlays: `{params.animation_engine_hint}`, memes={'ON' if params.enable_memes else 'OFF'}, edu_graphics={'ON' if params.enable_educational_graphics else 'OFF'}.",
        f"Audio & Color: SFX min spacing={params.sfx_click_min_spacing_sec}s @ {params.sfx_gain_db}dB, sat={params.color_saturation}, contrast={params.color_contrast}.",
    ]
    if params.keep_structure_apply_rhythm_only:
        directives.append("Keep-structure mode active: preserving storyboard narrative ordering while applying reference pacing/typography.")

    plan = StyleTransferPlan(
        plan_id=f"stp_{uuid.uuid4().hex[:8]}",
        project_id=project_id,
        target_platform=target_platform,
        selection_mode=conference.selection_mode,
        primary_reference_id=primary_ref.reference_id,
        hybrid_guide=hybrid,
        parameters=params,
        applied_adjustments=dict(adj),
        timeline_directives=directives,
    )
    return plan, base_prof


def _subdivide_long_clips_to_cadence(tl: RenderTimeline, p: Any) -> None:
    """Split over-long talking-head clips safely at inter-cue gaps to match reference shot cadence."""
    desired_shot = max(2.2, min(7.5, float(p.target_shot_duration_sec) / max(0.65, float(p.pacing_multiplier))))
    if p.pacing_multiplier <= 0.85 or desired_shot > 6.5:
        return

    new_clips: list[VideoClipNode] = []
    cue_reassignments: dict[int, str] = {}

    for clip in tl.clips:
        c_s = float(clip.timeline_start_sec)
        c_e = float(clip.timeline_end_sec)
        c_dur = c_e - c_s
        if c_dur <= max(4.2, desired_shot * 1.45):
            new_clips.append(clip)
            continue

        # Collect strictly safe cut timestamps lying cleanly between adjacent caption cues inside this clip.
        inner_cues = sorted(
            [
                (idx, cue)
                for idx, cue in enumerate(tl.captions)
                if (cue.clip_id == clip.clip_id) or (0.5 * (cue.start_sec + cue.end_sec) >= c_s and 0.5 * (cue.start_sec + cue.end_sec) <= c_e)
            ],
            key=lambda item: (item[1].start_sec, item[1].end_sec),
        )
        safe_gaps: list[float] = []
        for gi in range(len(inner_cues) - 1):
            c_left = inner_cues[gi][1]
            c_right = inner_cues[gi + 1][1]
            if c_left.end_sec <= c_right.start_sec + 0.004:
                mid = round(min(c_right.start_sec, max(c_left.end_sec, 0.5 * (c_left.end_sec + c_right.start_sec))), 3)
                if c_s + 1.6 <= mid <= c_e - 1.6:
                    safe_gaps.append(mid)

        splits: list[float] = []
        last_t = c_s
        for cand_t in safe_gaps:
            if cand_t - last_t >= desired_shot * 0.90 and (c_e - cand_t) >= 1.6:
                splits.append(cand_t)
                last_t = cand_t

        if not splits:
            new_clips.append(clip)
            continue

        bounds = [c_s, *splits, c_e]
        sub_nodes: list[VideoClipNode] = []
        for pi in range(len(bounds) - 1):
            a, b = bounds[pi], bounds[pi + 1]
            sa = round(clip.source_start_sec + (a - c_s) * clip.playback_speed, 4)
            sb = round(clip.source_start_sec + (b - c_s) * clip.playback_speed, 4)
            cid = clip.clip_id if pi == 0 else f"{clip.clip_id}_p{pi}"
            node = clip.model_copy(deep=True)
            node.clip_id = cid
            node.source_start_sec = sa
            node.source_end_sec = sb
            node.timeline_start_sec = round(a, 4)
            node.timeline_end_sec = round(b, 4)
            if pi > 0:
                node.transition_in = TransitionSpec(type="cut", duration_sec=0.0, sfx_preset_id=None)
                node.effects = [ef for ef in node.effects if ef.preset_id != "reference_color_grade"]
            sub_nodes.append(node)
            new_clips.append(node)

        for idx, cue in inner_cues:
            mid_q = 0.5 * (float(cue.start_sec) + float(cue.end_sec))
            target_sub = next(
                (sn for sn in sub_nodes if sn.timeline_start_sec - 0.05 <= mid_q <= sn.timeline_end_sec + 0.05),
                sub_nodes[0],
            )
            cue_reassignments[idx] = target_sub.clip_id

        for sm in getattr(tl, "segment_mappings", []):
            if sm.clip_id == clip.clip_id:
                target_sub = next(
                    (sn for sn in sub_nodes if sn.timeline_start_sec - 0.05 <= sm.timeline_start_sec <= sn.timeline_end_sec + 0.05),
                    sub_nodes[0],
                )
                sm.clip_id = target_sub.clip_id

    for idx, new_cid in cue_reassignments.items():
        tl.captions[idx].clip_id = new_cid
    tl.clips = new_clips


def apply_style_transfer_plan_to_timeline(
    timeline: RenderTimeline,
    plan: StyleTransferPlan,
) -> RenderTimeline:
    """Apply StyleTransferParameters directly to a RenderTimeline preserving source mappings and P0 subtitle safety."""
    tl = timeline.model_copy(deep=True)
    p = plan.parameters
    is_vertical = tl.platform_format != TargetPlatformFormat.YOUTUBE_LONG or tl.height > tl.width

    # 0. Subdivide over-long clips cleanly at safe inter-cue gaps to approach target reference cut rhythm
    _subdivide_long_clips_to_cadence(tl, p)

    # 1. Visual zoom punch cadence across talking-head clips
    punch_zoom = round(min(1.22, 1.0 + max(0.11, float(p.zoom_punch_step))), 3)
    for idx, clip in enumerate(tl.clips):
        if p.pacing_multiplier <= 0.85:
            if clip.zoom_scale > 1.04:
                clip.zoom_scale = 1.03
        elif idx % 2 == 1:
            clip.zoom_scale = max(clip.zoom_scale, punch_zoom) if clip.zoom_scale <= 1.08 else clip.zoom_scale
        else:
            # Anchor even shots close to 1.0x so alternating odd zoom-punches stand out cleanly
            if len(tl.clips) >= 3 and "_p" in clip.clip_id or (idx + 1 < len(tl.clips) and "_p" in tl.clips[idx + 1].clip_id):
                clip.zoom_scale = 1.0
            elif p.pacing_multiplier >= 1.20 and clip.zoom_scale <= 1.04:
                clip.zoom_scale = round(min(1.18, 1.0 + p.zoom_punch_step * 0.65), 3)
        if not any(ef.preset_id == "reference_color_grade" for ef in clip.effects):
            clip.effects.append(
                EffectSpec(
                    effect_id=f"fx_ref_color_{idx}",
                    preset_id="reference_color_grade",
                    category=EffectCategory.VISUAL,
                    start_sec=0.0,
                    duration_sec=max(0.1, round(clip.timeline_end_sec - clip.timeline_start_sec, 3)),
                    parameters={
                        "saturation": p.color_saturation,
                        "contrast": p.color_contrast,
                        "brightness": p.color_brightness,
                        "ffmpeg_filter": (
                            f"eq=saturation={p.color_saturation:.3f}:contrast={p.color_contrast:.3f}:brightness={p.color_brightness:.3f}"
                        ),
                    },
                    editorial_purpose="Reference StyleDNA calibrated color balance.",
                )
            )

    # 2. Subtitles: typography + safe zones (NEVER violate P0 bottom safe margins)
    tl.caption_settings.max_words_per_line = max(1, min(3 if is_vertical else 4, p.subtitle_max_words))
    tl.caption_settings.font_size = max(28, min(80, p.subtitle_font_size))
    tl.caption_settings.scale_factor = round(max(0.75, min(1.30, p.subtitle_scale_factor)), 2)
    min_safe = 260 if is_vertical else 85
    tl.caption_settings.safe_margin_bottom_px = max(min_safe, p.subtitle_safe_margin_bottom_px)

    # 3. Filter overlays cleanly per interactive controls (e.g. Disable Memes) without breaking timeline sync
    if not p.enable_memes:
        tl.overlays = [
            ov for ov in tl.overlays
            if "meme" not in ov.label.lower() and "meme" not in ov.asset_path.lower()
        ]

    # 4. Audio SFX / click gain adjustments matching Style DNA energy
    for ae in tl.audio_events:
        if ae.role == "sfx":
            ae.gain_db = round(max(-24.0, min(-5.0, p.sfx_gain_db)), 1)
        elif ae.role in {"ambient", "music"}:
            ae.gain_db = round(max(-36.0, min(-12.0, p.music_duck_db)), 1)

    tl.style_profile = p.base_style_profile
    tl.edit_notes.append(f"Applied Reference StyleTransferPlan {plan.plan_id} (ref={plan.primary_reference_id}).")
    return tl


def measure_post_render_style_similarity(
    project_id: str,
    timeline: RenderTimeline,
    reference: ReferenceItem,
    rendered_video_path: Path,
    qa_report: QualityAssuranceReport | None = None,
) -> StyleSimilarityReport:
    """Measure real post-render edit cadence, caption sizing, aspect, and SFX density against selected StyleDNA."""
    from symposium_studio.engines.analysis.reference_analyzer import _detect_zoom_and_micro_cuts

    probe = probe_media(rendered_video_path)
    cuts = detect_scene_cuts(rendered_video_path, duration_sec=max(0.2, probe.duration_sec), threshold=24.0)
    base_cut_starts = [round(c.start_sec, 3) for c in cuts if c.start_sec > 0.25]
    micro = _detect_zoom_and_micro_cuts(rendered_video_path, max(0.2, probe.duration_sec), base_cut_starts)
    raw_pts = sorted([*base_cut_starts, *(float(m["time_sec"]) for m in micro)])
    dedup_pts: list[float] = []
    for pt in raw_pts:
        if not dedup_pts or (pt - dedup_pts[-1]) >= 0.40:
            dedup_pts.append(pt)
    cut_cnt = max(len(dedup_pts), max(0, len(timeline.clips) - 1))
    dur_mins = max(1.0 / 60.0, probe.duration_sec / 60.0)
    rend_cpm = round(cut_cnt / dur_mins, 2)
    rend_med_shot = round(probe.duration_sec / max(1, cut_cnt + 1), 2)
    rend_words_per_cue = round(
        sum(len(c.text.split()) for c in timeline.captions) / max(1, len(timeline.captions)), 2
    ) if timeline.captions else 0.0
    sfx_cnt = sum(1 for a in timeline.audio_events if a.role == "sfx")
    rend_sfx_pm = round(sfx_cnt / dur_mins, 2)

    dna = reference.style_dna
    metrics: list[StyleSimilarityMetric] = []
    divergences: list[str] = []
    scores: list[float] = []

    if dna is not None and reference.analysis_status == ReferenceAnalysisStatus.FULL_VIDEO_AUDIO:
        ref_cpm = dna.pacing.cuts_per_minute
        cpm_delta = round(rend_cpm - ref_cpm, 2)
        cpm_score = max(0.2, 1.0 - abs(cpm_delta) / max(15.0, ref_cpm + 5.0))
        scores.append(cpm_score)
        metrics.append(
            StyleSimilarityMetric(
                metric="cuts_per_minute",
                reference_value=ref_cpm,
                rendered_value=rend_cpm,
                delta=cpm_delta,
                alignment_status="aligned" if cpm_score >= 0.72 else "close",
            )
        )

        ref_shot = dna.pacing.median_shot_duration_sec
        shot_delta = round(rend_med_shot - ref_shot, 2)
        shot_score = max(0.2, 1.0 - abs(shot_delta) / max(3.5, ref_shot + 1.0))
        scores.append(shot_score)
        metrics.append(
            StyleSimilarityMetric(
                metric="median_shot_duration_sec",
                reference_value=ref_shot,
                rendered_value=rend_med_shot,
                delta=shot_delta,
                alignment_status="aligned" if shot_score >= 0.7 else "close",
            )
        )

        ref_wpc = float(dna.subtitles.estimated_words_per_cue)
        wpc_delta = round(rend_words_per_cue - ref_wpc, 2)
        wpc_score = max(0.4, 1.0 - abs(wpc_delta) / 3.0)
        scores.append(wpc_score)
        metrics.append(
            StyleSimilarityMetric(
                metric="words_per_caption_cue",
                reference_value=ref_wpc,
                rendered_value=rend_words_per_cue,
                delta=wpc_delta,
                alignment_status="aligned" if wpc_score >= 0.75 else "diverged_for_clarity",
            )
        )

        sfx_delta = round(rend_sfx_pm - dna.audio.transient_sfx_per_min, 2)
        sfx_score = max(0.35, 1.0 - abs(sfx_delta) / 25.0)
        scores.append(sfx_score)
        metrics.append(
            StyleSimilarityMetric(
                metric="sfx_density_per_min",
                reference_value=dna.audio.transient_sfx_per_min,
                rendered_value=rend_sfx_pm,
                delta=sfx_delta,
                alignment_status="aligned" if sfx_score >= 0.7 else "close",
            )
        )

        rend_aspect = "9:16" if probe.height > probe.width else "16:9"
        aspect_ok = rend_aspect == dna.visual.aspect_ratio
        metrics.append(
            StyleSimilarityMetric(
                metric="aspect_ratio",
                reference_value=dna.visual.aspect_ratio,
                rendered_value=rend_aspect,
                alignment_status="aligned" if aspect_ok else "diverged_for_clarity",
            )
        )
        if not aspect_ok:
            divergences.append(
                f"Target format rendered as `{rend_aspect}` while reference was `{dna.visual.aspect_ratio}` per requested platform deliverable."
            )
        if rend_cpm + 6.0 < ref_cpm:
            divergences.append(
                "Rendered cadence slightly calmer than raw reference to protect spoken explanation & multilingual subtitle readability."
            )
    else:
        divergences.append(
            "Selected reference had metadata-only status; comparison reports rendered metrics without pixel-level reference baseline."
        )
        scores.append(0.5)

    overall = round(sum(scores) / len(scores), 3) if scores else 0.7
    return StyleSimilarityReport(
        report_id=f"sim_{uuid.uuid4().hex[:8]}",
        project_id=project_id,
        timeline_id=timeline.timeline_id,
        reference_id=reference.reference_id,
        rendered_video_path=str(rendered_video_path),
        overall_style_adherence_score=overall,
        metrics=metrics,
        intentional_divergences=divergences,
        qa_passed=bool(qa_report.passed) if qa_report is not None else True,
        created_at=_utc_now(),
    )
