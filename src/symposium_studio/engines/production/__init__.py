"""Editable timeline, OpenTimelineIO interchange, video rendering, and QA engine."""

from symposium_studio.engines.production.captions import (
    align_captions_to_final_audio,
    check_caption_quality,
    validate_subtitle_integrity,
    write_ass,
    write_srt,
)
from symposium_studio.engines.production.directors import (
    evaluate_ending_closure,
    evaluate_hook_candidates,
    plan_coherent_beat_sequence,
    select_best_hook,
    select_closing_payoff,
    snap_ending_boundary,
)
from symposium_studio.engines.production.qa import run_quality_assurance
from symposium_studio.engines.production.renderer import ensure_project_sfx, render_thumbnail, render_timeline
from symposium_studio.engines.production.timeline_builder import (
    add_click_sounds_to_timeline,
    apply_iterative_edit,
    build_timeline_from_storyboard,
    build_vertical_short_timelines,
    export_otio,
    remap_edited_timeline,
    reorder_timeline_clips,
)

__all__ = [
    "add_click_sounds_to_timeline",
    "align_captions_to_final_audio",
    "apply_iterative_edit",
    "build_timeline_from_storyboard",
    "build_vertical_short_timelines",
    "check_caption_quality",
    "ensure_project_sfx",
    "evaluate_ending_closure",
    "evaluate_hook_candidates",
    "export_otio",
    "plan_coherent_beat_sequence",
    "remap_edited_timeline",
    "render_thumbnail",
    "render_timeline",
    "reorder_timeline_clips",
    "run_quality_assurance",
    "select_best_hook",
    "select_closing_payoff",
    "snap_ending_boundary",
    "validate_subtitle_integrity",
    "write_ass",
    "write_srt",
]
