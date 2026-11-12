"""Intelligent editing & semantic media understanding engine."""

from symposium_studio.engines.analysis.pipeline import run_full_media_analysis
from symposium_studio.engines.analysis.probe import probe_media
from symposium_studio.engines.analysis.reference_analyzer import analyze_reference_item
from symposium_studio.engines.analysis.reference_library import ReferenceLibraryManager
from symposium_studio.engines.analysis.reference_selector import (
    apply_style_transfer_plan_to_timeline,
    build_hybrid_style_guide,
    build_style_transfer_plan,
    evaluate_reference_against_source,
    measure_post_render_style_similarity,
)
from symposium_studio.engines.analysis.styles import STYLE_PROFILES, analyze_reference_style, get_style_profile
from symposium_studio.engines.analysis.transcribe import (
    align_transcript_to_audio,
    align_words_to_envelope,
    apply_transcript_corrections,
    audit_and_verify_transcript,
    compute_audio_envelope,
    detect_speech_islands_in_window,
    extract_raw_mono_wav,
)

__all__ = [
    "ReferenceLibraryManager",
    "STYLE_PROFILES",
    "align_transcript_to_audio",
    "align_words_to_envelope",
    "analyze_reference_item",
    "analyze_reference_style",
    "apply_style_transfer_plan_to_timeline",
    "apply_transcript_corrections",
    "audit_and_verify_transcript",
    "build_hybrid_style_guide",
    "build_style_transfer_plan",
    "compute_audio_envelope",
    "detect_speech_islands_in_window",
    "evaluate_reference_against_source",
    "extract_raw_mono_wav",
    "get_style_profile",
    "measure_post_render_style_similarity",
    "probe_media",
    "run_full_media_analysis",
]
