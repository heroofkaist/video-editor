"""Intelligent editing & semantic media understanding engine."""

from symposium_studio.engines.analysis.pipeline import run_full_media_analysis
from symposium_studio.engines.analysis.probe import probe_media
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
    "STYLE_PROFILES",
    "align_transcript_to_audio",
    "align_words_to_envelope",
    "analyze_reference_style",
    "apply_transcript_corrections",
    "audit_and_verify_transcript",
    "compute_audio_envelope",
    "detect_speech_islands_in_window",
    "extract_raw_mono_wav",
    "get_style_profile",
    "probe_media",
    "run_full_media_analysis",
]
