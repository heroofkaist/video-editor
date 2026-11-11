"""Orchestrates full local media analysis (probe, ASR, diarization, vision, semantics)."""

from __future__ import annotations

from pathlib import Path

from symposium_studio.engines.analysis.diarize import detect_silences, diarize_segments
from symposium_studio.engines.analysis.probe import probe_media
from symposium_studio.engines.analysis.semantics import detect_repetitions, score_highlights, segment_topics
from symposium_studio.engines.analysis.transcribe import transcribe_media
from symposium_studio.engines.analysis.vision import analyze_visual_context, detect_scene_cuts
from symposium_studio.models.media import MediaAnalysisBundle


def run_full_media_analysis(
    project_id: str,
    media_path: Path,
    whisper_model: str = "mlx-community/whisper-small-mlx",
    language: str | None = None,
) -> MediaAnalysisBundle:
    probe = probe_media(media_path)
    transcript = transcribe_media(
        media_path=media_path,
        duration_sec=probe.duration_sec,
        model_name=whisper_model,
        language=language,
    )
    silences = detect_silences(media_path, duration_sec=probe.duration_sec) if probe.has_audio else []
    speaker_turns, updated_segments = (
        diarize_segments(media_path, transcript.segments) if probe.has_audio else ([], transcript.segments)
    )
    transcript.segments = updated_segments

    scene_cuts = detect_scene_cuts(media_path, duration_sec=probe.duration_sec)
    visual_samples, rec_crop_x = analyze_visual_context(media_path, sample_interval_sec=2.0)
    topics = segment_topics(transcript.segments)
    repetitions = detect_repetitions(transcript.segments)
    highlights = score_highlights(
        segments=transcript.segments,
        topics=topics,
        visual_samples=visual_samples,
        duration_sec=probe.duration_sec,
    )

    return MediaAnalysisBundle(
        project_id=project_id,
        probe=probe,
        transcript=transcript,
        speaker_turns=speaker_turns,
        silences=silences,
        scene_cuts=scene_cuts,
        visual_samples=visual_samples,
        topics=topics,
        highlights=highlights,
        repetitions_flagged=repetitions,
        recommended_crop_center_x=rec_crop_x,
    )
