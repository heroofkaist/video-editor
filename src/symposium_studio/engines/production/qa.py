"""Automated Quality Assurance Reviewer inspecting rendered media streams, black/freeze frames, loudness, and visual samples."""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

from symposium_studio.engines.analysis.probe import probe_media
from symposium_studio.models.social import QualityAssuranceReport, QualityCheckIssue
from symposium_studio.models.timeline import RenderTimeline


def _detect_intervals(video_path: Path, filter_expr: str, start_pat: str, end_pat: str) -> list[dict[str, float]]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video_path), "-vf", filter_expr, "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    txt = proc.stderr or ""
    intervals: list[dict[str, float]] = []
    cur_start: float | None = None
    for line in txt.splitlines():
        ms = re.search(start_pat, line)
        if ms:
            cur_start = float(ms.group(1))
        me = re.search(end_pat, line)
        if me:
            end_v = float(me.group(1))
            intervals.append({"start_sec": round(cur_start or 0.0, 3), "end_sec": round(end_v, 3)})
            cur_start = None
    return intervals


def run_quality_assurance(
    project_id: str,
    video_path: Path,
    timeline: RenderTimeline,
    frames_out_dir: Path,
) -> QualityAssuranceReport:
    probe = probe_media(video_path)
    issues: list[QualityCheckIssue] = []

    if probe.duration_sec <= 0.5:
        issues.append(QualityCheckIssue(severity="error", code="EMPTY_DURATION", message="Rendered video duration <= 0.5s."))
    if not probe.has_audio:
        issues.append(QualityCheckIssue(severity="error", code="MISSING_AUDIO", message="Rendered video has no audio stream."))
    elif probe.mean_volume_db is not None and probe.mean_volume_db < -42.0:
        issues.append(
            QualityCheckIssue(
                severity="warning",
                code="QUIET_AUDIO",
                message=f"Mean audio volume is low ({probe.mean_volume_db} dB).",
            )
        )
    if probe.width != timeline.width or probe.height != timeline.height:
        issues.append(
            QualityCheckIssue(
                severity="error",
                code="DIMENSION_MISMATCH",
                message=f"Expected {timeline.width}x{timeline.height}, got {probe.width}x{probe.height}.",
            )
        )

    blacks = _detect_intervals(
        video_path,
        "blackdetect=d=0.45:pix_th=0.10",
        r"black_start:([0-9.]+)",
        r"black_end:([0-9.]+)",
    )
    if blacks:
        issues.append(
            QualityCheckIssue(
                severity="warning",
                code="BLACK_FRAMES_DETECTED",
                message=f"Detected {len(blacks)} black frame interval(s) >= 0.45s.",
                timestamp_sec=blacks[0]["start_sec"],
            )
        )

    freezes = _detect_intervals(
        video_path,
        "freezedetect=n=0.002:d=2.5",
        r"freeze_start:\s*([0-9.]+)",
        r"freeze_end:\s*([0-9.]+)",
    )
    if freezes:
        issues.append(
            QualityCheckIssue(
                severity="info",
                code="FREEZE_FRAMES_OBSERVED",
                message=f"Detected {len(freezes)} static frame hold(s) >= 2.5s.",
                timestamp_sec=freezes[0]["start_sec"],
            )
        )

    if timeline.caption_settings.enabled and not timeline.captions:
        issues.append(QualityCheckIssue(severity="warning", code="NO_CAPTIONS", message="Captions enabled but cue list is empty."))

    # Extract 4 evenly spaced verification frames for visual inspection
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[str] = []
    for idx, frac in enumerate((0.05, 0.30, 0.55, 0.85)):
        ts = min(max(0.2, probe.duration_sec * frac), max(0.2, probe.duration_sec - 0.2))
        fpath = frames_out_dir / f"{video_path.stem}_check_{idx}.png"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{ts:.3f}",
                "-i", str(video_path),
                "-vframes", "1",
                str(fpath),
            ],
            check=True,
        )
        frame_paths.append(str(fpath))

    passed = not any(i.severity == "error" for i in issues)
    return QualityAssuranceReport(
        qa_id=f"qa_{uuid.uuid4().hex[:8]}",
        project_id=project_id,
        artifact_path=str(video_path),
        passed=passed,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        duration_sec=probe.duration_sec,
        has_video=True,
        has_audio=probe.has_audio,
        audio_mean_db=probe.mean_volume_db,
        audio_max_db=probe.max_volume_db,
        black_frame_intervals=blacks,
        freeze_frame_intervals=freezes,
        caption_count=len(timeline.captions),
        overlay_count=len(timeline.overlays),
        verification_frame_paths=frame_paths,
        issues=issues,
    )
