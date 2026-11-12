"""Automated Quality Assurance Reviewer inspecting rendered media, LUFS, captions, overlays, and story closure."""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from symposium_studio.engines.analysis.probe import probe_media
from symposium_studio.engines.analysis.semantics import DANGLING_STARTERS, INCOMPLETE_ENDINGS, _tokenize
from symposium_studio.engines.analysis.transcribe import extract_raw_mono_wav
from symposium_studio.engines.production.captions import check_caption_quality
from symposium_studio.engines.production.directors import evaluate_ending_closure
from symposium_studio.models.media import MediaAnalysisBundle
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


def _measure_integrated_lufs(video_path: Path) -> float | None:
    """Measure true EBU R128 integrated loudness (LUFS) via FFmpeg ebur128 filter."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video_path), "-af", "ebur128", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    txt = proc.stderr or ""
    # Summary block prints: "    I:         -14.3 LUFS"
    matches = re.findall(r"I:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*LUFS", txt)
    if matches:
        try:
            return round(float(matches[-1]), 2)
        except ValueError:
            return None
    return None


def _box_intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def run_quality_assurance(
    project_id: str,
    video_path: Path,
    timeline: RenderTimeline,
    frames_out_dir: Path,
    analysis: MediaAnalysisBundle | None = None,
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

    measured_lufs = _measure_integrated_lufs(video_path) if probe.has_audio else None
    if measured_lufs is not None and (measured_lufs < -20.5 or measured_lufs > -10.5):
        issues.append(
            QualityCheckIssue(
                severity="warning",
                code="LOUDNESS_OUT_OF_TARGET",
                message=f"Integrated loudness is {measured_lufs} LUFS (target streaming band: -20.5 to -10.5 LUFS).",
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

    cap_quality: dict[str, Any]
    try:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="qa_cap_audio_") as tmp_qa:
            wav_cand = Path(tmp_qa) / "qa_final_16k.wav"
            extract_raw_mono_wav(video_path, wav_cand, sample_rate=16000)
            cap_quality = check_caption_quality(timeline, final_audio_wav=wav_cand)
    except Exception:
        cap_quality = check_caption_quality(timeline)

    if not cap_quality.get("passed", cap_quality.get("ok", True)):
        for c_issue in cap_quality.get("issues", [])[:6]:
            is_integrity_err = "Subtitle integrity" in str(c_issue)
            issues.append(
                QualityCheckIssue(
                    severity="error" if is_integrity_err else "warning",
                    code="SUBTITLE_INTEGRITY_ERROR" if is_integrity_err else "CAPTION_QUALITY",
                    message=str(c_issue),
                )
            )
    pre_integrity = timeline.subtitle_integrity or {}
    if pre_integrity and pre_integrity.get("passed") is False:
        for cat in ("missing", "duplicated", "out_of_order", "misaligned"):
            for item in (pre_integrity.get(cat) or [])[:3]:
                issues.append(
                    QualityCheckIssue(
                        severity="error",
                        code="SUBTITLE_INTEGRITY_ERROR",
                        message=f"Render-stage subtitle integrity ({cat}): {item.get('code')} @ {item.get('time_sec', item.get('clip_id', ''))}",
                    )
                )

    editorial_checks: dict[str, Any] = {}
    # 1. Opening hook clean-start verification
    if timeline.captions:
        first_txt = timeline.captions[0].text.strip()
        first_toks = _tokenize(first_txt)
        hook_dangling = bool(
            (first_toks and first_toks[0] in DANGLING_STARTERS)
            or (first_txt and first_txt[0].islower())
            or first_txt.startswith((",", ";", "-", "—"))
        )
        editorial_checks["hook_clean_start"] = not hook_dangling
        editorial_checks["hook_first_caption"] = first_txt
        if hook_dangling:
            issues.append(
                QualityCheckIssue(
                    severity="warning",
                    code="WEAK_OR_DANGLING_HOOK",
                    message=f"Opening caption starts mid-thought: '{first_txt}'.",
                    timestamp_sec=timeline.captions[0].start_sec,
                )
            )

    # 2. Ending closure verification (via transcript segments if analysis supplied + final caption fallback)
    if timeline.clips and analysis is not None and analysis.transcript.segments:
        last_clip = timeline.clips[-1]
        closure = evaluate_ending_closure(
            analysis.transcript.segments,
            last_clip.source_start_sec,
            last_clip.source_end_sec,
        )
        editorial_checks["ending_closure"] = closure
        if not closure.get("closure_ok", True):
            issues.append(
                QualityCheckIssue(
                    severity="warning",
                    code="ABRUPT_ENDING",
                    message="; ".join(closure.get("issues", [])) or "Closing clip ends before sentence completion.",
                    timestamp_sec=last_clip.timeline_end_sec,
                )
            )
    elif timeline.captions:
        last_txt = timeline.captions[-1].text.strip()
        last_toks = _tokenize(last_txt)
        if (last_toks and last_toks[-1] in INCOMPLETE_ENDINGS) or last_txt.endswith((",", "-", "—", ":")):
            editorial_checks["ending_closure"] = {"closure_ok": False, "spoken_end_text": last_txt}
            issues.append(
                QualityCheckIssue(
                    severity="warning",
                    code="ABRUPT_ENDING",
                    message=f"Closing caption appears incomplete: '{last_txt}'.",
                    timestamp_sec=timeline.captions[-1].end_sec,
                )
            )
        else:
            editorial_checks["ending_closure"] = {"closure_ok": True, "spoken_end_text": last_txt}

    # 3. Overlay resolution upscale & face-box overlap diagnostics
    overlay_warnings: list[str] = []
    for ov in timeline.overlays:
        ov_rect = (ov.x_norm, ov.y_norm, ov.x_norm + ov.w_norm, ov.y_norm + ov.h_norm)
        if analysis is not None and ov.mode in {"corner_sticker_clean", "pip_top_left", "pip_top_right"}:
            host_clip = next(
                (c for c in timeline.clips if c.timeline_start_sec <= ov.timeline_start_sec <= c.timeline_end_sec),
                None,
            )
            if host_clip is not None:
                faces = [
                    (f.x_norm, max(0.0, f.y_norm - 0.06), f.x_norm + f.w_norm, f.y_norm + f.h_norm)
                    for s in analysis.visual_samples
                    if host_clip.source_start_sec - 0.5 <= s.time_sec <= host_clip.source_end_sec + 0.5
                    for f in s.faces
                ]
                ov_area = max(1e-5, ov.w_norm * ov.h_norm)
                max_cov = max((_box_intersection(ov_rect, fb) / ov_area for fb in faces), default=0.0)
                if max_cov > 0.25:
                    msg = f"Overlay {ov.label} overlaps {max_cov:.0%} with detected speaker face/cap area."
                    overlay_warnings.append(msg)
                    issues.append(
                        QualityCheckIssue(
                            severity="warning",
                            code="OVERLAY_FACE_OVERLAP",
                            message=msg,
                            timestamp_sec=ov.timeline_start_sec,
                        )
                    )
        if ov.scene_spec_path and Path(ov.scene_spec_path).is_file() and ov.mode != "cutaway_blur_fill":
            try:
                spec_data = json.loads(Path(ov.scene_spec_path).read_text(encoding="utf-8"))
                src_w = int(spec_data.get("width") or 0)
                tgt_w = int(round(timeline.width * ov.w_norm))
                if src_w > 0 and tgt_w > src_w * 2.2:
                    msg = f"Overlay {ov.label} scales {src_w}px source to {tgt_w}px target (>2.2x)."
                    overlay_warnings.append(msg)
                    issues.append(
                        QualityCheckIssue(
                            severity="info",
                            code="OVERLAY_UPSCALE_RISK",
                            message=msg,
                            timestamp_sec=ov.timeline_start_sec,
                        )
                    )
            except Exception:
                pass
    editorial_checks["overlay_warnings"] = overlay_warnings

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
        measured_lufs=measured_lufs,
        black_frame_intervals=blacks,
        freeze_frame_intervals=freezes,
        caption_count=len(timeline.captions),
        overlay_count=len(timeline.overlays),
        caption_quality=cap_quality,
        editorial_checks=editorial_checks,
        verification_frame_paths=frame_paths,
        issues=issues,
    )


def evaluate_publish_readiness(
    qa_reports: list[QualityAssuranceReport] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate strict publication gate: critical QA/subtitle integrity errors block ready_for_publish."""
    critical_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    checked_artifacts: list[dict[str, Any]] = []

    for raw in qa_reports:
        rep = raw if isinstance(raw, QualityAssuranceReport) else QualityAssuranceReport.model_validate(raw)
        artifact_errors: list[dict[str, Any]] = []
        for iss in rep.issues:
            entry = {
                "artifact_path": rep.artifact_path,
                "code": iss.code,
                "severity": iss.severity,
                "message": iss.message,
                "timestamp_sec": iss.timestamp_sec,
            }
            if iss.severity == "error":
                artifact_errors.append(entry)
                critical_errors.append(entry)
            elif iss.severity == "warning":
                warnings.append(entry)

        integrity = (rep.caption_quality or {}).get("integrity") or {}
        if integrity and integrity.get("passed") is False and not artifact_errors:
            entry = {
                "artifact_path": rep.artifact_path,
                "code": "SUBTITLE_INTEGRITY_FAILED",
                "severity": "error",
                "message": f"Subtitle integrity reported {integrity.get('error_count', 1)} error(s).",
            }
            artifact_errors.append(entry)
            critical_errors.append(entry)

        if not rep.passed and not artifact_errors:
            entry = {
                "artifact_path": rep.artifact_path,
                "code": "QA_FAILED",
                "severity": "error",
                "message": f"QA failed for {Path(rep.artifact_path).name}.",
            }
            artifact_errors.append(entry)
            critical_errors.append(entry)

        checked_artifacts.append(
            {
                "artifact_path": rep.artifact_path,
                "passed": bool(rep.passed) and len(artifact_errors) == 0,
                "error_count": len(artifact_errors),
            }
        )

    ready = bool(checked_artifacts) and len(critical_errors) == 0 and all(a["passed"] for a in checked_artifacts)
    blocking_reasons = [f"{e['code']}: {e['message']}" for e in critical_errors[:10]]
    if not checked_artifacts:
        blocking_reasons = ["No QA reports found for rendered deliverables."]

    return {
        "ready_for_publish": ready,
        "recommended_status": "ready_for_publish" if ready else "requires_review",
        "total_reports": len(checked_artifacts),
        "passed_reports": sum(1 for a in checked_artifacts if a["passed"]),
        "failed_reports": sum(1 for a in checked_artifacts if not a["passed"]),
        "critical_errors": critical_errors,
        "warning_count": len(warnings),
        "blocking_reasons": blocking_reasons,
        "artifacts": checked_artifacts,
    }
