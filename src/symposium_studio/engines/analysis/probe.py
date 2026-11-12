"""Media probing via FFprobe, PyAV, and FFmpeg volumedetect."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from symposium_studio.models.media import MediaProbeResult
from symposium_studio.storage.workspace import sha256_file


def _parse_fps(rate_str: str | None) -> float:
    if not rate_str or rate_str == "0/0":
        return 30.0
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        d = float(den)
        return round(float(num) / d, 3) if d else 30.0
    return round(float(rate_str), 3)


def _format_aspect(w: int, h: int) -> str:
    if w <= 0 or h <= 0:
        return "unknown"
    g = math.gcd(w, h)
    rw, rh = w // g, h // g
    if abs((w / h) - (16 / 9)) < 0.05:
        return "16:9"
    if abs((w / h) - (9 / 16)) < 0.05:
        return "9:16"
    if abs((w / h) - 1.0) < 0.05:
        return "1:1"
    return f"{rw}:{rh}"


def probe_audio_loudness(media_path: Path) -> tuple[float | None, float | None]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(media_path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stderr or ""
    mean_match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", text)
    max_match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", text)
    mean_db = float(mean_match.group(1)) if mean_match else None
    max_db = float(max_match.group(1)) if max_match else None
    return mean_db, max_db


def probe_media(media_path: Path) -> MediaProbeResult:
    if not media_path.exists():
        raise FileNotFoundError(f"Media file does not exist: {media_path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(media_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    fmt = payload.get("format", {})

    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v_stream is None:
        raise ValueError(f"No video stream found in {media_path}")

    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))
    rot_raw = (v_stream.get("tags") or {}).get("rotate")
    if rot_raw is None:
        for sd in v_stream.get("side_data_list") or []:
            if "rotation" in sd:
                rot_raw = sd.get("rotation")
                break
    try:
        rot_deg = abs(int(round(float(rot_raw or 0.0)))) % 360
    except (TypeError, ValueError):
        rot_deg = 0
    if rot_deg in {90, 270} and width > 0 and height > 0:
        width, height = height, width
    fps = _parse_fps(v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate"))
    duration = float(fmt.get("duration") or v_stream.get("duration") or 0.0)

    # Cross-check duration with PyAV when format metadata omits it
    if duration <= 0.0:
        import av  # type: ignore

        with av.open(str(media_path)) as container:
            if container.duration:
                duration = float(container.duration) / 1_000_000.0

    mean_db, max_db = (None, None)
    if a_stream is not None:
        mean_db, max_db = probe_audio_loudness(media_path)

    return MediaProbeResult(
        path=str(media_path.resolve()),
        sha256=sha256_file(media_path),
        size_bytes=media_path.stat().st_size,
        duration_sec=round(duration, 3),
        width=width,
        height=height,
        fps=fps,
        video_codec=str(v_stream.get("codec_name", "unknown")),
        audio_codec=str(a_stream.get("codec_name")) if a_stream else None,
        audio_sample_rate=int(a_stream.get("sample_rate")) if a_stream and a_stream.get("sample_rate") else None,
        audio_channels=int(a_stream.get("channels")) if a_stream and a_stream.get("channels") else None,
        has_audio=a_stream is not None,
        aspect_ratio=_format_aspect(width, height),
        mean_volume_db=mean_db,
        max_volume_db=max_db,
    )
