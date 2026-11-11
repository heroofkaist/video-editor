"""Silence detection and local acoustic speaker diarization."""

from __future__ import annotations

import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from symposium_studio.engines.analysis.transcribe import extract_mono_wav
from symposium_studio.models.media import SilenceInterval, SpeakerTurn, TranscriptSegment


def detect_silences(
    media_path: Path,
    duration_sec: float,
    noise_db: float | None = None,
    min_silence_sec: float = 0.55,
) -> list[SilenceInterval]:
    if noise_db is None:
        # Adapt threshold for quiet laptop microphones (mean volume < -28 dB)
        vol_proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(media_path), "-af", "highpass=f=80,volumedetect", "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        m_mean = re.search(r"mean_volume:\s*([-0-9.]+)\s*dB", vol_proc.stderr or "")
        if m_mean:
            mean_db = float(m_mean.group(1))
            eff_noise_db = max(-52.0, min(-30.0, round(mean_db - 2.5, 1))) if mean_db < -28.0 else -34.0
        else:
            eff_noise_db = -34.0
    else:
        eff_noise_db = noise_db

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(media_path),
        "-af",
        f"highpass=f=80,silencedetect=noise={eff_noise_db}dB:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stderr or ""

    silences: list[SilenceInterval] = []
    cur_start: float | None = None
    for line in text.splitlines():
        m_start = re.search(r"silence_start:\s*([0-9.]+)", line)
        if m_start:
            cur_start = float(m_start.group(1))
            continue
        m_end = re.search(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)", line)
        if m_end:
            end_val = float(m_end.group(1))
            dur_val = float(m_end.group(2))
            start_val = cur_start if cur_start is not None else max(0.0, end_val - dur_val)
            silences.append(
                SilenceInterval(
                    start_sec=round(start_val, 3),
                    end_sec=round(end_val, 3),
                    duration_sec=round(dur_val, 3),
                )
            )
            cur_start = None
    if cur_start is not None and duration_sec > cur_start:
        silences.append(
            SilenceInterval(
                start_sec=round(cur_start, 3),
                end_sec=round(duration_sec, 3),
                duration_sec=round(duration_sec - cur_start, 3),
            )
        )
    return silences


def _extract_segment_spectral_signature(samples: np.ndarray, sr: int) -> np.ndarray:
    if len(samples) < 256:
        return np.zeros(12, dtype=np.float32)
    # Compute log-energy across 12 logarithmically spaced frequency bands + ZCR
    win = np.hanning(min(len(samples), 4096))
    chunk = samples[: len(win)] * win
    spec = np.abs(np.fft.rfft(chunk)) + 1e-8
    freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)
    edges = np.geomspace(80.0, max(3800.0, sr / 2.2), num=13)
    bands: list[float] = []
    for i in range(12):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        val = float(np.log1p(np.mean(spec[mask]))) if np.any(mask) else 0.0
        bands.append(val)
    vec = np.asarray(bands, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-6 else vec


def diarize_segments(
    media_path: Path,
    segments: list[TranscriptSegment],
    max_speakers: int = 3,
) -> tuple[list[SpeakerTurn], list[TranscriptSegment]]:
    """Assign speaker IDs using acoustic spectral signatures over transcript windows.

    Avoids gated checkpoints by default while cleanly handling single-speaker vs multi-speaker clips.
    """
    if not segments:
        return [], segments

    with tempfile.TemporaryDirectory(prefix="symposium_diar_") as tmpdir:
        wav_path = extract_mono_wav(media_path, Path(tmpdir) / "diar_16k.wav", sample_rate=16000)
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    sigs: list[np.ndarray] = []
    for seg in segments:
        i0 = max(0, int(seg.start_sec * sr))
        i1 = min(len(samples), int(seg.end_sec * sr))
        sigs.append(_extract_segment_spectral_signature(samples[i0:i1], sr))

    # Incremental cosine-similarity centroid clustering (threshold keeps single-speaker monologue clean)
    centroids: list[np.ndarray] = []
    assignments: list[int] = []
    similarity_threshold = 0.90

    for sig in sigs:
        if not centroids:
            centroids.append(sig)
            assignments.append(0)
            continue
        sims = [float(np.dot(sig, c)) for c in centroids]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] < similarity_threshold and len(centroids) < max_speakers:
            centroids.append(sig)
            assignments.append(len(centroids) - 1)
        else:
            assignments.append(best_idx)
            updated = centroids[best_idx] * 0.8 + sig * 0.2
            n = float(np.linalg.norm(updated))
            centroids[best_idx] = updated / n if n > 1e-6 else updated

    turns: list[SpeakerTurn] = []
    for seg, spk_idx in zip(segments, assignments):
        spk_id = f"speaker_{spk_idx}"
        seg.speaker_id = spk_id
        if turns and turns[-1].speaker_id == spk_id and seg.start_sec - turns[-1].end_sec < 1.2:
            turns[-1].end_sec = seg.end_sec
        else:
            turns.append(
                SpeakerTurn(
                    speaker_id=spk_id,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    confidence=0.82 if len(centroids) == 1 else 0.72,
                )
            )
    return turns, segments
