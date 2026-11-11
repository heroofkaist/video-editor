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


def _smoothed_rms_db(media_path: Path) -> np.ndarray | None:
    """Extract 100ms smoothed RMS dBFS contour via single 8kHz high-passed FFmpeg PCM pass."""
    pcm_proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-af",
            "highpass=f=80",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
    )
    raw = pcm_proc.stdout
    if pcm_proc.returncode != 0 or len(raw) < 1600:
        return None
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    win = 800  # 100ms windows at 8kHz
    usable = (len(samples) // win) * win
    if usable < win * 4:
        return None
    frames = samples[:usable].reshape(-1, win)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    sm_rms = np.convolve(rms, np.ones(3, dtype=np.float32) / 3.0, mode="same")
    return 20.0 * np.log10(sm_rms + 1e-10)


def _estimate_adaptive_silence_db(media_path: Path) -> float:
    """Compute an adaptive silence threshold (dBFS) between room noise floor and active speech."""
    db = _smoothed_rms_db(media_path)
    if db is not None and len(db) >= 4:
        p20 = float(np.percentile(db, 20))
        p80 = float(np.percentile(db, 80))
        if p80 - p20 >= 5.0:
            thr = p20 + max(3.5, (p80 - p20) * 0.36)
            return round(max(-54.0, min(-24.0, thr)), 1)

    vol_proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(media_path), "-af", "highpass=f=80,volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    m_max = re.search(r"max_volume:\s*([-0-9.]+)\s*dB", vol_proc.stderr or "")
    m_mean = re.search(r"mean_volume:\s*([-0-9.]+)\s*dB", vol_proc.stderr or "")
    if m_max and m_mean:
        max_db = float(m_max.group(1))
        mean_db = float(m_mean.group(1))
        thr = max(mean_db + 2.0, max_db - 22.0) if mean_db < -28.0 else -34.0
        return round(max(-54.0, min(-24.0, thr)), 1)
    if m_mean:
        mean_db = float(m_mean.group(1))
        return round(max(-52.0, min(-26.0, mean_db + 2.5)), 1) if mean_db < -28.0 else -34.0
    return -34.0


def detect_silences(
    media_path: Path,
    duration_sec: float,
    noise_db: float | None = None,
    min_silence_sec: float = 0.55,
) -> list[SilenceInterval]:
    db = _smoothed_rms_db(media_path)
    if db is not None and len(db) >= 6:
        if noise_db is None:
            p20 = float(np.percentile(db, 20))
            p80 = float(np.percentile(db, 80))
            if p80 - p20 >= 5.0:
                eff_thr = round(max(-54.0, min(-24.0, p20 + max(3.5, (p80 - p20) * 0.36))), 1)
            else:
                eff_thr = None
        else:
            eff_thr = float(noise_db)

        if eff_thr is not None:
            silent_mask = db <= eff_thr
            silences_rms: list[SilenceInterval] = []
            run_start: int | None = None
            n_frames = len(silent_mask)
            for idx, is_sil in enumerate(silent_mask):
                if is_sil and run_start is None:
                    run_start = idx
                elif not is_sil and run_start is not None:
                    raw_s = run_start * 0.1
                    raw_e = min(duration_sec, idx * 0.1)
                    # Preserve tiny breathing buffer around speech edges so consonants never clip
                    pad_s = raw_s + 0.08 if raw_s > 0.05 else 0.0
                    pad_e = raw_e - 0.08
                    if pad_e - pad_s >= min_silence_sec:
                        silences_rms.append(
                            SilenceInterval(
                                start_sec=round(pad_s, 3),
                                end_sec=round(pad_e, 3),
                                duration_sec=round(pad_e - pad_s, 3),
                            )
                        )
                    run_start = None
            if run_start is not None:
                raw_s = run_start * 0.1
                raw_e = min(duration_sec, n_frames * 0.1)
                pad_s = raw_s + 0.08 if raw_s > 0.05 else 0.0
                if raw_e - pad_s >= min_silence_sec:
                    silences_rms.append(
                        SilenceInterval(
                            start_sec=round(pad_s, 3),
                            end_sec=round(raw_e, 3),
                            duration_sec=round(raw_e - pad_s, 3),
                        )
                    )
            return silences_rms

    eff_noise_db = _estimate_adaptive_silence_db(media_path) if noise_db is None else noise_db

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
