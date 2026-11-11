"""Procedural WAV sound effects synthesizer for transitions, impacts, risers, UI pops, and ambient beds."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 48000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -0.95, 0.95)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


def synthesize_sfx(preset_id: str, out_path: Path, duration_sec: float = 0.65, sample_rate: int = 48000) -> Path:
    """Synthesize clean, royalty-free studio transition & accent SFX via DSP."""
    rng = np.random.default_rng(seed=42)
    n = max(1, int(round(duration_sec * sample_rate)))
    t = np.linspace(0.0, duration_sec, n, endpoint=False)

    if preset_id == "whoosh_transition":
        noise = rng.normal(0.0, 1.0, n)
        # Split into 4 overlapping swept bandpass envelopes
        env = np.sin(np.pi * (t / max(0.01, duration_sec))) ** 1.6
        sos_low = butter(4, [240.0, 1400.0], btype="bandpass", fs=sample_rate, output="sos")
        sos_high = butter(4, [900.0, 3400.0], btype="bandpass", fs=sample_rate, output="sos")
        w1 = sosfilt(sos_low, noise) * (1.0 - t / duration_sec)
        w2 = sosfilt(sos_high, noise) * np.sin(np.pi * t / duration_sec)
        sig = (w1 * 0.55 + w2 * 0.45) * env * 0.55
        return _write_wav(out_path, sig, sample_rate)

    if preset_id == "sub_impact":
        freq = 120.0 * np.exp(-6.5 * t) + 38.0
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
        env = np.exp(-4.5 * t)
        click = rng.normal(0.0, 0.25, n) * np.exp(-45.0 * t)
        sig = (np.sin(phase) * 0.75 + click) * env * 0.65
        return _write_wav(out_path, sig, sample_rate)

    if preset_id == "riser_tension":
        dur = max(1.2, duration_sec)
        n = int(round(dur * sample_rate))
        t = np.linspace(0.0, dur, n, endpoint=False)
        freq = 110.0 * (2.0 ** (2.5 * t / dur))
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
        noise = sosfilt(butter(3, [600.0, 4200.0], btype="bandpass", fs=sample_rate, output="sos"), rng.normal(0, 0.3, n))
        env = (t / dur) ** 1.8 * np.minimum(1.0, (dur - t) / 0.04)
        sig = (0.45 * np.sin(phase) + 0.55 * noise) * env * 0.45
        return _write_wav(out_path, sig, sample_rate)

    if preset_id == "ui_pop":
        dur = min(0.22, duration_sec)
        n = int(round(dur * sample_rate))
        t = np.linspace(0.0, dur, n, endpoint=False)
        freq = 540.0 + 420.0 * np.exp(-25.0 * t)
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
        sig = np.sin(phase) * np.exp(-24.0 * t) * 0.5
        return _write_wav(out_path, sig, sample_rate)

    # Subtle ambient harmonic pad bed
    dur = max(3.0, duration_sec)
    n = int(round(dur * sample_rate))
    t = np.linspace(0.0, dur, n, endpoint=False)
    fade = np.minimum(1.0, np.minimum(t / 0.4, (dur - t) / 0.4))
    pad = (
        0.35 * np.sin(2 * np.pi * 110.0 * t)
        + 0.25 * np.sin(2 * np.pi * 164.81 * t)
        + 0.20 * np.sin(2 * np.pi * 220.0 * t)
        + 0.15 * np.sin(2 * np.pi * 277.18 * t)
    )
    return _write_wav(out_path, pad * fade * 0.12, sample_rate)
