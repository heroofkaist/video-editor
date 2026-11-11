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

    if preset_id in {"mouse_click", "mechanical_click", "switch_click"}:
        click_cfg = {
            "mouse_click": (0.11, 1850.0, 62.0),
            "mechanical_click": (0.13, 1320.0, 52.0),
            "switch_click": (0.12, 2350.0, 68.0),
        }
        dur, body_hz, decay = click_cfg[preset_id]
        dur = max(0.08, min(dur, duration_sec))
        n = int(round(dur * sample_rate))
        t = np.linspace(0.0, dur, n, endpoint=False)
        snap = sosfilt(
            butter(3, [1200.0, 6800.0], btype="bandpass", fs=sample_rate, output="sos"),
            rng.normal(0.0, 1.0, n),
        ) * np.exp(-95.0 * t)
        body = np.sin(2.0 * np.pi * body_hz * t) * np.exp(-decay * t)
        return _write_wav(out_path, (snap * 0.65 + body * 0.35) * 0.72, sample_rate)

    if preset_id == "vine_boom":
        dur = min(0.85, max(0.55, duration_sec))
        n = int(round(dur * sample_rate))
        t = np.linspace(0.0, dur, n, endpoint=False)
        # Punchy Gen-Z bass thump: rapid pitch drop 135Hz -> 36Hz + transient snap
        freq = 36.0 + 99.0 * np.exp(-16.0 * t)
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
        sub = np.tanh(1.65 * np.sin(phase)) * np.exp(-4.6 * t)
        transient = (rng.normal(0.0, 0.25, n) + 0.45 * np.sin(2.0 * np.pi * 220.0 * t)) * np.exp(-65.0 * t)
        sig = (sub * 0.85 + transient * 0.35)
        return _write_wav(out_path, sig, sample_rate)

    # Subtle ambient harmonic pad bed
    return synthesize_ambient_bed(out_path, duration_sec=max(3.0, duration_sec), mood="tech_focus", sample_rate=sample_rate)


def synthesize_ambient_bed(
    out_path: Path,
    duration_sec: float = 20.0,
    mood: str = "tech_focus",
    sample_rate: int = 48000,
) -> Path:
    """Synthesize a clean, royalty-free continuous ambient underscore bed for sidechain ducking."""
    dur = max(3.0, duration_sec)
    n = int(round(dur * sample_rate))
    t = np.linspace(0.0, dur, n, endpoint=False)
    fade = np.minimum(1.0, np.minimum(t / 0.5, (dur - t) / 0.5))

    if mood == "warm_documentary":
        freqs = (130.81, 196.0, 246.94, 329.63)  # C3 maj7 warmth
    elif mood == "cinematic_pulse":
        freqs = (98.0, 146.83, 220.0, 293.66)    # G2 sus2 energy
    else:
        freqs = (110.0, 164.81, 220.0, 277.18)   # A2 clean tech pad

    lfo = 0.78 + 0.22 * np.sin(2.0 * np.pi * 0.22 * t)
    pad = (
        0.38 * np.sin(2 * np.pi * freqs[0] * t)
        + 0.26 * np.sin(2 * np.pi * freqs[1] * t)
        + 0.20 * np.sin(2 * np.pi * freqs[2] * t)
        + 0.14 * np.sin(2 * np.pi * freqs[3] * t)
    )
    return _write_wav(out_path, pad * fade * lfo * 0.16, sample_rate)
