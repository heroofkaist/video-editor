"""Multilingual local transcription via mlx-whisper, whisper.cpp (whisper-cli Metal), or faster-whisper."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from symposium_studio.models.media import (
    TranscriptQualityReport,
    TranscriptResult,
    TranscriptReviewItem,
    TranscriptSegment,
    WordTimestamp,
)

_PHANTOM_SUBTITLE_RE = re.compile(
    r"(субтитры\s+(делал|создавал|предоставлены|подготовил)|добавил\s+субтитры|dimatorzok|корректор\s+а\.?\s*егорова|subtitles\s+by\b"
    r"|(?:интригующая|динамичная|тревожная|весел[аяо]|грустная|спокойная)\s+музыка\b|музыкальная\s+заставка\b"
    r"|продолжение\s+следует\.?|^\s*[\[(]?(?:music|applause|laughter|silence|музыка|аплодисменты|смех|тишина)[\])]?\.?\s*$)",
    flags=re.IGNORECASE | re.UNICODE,
)
_CYRILLIC_RE = re.compile(r"[а-яёәіңғүұқөһ]", flags=re.IGNORECASE)
_KAZAKH_SPECIFIC_RE = re.compile(r"[әіңғүұқөһ]", flags=re.IGNORECASE)
_LATIN_RE = re.compile(r"[a-z]", flags=re.IGNORECASE)
_OUTLIER_LANG_CODES = {"ko", "ja", "zh", "tr", "de", "bg", "ar", "vi", "th", "pl", "nl", "fr", "es", "it"}


def extract_raw_mono_wav(media_path: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    """Extract dynamic-range-preserving 16kHz mono WAV for acoustic envelope word alignment."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-af",
        "highpass=f=80",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav


def extract_mono_wav(media_path: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-af",
        "highpass=f=80,dynaudnorm=f=150:g=15,loudnorm=I=-16:TP=-1.5:LRA=9",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Fallback without dynamic normalization for silent/tiny synthetic clips
        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
        subprocess.run(fallback_cmd, check=True)
    return out_wav


def _logprob_to_conf(avg_logprob: float | None, no_speech_prob: float | None) -> tuple[float, bool, str | None]:
    lp = -0.35 if avg_logprob is None else float(avg_logprob)
    nsp = 0.05 if no_speech_prob is None else float(no_speech_prob)
    raw_conf = math.exp(max(-4.0, min(0.0, lp))) * (1.0 - min(0.95, max(0.0, nsp)))
    conf = round(max(0.0, min(1.0, raw_conf)), 3)
    reasons: list[str] = []
    if nsp > 0.55:
        reasons.append(f"high_no_speech_prob={nsp:.2f}")
    if lp < -1.0:
        reasons.append(f"low_avg_logprob={lp:.2f}")
    if conf < 0.45:
        reasons.append(f"low_confidence={conf:.2f}")
    return conf, bool(reasons), "; ".join(reasons) if reasons else None


def _collapse_consecutive_word_repeats(words: list[WordTimestamp]) -> tuple[list[WordTimestamp], bool]:
    """Collapse degenerate 3+ consecutive identical words inside word-level timestamps."""
    if len(words) < 4:
        return words, False
    filtered_words: list[WordTimestamp] = []
    prev_norm = ""
    rep_run = 0
    trimmed_repeat = False
    for w in words:
        norm_w = re.sub(r"[^\w]+", "", w.word.casefold())
        if norm_w and norm_w == prev_norm:
            rep_run += 1
            if rep_run >= 2:
                trimmed_repeat = True
                continue
        else:
            prev_norm = norm_w
            rep_run = 1
        filtered_words.append(w)
    return filtered_words, trimmed_repeat


def _shift_segments_offset(segments: list[TranscriptSegment], offset_sec: float) -> list[TranscriptSegment]:
    """Shift segment and word timestamps cleanly by chunk offset_sec."""
    if abs(offset_sec) < 1e-6:
        return segments
    for seg in segments:
        seg.start_sec = round(float(seg.start_sec) + offset_sec, 3)
        seg.end_sec = round(max(seg.start_sec + 0.05, float(seg.end_sec) + offset_sec), 3)
        for w in seg.words:
            w.start_sec = round(float(w.start_sec) + offset_sec, 3)
            w.end_sec = round(max(w.start_sec + 0.02, float(w.end_sec) + offset_sec), 3)
    return segments


def _convert_whisper_segments(
    raw_segments: list[dict[str, Any]],
    detected_language: str,
) -> tuple[list[TranscriptSegment], list[str]]:
    segments: list[TranscriptSegment] = []
    uncertain_ids: list[str] = []

    for seg in raw_segments:
        raw_text = str(seg.get("text", "")).strip()
        if not raw_text:
            continue
        text, trimmed_repeat = _clean_hallucinated_repeats(raw_text)
        if not text:
            continue
        start_sec = round(float(seg.get("start", 0.0)), 3)
        end_sec = round(max(start_sec + 0.05, float(seg.get("end", start_sec + 0.5))), 3)
        if segments and text.casefold() == segments[-1].text.casefold():
            segments[-1].end_sec = max(segments[-1].end_sec, end_sec)
            continue

        conf, low_conf, reason = _logprob_to_conf(
            seg.get("avg_logprob"),
            seg.get("no_speech_prob"),
        )
        words_raw = seg.get("words") or []
        words: list[WordTimestamp] = []
        if words_raw:
            for w in words_raw:
                w_text = str(w.get("word", "")).strip()
                if not w_text:
                    continue
                prob = float(w.get("probability", conf))
                words.append(
                    WordTimestamp(
                        word=w_text,
                        start_sec=round(float(w.get("start", start_sec)), 3),
                        end_sec=round(float(w.get("end", end_sec)), 3),
                        probability=round(prob, 3),
                        low_confidence=prob < 0.45,
                    )
                )
        else:
            # Estimate word slots uniformly ONLY from verbatim spoken text tokens when backend omits word timings
            tokens = text.split()
            if tokens:
                step = (end_sec - start_sec) / len(tokens)
                for w_i, tok in enumerate(tokens):
                    words.append(
                        WordTimestamp(
                            word=tok,
                            start_sec=round(start_sec + w_i * step, 3),
                            end_sec=round(start_sec + (w_i + 1) * step, 3),
                            probability=conf,
                            low_confidence=low_conf,
                        )
                    )

        words, word_trimmed = _collapse_consecutive_word_repeats(words)
        trimmed_repeat = trimmed_repeat or word_trimmed
        if trimmed_repeat:
            conf = min(conf, 0.42)
            low_conf = True
            reason = "collapsed_repetitive_hallucination"

        seg_id = f"seg_{len(segments):04d}"
        if low_conf:
            uncertain_ids.append(seg_id)

        segments.append(
            TranscriptSegment(
                segment_id=seg_id,
                start_sec=start_sec,
                end_sec=end_sec,
                text=text,
                language=str(seg.get("language") or detected_language),
                confidence=conf,
                low_confidence=low_conf,
                low_confidence_reason=reason,
                words=words,
                verification_status="uncertain_needs_review" if low_conf else "asr_raw",
            )
        )
    return segments, uncertain_ids


def find_whisper_cpp_model(preferred_hint: str = "") -> Path | None:
    """Locate an installed GGML model file for whisper.cpp."""
    env_path = os.environ.get("SYMPOSIUM_WHISPER_GGML")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    search_dirs = [
        Path.home() / ".cache" / "whisper-cpp",
        Path.home() / ".cache" / "whisper",
        Path("/opt/homebrew/share/whisper-cpp"),
    ]
    hint_lower = preferred_hint.lower()
    if "tiny" in hint_lower:
        preferred_names = ["ggml-tiny.bin", "ggml-base.bin", "ggml-small.bin"]
    elif "base" in hint_lower:
        preferred_names = [
            "ggml-base.bin",
            "ggml-large-v3-turbo-q5_0.bin",
            "ggml-large-v3-turbo.bin",
            "ggml-small.bin",
            "ggml-tiny.bin",
        ]
    else:
        preferred_names = [
            "ggml-large-v3-q5_0.bin",
            "ggml-large-v3.bin",
            "ggml-large-v3-turbo-q5_0.bin",
            "ggml-large-v3-turbo.bin",
            "ggml-medium.bin",
            "ggml-small.bin",
            "ggml-base.bin",
            "ggml-tiny.bin",
        ]
    for d in search_dirs:
        if not d.exists():
            continue
        for fname in preferred_names:
            cand = d / fname
            if cand.is_file() and cand.stat().st_size > 1_000_000:
                return cand
        bins = sorted(d.glob("ggml-*.bin"))
        if bins:
            return bins[0]
    return None


def _tokens_to_words(tokens: list[dict[str, Any]], seg_start: float, seg_end: float) -> tuple[list[WordTimestamp], float]:
    """Merge BPE subword tokens from whisper.cpp JSON into whole WordTimestamp objects and compute avg prob."""
    words: list[WordTimestamp] = []
    probs_all: list[float] = []

    cur_text = ""
    cur_start = seg_start
    cur_end = seg_start
    cur_probs: list[float] = []

    def _flush() -> None:
        nonlocal cur_text, cur_probs
        cleaned = cur_text.strip()
        if cleaned:
            prob = sum(cur_probs) / len(cur_probs) if cur_probs else 0.75
            words.append(
                WordTimestamp(
                    word=cleaned,
                    start_sec=round(cur_start, 3),
                    end_sec=round(max(cur_start + 0.02, cur_end), 3),
                    probability=round(max(0.0, min(1.0, prob)), 3),
                    low_confidence=prob < 0.45,
                )
            )
        cur_text = ""
        cur_probs = []

    for tok in tokens:
        raw = str(tok.get("text", "")).replace("\u2581", " ")
        stripped = raw.strip()
        if not stripped:
            if cur_text:
                _flush()
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        offs = tok.get("offsets") or {}
        t0 = float(offs.get("from", int(seg_start * 1000))) / 1000.0
        t1 = float(offs.get("to", int(t0 * 1000))) / 1000.0
        p = float(tok.get("p", 0.75))
        probs_all.append(p)

        # Flush current word on leading whitespace so adjacent words never fuse across subwords
        if raw[0].isspace() and cur_text:
            _flush()
            cur_start = t0
            cur_end = t1
            cur_text = raw
            cur_probs = [p]
        else:
            if not cur_text:
                cur_start = t0
            cur_end = max(cur_end, t1)
            cur_text += raw
            cur_probs.append(p)

    _flush()
    seg_conf = sum(probs_all) / len(probs_all) if probs_all else 0.75
    return words, round(max(0.0, min(1.0, seg_conf)), 3)


def _plan_wav_chunks(
    wav_path: Path,
    target_sec: float = 16.0,
    min_sec: float = 9.5,
    max_sec: float = 22.0,
) -> list[tuple[float, float]]:
    """Split a mono PCM WAV at genuine smoothed low-energy pauses so code-switching turns stay clean."""
    import array
    import math
    import wave

    if not wav_path.exists():
        return []
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            if sr <= 0 or wf.getsampwidth() != 2 or wf.getnchannels() != 1 or n_frames <= 0:
                return []
            total_sec = n_frames / sr
            if total_sec <= max_sec + 2.0:
                return [(0.0, round(total_sec, 3))]
            win_frames = max(1, sr // 10)  # 100ms RMS windows
            rms: list[float] = []
            while True:
                raw = wf.readframes(win_frames)
                if not raw:
                    break
                samples = array.array("h")
                samples.frombytes(raw)
                if samples:
                    sq = sum(s * s for s in samples)
                    rms.append(math.sqrt(sq / len(samples)))
    except (wave.Error, OSError):
        return []

    if not rms:
        return [(0.0, round(total_sec, 3))]

    # Smooth over 5-window (~0.5s) neighborhoods so real sentence pauses outrank tiny syllable gaps
    n_win = len(rms)
    smoothed = [
        sum(rms[max(0, i - 2) : min(n_win, i + 3)]) / max(1, min(n_win, i + 3) - max(0, i - 2))
        for i in range(n_win)
    ]

    chunks: list[tuple[float, float]] = []
    cur_idx = 0
    win_sec = win_frames / sr
    min_wins = max(1, int(min_sec / win_sec))
    max_wins = max(min_wins + 1, int(max_sec / win_sec))
    target_wins = target_sec / win_sec
    tail_guard = max(1, int(4.5 / win_sec))

    while cur_idx < n_win:
        if n_win - cur_idx <= max_wins + 2:
            chunks.append((round(cur_idx * win_sec, 3), round(total_sec, 3)))
            break
        lo = min(n_win - tail_guard, cur_idx + min_wins)
        hi = min(n_win - tail_guard, cur_idx + max_wins)
        if lo >= hi:
            chunks.append((round(cur_idx * win_sec, 3), round(total_sec, 3)))
            break
        span = max(1.0, float(hi - lo))
        best_idx = min(
            range(lo, hi + 1),
            key=lambda i: smoothed[i] * (1.0 + 0.12 * abs((i - cur_idx) - target_wins) / span),
        )
        chunks.append((round(cur_idx * win_sec, 3), round(best_idx * win_sec, 3)))
        cur_idx = best_idx
    return chunks


def _write_wav_slice(wav_path: Path, out_wav: Path, start_sec: float, end_sec: float) -> None:
    """Copy a [start_sec, end_sec] slice from a PCM WAV into out_wav."""
    import wave

    with wave.open(str(wav_path), "rb") as src:
        params = src.getparams()
        sr = params.framerate
        i0 = max(0, min(params.nframes, int(round(start_sec * sr))))
        i1 = max(i0, min(params.nframes, int(round(end_sec * sr))))
        src.setpos(i0)
        raw = src.readframes(i1 - i0)
    with wave.open(str(out_wav), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(raw)


def _clean_hallucinated_repeats(text: str) -> tuple[str, bool]:
    """Collapse degenerate consecutive Whisper repetition loops and phantom subtitle credits."""
    had_phantom = bool(_PHANTOM_SUBTITLE_RE.search(text))
    cleaned = _PHANTOM_SUBTITLE_RE.sub("", text).strip()
    before_rep = cleaned
    cleaned = re.sub(
        r"\b(\w+(?:\s+\w+)?)(?:\s*[,.;:!?-]?\s*\1){3,}\b",
        r"\1 \1",
        cleaned,
        flags=re.IGNORECASE | re.UNICODE,
    )
    # Also handle Cyrillic/spaced multi-word repetition like "Два дня, три недели" loops
    cleaned = re.sub(
        r"([\w\s]{4,30}?)(?:\s*[,.;:!?]\s*\1){2,}",
        r"\1",
        cleaned,
        flags=re.IGNORECASE | re.UNICODE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;-")
    return cleaned, (had_phantom or cleaned != before_rep.strip() or cleaned != text.strip())


def _run_whisper_cpp_chunk(
    chunk_wav: Path,
    out_prefix: Path,
    whisper_cli: str,
    model_path: Path,
    language: str | None,
    offset_sec: float,
    start_seg_idx: int,
) -> tuple[str, float, list[TranscriptSegment], list[str]]:
    cmd = [
        whisper_cli,
        "-m", str(model_path),
        "-f", str(chunk_wav),
        "-l", language if language else "auto",
        "-mc", "0",
        "-sns",
        "-ojf",
        "-of", str(out_prefix),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    json_path = Path(f"{out_prefix}.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    detected_lang = str((payload.get("result") or {}).get("language") or language or "unknown")
    lang_prob = 0.95 if language else 0.0
    stderr_txt = (proc.stderr or "") + "\n" + (proc.stdout or "")
    m_lp = re.search(r"auto-detected language:\s*\S+\s*\(p\s*=\s*([0-9.]+)\)", stderr_txt)
    if m_lp:
        lang_prob = round(float(m_lp.group(1)), 3)

    segments: list[TranscriptSegment] = []
    uncertain_ids: list[str] = []
    seg_idx = start_seg_idx
    for item in payload.get("transcription") or []:
        raw_text = str(item.get("text", "")).strip()
        if not raw_text:
            continue
        text, trimmed_repeat = _clean_hallucinated_repeats(raw_text)
        if not text:
            continue
        offs = item.get("offsets") or {}
        s0 = round(offset_sec + float(offs.get("from", 0)) / 1000.0, 3)
        s1 = round(
            offset_sec + max((s0 - offset_sec) + 0.05, float(offs.get("to", int((s0 - offset_sec + 0.5) * 1000))) / 1000.0),
            3,
        )
        if segments and text.casefold() == segments[-1].text.casefold():
            segments[-1].end_sec = max(segments[-1].end_sec, s1)
            continue
        raw_tokens = item.get("tokens") or []
        words, seg_conf = _tokens_to_words(raw_tokens, s0 - offset_sec, s1 - offset_sec)
        for w in words:
            w.start_sec = round(w.start_sec + offset_sec, 3)
            w.end_sec = round(w.end_sec + offset_sec, 3)
        words, word_trimmed = _collapse_consecutive_word_repeats(words)
        trimmed_repeat = trimmed_repeat or word_trimmed
        if trimmed_repeat:
            seg_conf = min(seg_conf, 0.42)
        low_conf = seg_conf < 0.48 or trimmed_repeat
        seg_id = f"seg_{seg_idx:04d}"
        seg_idx += 1
        if low_conf:
            uncertain_ids.append(seg_id)
        segments.append(
            TranscriptSegment(
                segment_id=seg_id,
                start_sec=s0,
                end_sec=s1,
                text=text,
                language=detected_lang,
                confidence=seg_conf,
                low_confidence=low_conf,
                low_confidence_reason=(
                    "collapsed_repetitive_hallucination"
                    if trimmed_repeat
                    else (f"low_token_confidence={seg_conf:.2f}" if low_conf else None)
                ),
                words=words,
                verification_status="uncertain_needs_review" if low_conf else "asr_raw",
            )
        )
    return detected_lang, lang_prob, segments, uncertain_ids


def _token_repetition_metrics(words_low: list[str]) -> tuple[bool, bool]:
    """Return (has_adjacent_dup, is_repeat_loop) without guessing or rewriting words."""
    has_adj = False
    if len(words_low) >= 2:
        for i in range(len(words_low) - 1):
            if words_low[i] == words_low[i + 1] and len(words_low[i]) >= 3:
                has_adj = True
                break
    unique_cnt = len(set(words_low))
    is_loop = (len(words_low) >= 3 and unique_cnt / len(words_low) <= 0.55) or (
        len(words_low) == 2 and words_low[0] == words_low[1] and len(words_low[0]) >= 4
    )
    return has_adj, is_loop


def _chunk_quality_score(lang: str, segments: list[TranscriptSegment]) -> float:
    """Rank competing transcription hypotheses while penalizing repetition loops and sparse stubs."""
    if not segments:
        return -1.0
    total_chars = sum(max(1, len(s.text)) for s in segments)
    mean_conf = sum(s.confidence * max(1, len(s.text)) for s in segments) / total_chars
    rep_pen = 0.0
    sparse_pen = 0.0
    for s in segments:
        if s.low_confidence_reason == "collapsed_repetitive_hallucination":
            rep_pen += 0.25
        words_low = [re.sub(r"[^\w]+", "", w.casefold()) for w in s.text.split() if re.sub(r"[^\w]+", "", w)]
        _, is_loop = _token_repetition_metrics(words_low)
        if is_loop:
            rep_pen += 0.45
        seg_dur = max(0.0, float(s.end_sec) - float(s.start_sec))
        if seg_dur >= 3.0 and len(words_low) <= 2:
            sparse_pen += 0.45

    outlier_pen = 0.20 if lang in _OUTLIER_LANG_CODES else 0.0
    text_all = " ".join(s.text for s in segments)
    mixed_tok_pen = 0.0
    for tok in re.findall(r"[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+", text_all):
        if _LATIN_RE.search(tok) and _CYRILLIC_RE.search(tok):
            mixed_tok_pen += 0.08
    return mean_conf - rep_pen - sparse_pen - outlier_pen - mixed_tok_pen


def _detect_segment_anomalies(seg: TranscriptSegment) -> list[str]:
    """Detect non-invented phonetic, mixed-script, repetition, or low-confidence anomalies on a segment."""
    flags: list[str] = []
    txt = seg.text.strip()
    if seg.low_confidence_reason == "collapsed_repetitive_hallucination":
        flags.append("hallucination_repeat_or_phantom_cleaned")
    if seg.confidence < 0.56:
        flags.append("low_token_confidence")
    elif seg.confidence < 0.68:
        flags.append("moderate_uncertainty")

    tokens = re.findall(r"[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+", txt)
    for tok in tokens:
        if _LATIN_RE.search(tok) and _CYRILLIC_RE.search(tok):
            flags.append("mixed_cyrillic_latin_token")
            break

    # Detect Kazakh/Russian phonetic garble artifacts without replacing text
    low = txt.casefold()
    if _KAZAKH_SPECIFIC_RE.search(low):
        suspicious_clusters = (
            "ұйы", "ініст", "құй", "ыңшы", "ңшы", "кә ", "әсінің әсінің",
            "ғайтып", "панде", "ажыд", "бақаш", "стиж", "сүрде", "қолсаңд", "әйтсет",
        )
        if any(cl in low for cl in suspicious_clusters) or (
            re.search(r"[қғңұүіәө][ъщ]", low) is not None
        ):
            flags.append("possible_code_switch_phonetic_garble")

    words_low = [re.sub(r"[^\w]+", "", w.casefold()) for w in txt.split()]
    words_low = [w for w in words_low if w]
    has_adj, is_loop = _token_repetition_metrics(words_low)
    if has_adj:
        flags.append("adjacent_duplicate_word")
    if is_loop:
        flags.append("single_token_repeat_loop")

    seg_dur = max(0.0, float(seg.end_sec) - float(seg.start_sec))
    if seg_dur >= 3.0 and len(words_low) <= 2:
        flags.append("sparse_words_over_long_window")

    if len(words_low) <= 2 and seg.confidence < 0.65:
        flags.append("short_uncertain_fragment")

    return list(dict.fromkeys(flags))


def audit_and_verify_transcript(result: TranscriptResult) -> TranscriptResult:
    """Second-stage deterministic audit: flags anomalous segments without inventing unverified words."""
    uncertain_ids: list[str] = []
    review_queue: list[TranscriptReviewItem] = []
    high_conf_count = 0
    confirmed_count = 0
    anomalous_count = 0
    cleaned_count = 0
    lang_chars: dict[str, int] = {}

    audited_segments = []
    for seg in result.segments:
        if seg.verification_status == "human_confirmed":
            confirmed_count += 1
            seg.anomaly_flags = []
            seg.low_confidence = False
            high_conf_count += 1
        else:
            cleaned_txt, was_cleaned = _clean_hallucinated_repeats(seg.text)
            if was_cleaned and not cleaned_txt:
                cleaned_count += 1
                continue
            if was_cleaned and cleaned_txt:
                seg.text = cleaned_txt
                if not seg.low_confidence_reason:
                    seg.low_confidence_reason = "collapsed_repetitive_hallucination"
            prev_energy_flags = [f for f in seg.anomaly_flags if f == "low_speech_energy_hallucination"]
            flags = _detect_segment_anomalies(seg)
            for pf in prev_energy_flags:
                if pf not in flags:
                    flags.append(pf)
            if was_cleaned and "hallucination_repeat_or_phantom_cleaned" not in flags:
                flags.insert(0, "hallucination_repeat_or_phantom_cleaned")
            seg.anomaly_flags = flags
            if "hallucination_repeat_or_phantom_cleaned" in flags:
                cleaned_count += 1
            if flags:
                anomalous_count += 1
            needs_review = (
                seg.low_confidence
                or seg.confidence < 0.56
                or "mixed_cyrillic_latin_token" in flags
                or "possible_code_switch_phonetic_garble" in flags
                or "single_token_repeat_loop" in flags
                or "sparse_words_over_long_window" in flags
                or "low_speech_energy_hallucination" in flags
                or len(flags) >= 2
            )
            if needs_review:
                seg.low_confidence = True
                if not seg.low_confidence_reason:
                    seg.low_confidence_reason = ",".join(flags) if flags else f"confidence={seg.confidence:.2f}"
                if seg.verification_status not in {"alternate_hypothesis_selected"}:
                    seg.verification_status = "uncertain_needs_review"
                uncertain_ids.append(seg.segment_id)
                review_queue.append(
                    TranscriptReviewItem(
                        segment_id=seg.segment_id,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        text=seg.text,
                        language=seg.language,
                        confidence=seg.confidence,
                        reasons=flags or [seg.low_confidence_reason or "low_confidence"],
                        candidate_alternatives=list(seg.candidate_alternatives),
                        verification_status=seg.verification_status,
                    )
                )
            else:
                if seg.confidence >= 0.72 and not flags:
                    seg.verification_status = (
                        "alternate_hypothesis_selected"
                        if seg.verification_status == "alternate_hypothesis_selected"
                        else "high_confidence"
                    )
                    high_conf_count += 1
                elif seg.verification_status == "asr_raw":
                    seg.verification_status = "high_confidence" if not flags else "uncertain_needs_review"
                    if not flags:
                        high_conf_count += 1

        audited_segments.append(seg)
        seg_lang = seg.language or result.detected_language or "unknown"
        if seg_lang != "unknown":
            lang_chars[seg_lang] = lang_chars.get(seg_lang, 0) + max(1, len(seg.text))

    result.segments = audited_segments
    ordered_langs = [k for k, _ in sorted(lang_chars.items(), key=lambda kv: kv[1], reverse=True)]
    if not ordered_langs and result.languages_observed:
        ordered_langs = list(result.languages_observed)
    mean_conf = (
        round(sum(s.confidence for s in result.segments) / len(result.segments), 3)
        if result.segments
        else 0.0
    )
    all_text = " ".join(s.text for s in result.segments)
    has_kk_chars = bool(_KAZAKH_SPECIFIC_RE.search(all_text))
    lat_words = re.findall(r"\b[A-Za-z]{2,}\b", all_text)
    has_lat_chars = bool(lat_words)
    if has_lat_chars and "en" not in ordered_langs:
        ordered_langs.append("en")
    if has_kk_chars and "kk" not in ordered_langs:
        ordered_langs.append("kk")
    code_switched = len(ordered_langs) > 1 or (has_kk_chars and ("ru" in ordered_langs or has_lat_chars))

    notes: list[str] = []
    if review_queue:
        notes.append(
            f"{len(review_queue)} segment(s) flagged for human review due to low confidence, code-switch phonetic ambiguity, or repetition cleanup."
        )
    if code_switched:
        notes.append(f"Multilingual code-switching detected across {', '.join(ordered_langs) or 'multiple scripts'}.")
    if not notes:
        notes.append("All segments passed automated confidence and script consistency checks.")

    result.uncertain_segment_ids = list(dict.fromkeys(uncertain_ids))
    result.languages_observed = ordered_langs or result.languages_observed
    result.quality_report = TranscriptQualityReport(
        total_segments=len(result.segments),
        high_confidence_count=high_conf_count,
        uncertain_count=len(review_queue),
        human_confirmed_count=confirmed_count,
        anomalous_segment_count=anomalous_count,
        hallucinations_cleaned=cleaned_count,
        mean_confidence=mean_conf,
        code_switching_detected=code_switched,
        languages_observed=list(result.languages_observed),
        review_queue=review_queue,
        notes=notes,
    )
    return result


def apply_transcript_corrections(
    result: TranscriptResult,
    corrections: dict[str, str | dict[str, object]],
    language_overrides: dict[str, str] | None = None,
    mark_confirmed: bool = True,
) -> TranscriptResult:
    """Apply manual or verified segment edits cleanly without re-running ASR, redistributing word timestamps."""
    updated = result.model_copy(deep=True)
    lang_map = language_overrides or {}
    for seg in updated.segments:
        if seg.segment_id not in corrections and seg.segment_id not in lang_map:
            continue
        if seg.segment_id in lang_map:
            seg.language = str(lang_map[seg.segment_id]).strip() or seg.language
        if seg.segment_id in corrections:
            raw_corr = corrections[seg.segment_id]
            if isinstance(raw_corr, dict):
                if raw_corr.get("start_sec") is not None:
                    seg.start_sec = max(0.0, float(raw_corr["start_sec"]))
                if raw_corr.get("end_sec") is not None:
                    seg.end_sec = max(seg.start_sec + 0.12, float(raw_corr["end_sec"]))
                if raw_corr.get("language"):
                    seg.language = str(raw_corr["language"]).strip() or seg.language
                text_val = raw_corr.get("text")
                new_text = re.sub(r"\s{2,}", " ", str(text_val if text_val is not None else seg.text)).strip()
            else:
                new_text = re.sub(r"\s{2,}", " ", str(raw_corr)).strip()
            if new_text:
                seg.text = new_text
                # Re-align word timestamps proportionally inside segment [start_sec, end_sec]
                tokens = [tok for tok in new_text.split() if tok.strip()]
                s0 = float(seg.start_sec)
                s1 = max(s0 + 0.12, float(seg.end_sec))
                pad = min(0.06, (s1 - s0) * 0.05)
                span_start = s0 + pad
                span_end = max(span_start + 0.08, s1 - pad)
                weights = [_syllable_word_weight(tok) for tok in tokens]
                total_w = sum(weights) or float(len(tokens) or 1)
                cur_t = span_start
                new_words: list[WordTimestamp] = []
                for tok, w in zip(tokens, weights):
                    dur = (span_end - span_start) * (w / total_w)
                    w_end = min(span_end, cur_t + dur)
                    new_words.append(
                        WordTimestamp(
                            word=tok,
                            start_sec=round(cur_t, 3),
                            end_sec=round(max(cur_t + 0.04, w_end), 3),
                            probability=1.0 if mark_confirmed else max(0.85, seg.confidence),
                        )
                    )
                    cur_t = w_end
                seg.words = new_words
        if mark_confirmed:
            seg.confidence = 1.0
            seg.low_confidence = False
            seg.low_confidence_reason = None
            seg.anomaly_flags = []
            seg.verification_status = "human_confirmed"
    return audit_and_verify_transcript(updated)


_VOWEL_RE = re.compile(r"[аеёиоуыэюяәөүұіaeiouy]", flags=re.IGNORECASE)


def _syllable_word_weight(token: str) -> float:
    clean = re.sub(r"[^\w\-]+", "", token.strip())
    if not clean:
        return 1.0
    vowels = len(_VOWEL_RE.findall(clean))
    digits = sum(ch.isdigit() for ch in clean)
    if vowels == 0 and digits > 0:
        vowels = max(1, digits)
    elif vowels == 0 and clean.isupper() and len(clean) <= 5:
        vowels = max(1, len(clean))
    return max(1.2, float(max(1, vowels)) * 1.15 + 0.18 * len(clean))


def _load_wav_mono_float(wav_path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sr <= 0 or not raw or sampwidth != 2:
        return 0, np.empty(0, dtype=np.float32)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    return sr, pcm


def compute_audio_envelope(media_or_wav: Path) -> dict[str, Any] | None:
    """Compute smoothed RMS speech energy envelope & adaptive log-hysteresis thresholds from clean audio."""
    src = Path(media_or_wav)
    if not src.exists():
        return None
    if src.suffix.lower() == ".wav":
        sr, samples = _load_wav_mono_float(src)
    else:
        with tempfile.TemporaryDirectory(prefix="symp_env_") as tmpdir:
            tmp_wav = extract_raw_mono_wav(src, Path(tmpdir) / "raw_16k.wav")
            sr, samples = _load_wav_mono_float(tmp_wav)

    if sr <= 0 or len(samples) < int(0.05 * sr):
        return None
    hop = max(1, int(round(0.010 * sr)))
    win = max(hop, int(round(0.025 * sr)))
    if len(samples) < win:
        return None
    frame_dt = hop / float(sr)

    cs = np.concatenate(([0.0], np.cumsum(samples.astype(np.float64) ** 2)))
    starts = np.arange(0, len(samples) - win + 1, hop)
    rms = np.sqrt(np.maximum(0.0, (cs[starts + win] - cs[starts]) / float(win)))
    t_center = (starts + win * 0.5) / float(sr)

    kernel = np.array([0.25, 0.75, 1.0, 0.75, 0.25], dtype=np.float64)
    kernel /= kernel.sum()
    smooth_rms = np.convolve(rms, kernel, mode="same")

    max_rms = float(np.max(smooth_rms))
    if max_rms < 1e-6:
        return None

    floor = max(1e-5, float(np.percentile(smooth_rms, 18)))
    active_global = smooth_rms[smooth_rms > max(floor * 2.4, max_rms * 0.12)]
    speech_med = float(np.median(active_global)) if len(active_global) > 0 else max_rms
    log_f = math.log(floor)
    log_s = math.log(max(speech_med, floor * 1.25))
    global_hi = max(floor * 2.3, math.exp(log_f + 0.68 * (log_s - log_f)))
    global_lo = max(floor * 1.65, math.exp(log_f + 0.47 * (log_s - log_f)))
    if global_hi < global_lo * 1.20:
        global_hi = global_lo * 1.20

    return {
        "sr": sr,
        "duration_sec": len(samples) / float(sr),
        "frame_dt": frame_dt,
        "t_center": t_center,
        "smooth_rms": smooth_rms,
        "floor": floor,
        "speech_med": speech_med,
        "global_lo": global_lo,
        "global_hi": global_hi,
        "max_rms": max_rms,
    }


def detect_speech_islands_in_window(
    env: dict[str, Any],
    start_sec: float,
    end_sec: float,
    min_pause_sec: float = 0.10,
    pad_sec: float = 0.04,
) -> dict[str, Any]:
    """Detect active speech frame islands inside [start_sec, end_sec] using dual-hysteresis thresholds."""
    t_center: np.ndarray = env["t_center"]
    smooth_rms: np.ndarray = env["smooth_rms"]
    frame_dt = float(env["frame_dt"])
    floor = float(env["floor"])
    speech_med = float(env["speech_med"])
    global_hi = float(env["global_hi"])
    global_lo = float(env["global_lo"])
    min_pause_frames = max(1, int(round(min_pause_sec / frame_dt)))

    s0 = max(0.0, float(start_sec))
    s1 = max(s0 + 0.08, float(end_sec))
    k0 = max(0, int(np.searchsorted(t_center, max(0.0, s0 - pad_sec))))
    k1 = min(len(smooth_rms), int(np.searchsorted(t_center, s1 + pad_sec, side="right")))

    if k1 - k0 < 3:
        return {
            "islands_idx": [],
            "islands_sec": [],
            "mask": None,
            "k0": k0,
            "k1": k1,
            "seg_p92": speech_med,
            "lo_thr": global_lo,
            "peak_rms": 0.0,
        }

    seg_slice = smooth_rms[k0:k1]
    peak_rms = float(np.max(seg_slice))
    seg_p92 = max(floor * 1.5, float(np.percentile(seg_slice, 92)))
    hi_thr = max(floor * 2.2, min(global_hi, seg_p92 * 0.42))
    lo_thr = max(floor * 1.65, min(global_lo, hi_thr * 0.72))
    if hi_thr < lo_thr * 1.18:
        hi_thr = lo_thr * 1.18

    mask = np.zeros(len(seg_slice), dtype=bool)
    active = False
    for idx in range(len(seg_slice)):
        val = float(seg_slice[idx])
        if not active and val >= hi_thr:
            active = True
            mask[idx] = True
            for bk in range(idx - 1, max(-1, idx - 5), -1):
                if float(seg_slice[bk]) >= lo_thr:
                    mask[bk] = True
                else:
                    break
        elif active:
            if val >= lo_thr:
                mask[idx] = True
            else:
                active = False

    n_m = len(mask)
    i = 0
    while i < n_m:
        if mask[i]:
            j = i
            while j < n_m and mask[j]:
                j += 1
            gap_end = j
            while gap_end < n_m and not mask[gap_end]:
                gap_end += 1
            if gap_end < n_m and (gap_end - j) < min_pause_frames:
                mask[j:gap_end] = True
                continue
            i = gap_end
        else:
            i += 1

    raw_islands: list[tuple[int, int]] = []
    i = 0
    while i < n_m:
        if mask[i]:
            j = i
            while j < n_m and mask[j]:
                j += 1
            raw_islands.append((k0 + i, k0 + j))
            i = j
        else:
            i += 1

    long_islands = [isl for isl in raw_islands if (isl[1] - isl[0]) >= 4]
    islands = long_islands if long_islands else raw_islands
    islands_sec = [
        (
            round(max(s0, float(t_center[ia]) - 0.5 * frame_dt), 3),
            round(min(s1, float(t_center[ib - 1]) + 0.5 * frame_dt), 3),
        )
        for ia, ib in islands
    ]
    return {
        "islands_idx": islands,
        "islands_sec": islands_sec,
        "mask": mask,
        "k0": k0,
        "k1": k1,
        "seg_p92": seg_p92,
        "lo_thr": lo_thr,
        "peak_rms": peak_rms,
    }


def align_words_to_envelope(
    tokens: list[str],
    word_weights: list[float],
    env: dict[str, Any],
    start_sec: float,
    end_sec: float,
    min_pause_sec: float = 0.10,
    probs: list[float] | None = None,
    pad_sec: float = 0.04,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map tokens onto detected acoustic speech islands inside [start_sec, end_sec]."""
    s0 = max(0.0, float(start_sec))
    s1 = max(s0 + 0.10, float(end_sec))
    det = detect_speech_islands_in_window(env, s0, s1, min_pause_sec=min_pause_sec, pad_sec=pad_sec)
    islands: list[tuple[int, int]] = det["islands_idx"]
    token_probs = probs if (probs and len(probs) == len(tokens)) else [0.95] * len(tokens)

    if not tokens:
        return [], det

    if not islands:
        pad = min(0.05, (s1 - s0) * 0.04)
        span_start = s0 + pad
        span_end = max(span_start + 0.08, s1 - pad)
        total_w = sum(word_weights) or 1.0
        cur_t = span_start
        out: list[dict[str, Any]] = []
        for tok, ww, pr in zip(tokens, word_weights, token_probs):
            dur = (span_end - span_start) * (ww / total_w)
            w_end = min(span_end, cur_t + dur)
            out.append(
                {
                    "word": tok,
                    "start_sec": round(cur_t, 3),
                    "end_sec": round(max(cur_t + 0.04, w_end), 3),
                    "probability": round(float(pr), 3),
                }
            )
            cur_t = w_end
        return out, det

    smooth_rms: np.ndarray = env["smooth_rms"]
    t_center: np.ndarray = env["t_center"]
    frame_dt = float(env["frame_dt"])
    seg_p92 = float(det["seg_p92"])
    lo_thr = float(det["lo_thr"])

    multi_isl = len(islands) > 1
    frame_islands: list[int] = []
    frame_mass: list[float] = []
    for isl_idx, (ia, ib) in enumerate(islands):
        length = max(1, ib - ia)
        if multi_isl:
            usable = max(float(min(length, 8)), float(length) - min(float(length) * 0.28, 24.0))
            scale = usable / float(length)
        else:
            scale = 1.0
        for fk in range(ia, ib):
            prominence = max(0.35, min(1.45, float(smooth_rms[fk]) / seg_p92)) ** 0.35
            frame_islands.append(isl_idx)
            frame_mass.append(scale * prominence)

    cum_mass = np.cumsum(frame_mass, dtype=np.float64)
    total_mass = float(cum_mass[-1]) if len(cum_mass) > 0 else 1.0
    total_w = sum(word_weights) or 1.0

    grouped: dict[int, list[tuple[str, float, float]]] = {j: [] for j in range(len(islands))}
    acc_w = 0.0
    for tok, ww, pr in zip(tokens, word_weights, token_probs):
        mid_frac = (acc_w + 0.5 * ww) / total_w
        acc_w += ww
        target_mass = mid_frac * total_mass
        pos = min(len(cum_mass) - 1, int(np.searchsorted(cum_mass, target_mass)))
        grouped[frame_islands[pos]].append((tok, ww, pr))

    aligned: list[dict[str, Any]] = []
    for isl_idx, (ia, ib) in enumerate(islands):
        grp = grouped[isl_idx]
        if not grp:
            continue
        t_isl_start = max(s0, float(t_center[ia]) - 0.5 * frame_dt)
        t_isl_end = min(s1, float(t_center[ib - 1]) + 0.5 * frame_dt)
        if t_isl_end <= t_isl_start + 0.04:
            t_isl_end = min(s1, t_isl_start + 0.08)

        local_rms = np.maximum(lo_thr * 0.5, smooth_rms[ia:ib])
        local_power = (local_rms / max(float(np.max(local_rms)), 1e-6)) ** 0.30
        local_cum = np.concatenate(([0.0], np.cumsum(local_power, dtype=np.float64)))
        local_total = float(local_cum[-1]) or 1.0
        isl_frame_times = np.linspace(t_isl_start, t_isl_end, (ib - ia) + 1)

        grp_total_w = sum(w for _, w, _ in grp) or 1.0
        cur_w_acc = 0.0
        prev_t = t_isl_start
        for idx_in_grp, (tok, ww, pr) in enumerate(grp):
            cur_w_acc += ww
            if idx_in_grp == len(grp) - 1:
                w_end = t_isl_end
            else:
                target_local = (cur_w_acc / grp_total_w) * local_total
                w_end = float(np.interp(target_local, local_cum, isl_frame_times))
            w_start = max(prev_t, t_isl_start)
            w_end = max(w_start + 0.04, min(t_isl_end, w_end))
            aligned.append(
                {
                    "word": tok,
                    "start_sec": round(w_start, 3),
                    "end_sec": round(w_end, 3),
                    "probability": round(float(pr), 3),
                }
            )
            prev_t = w_end

    return aligned, det


def align_transcript_to_audio(
    transcript: TranscriptResult,
    media_or_wav: Path,
    segment_ids: set[str] | None = None,
    force_all: bool = False,
    min_pause_sec: float = 0.10,
) -> TranscriptResult:
    """Align segment word timestamps onto real acoustic speech islands (dual-hysteresis log-RMS).

    Eliminates linear drift across pauses when applying human corrections or when ASR emits
    silence-padded word timings. Leaves text untouched.
    """
    env = compute_audio_envelope(media_or_wav)
    if env is None:
        return transcript

    t_center: np.ndarray = env["t_center"]
    floor = float(env["floor"])
    max_rms = float(env["max_rms"])

    for seg in transcript.segments:
        if segment_ids is not None and seg.segment_id not in segment_ids:
            continue
        tokens = [tok for tok in seg.text.split() if tok.strip()]
        if not tokens:
            continue

        s0 = max(0.0, float(seg.start_sec))
        s1 = max(s0 + 0.12, float(seg.end_sec))
        word_weights = [_syllable_word_weight(tok) for tok in tokens]
        prob_val = 1.0 if seg.verification_status == "human_confirmed" else max(0.85, float(seg.confidence))

        det = detect_speech_islands_in_window(env, s0, s1, min_pause_sec=min_pause_sec)
        if (
            seg.verification_status != "human_confirmed"
            and det["peak_rms"] < max(floor * 2.1, max_rms * 0.045)
            and len(tokens) <= 4
        ):
            if "low_speech_energy_hallucination" not in seg.anomaly_flags:
                seg.anomaly_flags.append("low_speech_energy_hallucination")

        islands: list[tuple[int, int]] = det["islands_idx"]
        active_set_mask = det["mask"]
        k0 = int(det["k0"])
        k1 = int(det["k1"])

        # When force_all is False, only overwrite ASR timings if alignment visibly drifts into silence
        if not force_all and len(seg.words) == len(tokens) and islands and active_set_mask is not None and (k1 - k0 >= 3):
            bad_words = 0
            for w in seg.words:
                wm_t = 0.5 * (float(w.start_sec) + float(w.end_sec))
                wm_k = min(k1 - 1, max(k0, int(np.searchsorted(t_center, wm_t))))
                wk0 = max(k0, int(np.searchsorted(t_center, float(w.start_sec))))
                wk1 = min(k1, max(wk0 + 1, int(np.searchsorted(t_center, float(w.end_sec)))))
                silent_frac = 1.0 - float(np.mean(active_set_mask[wk0 - k0 : wk1 - k0]))
                if (not active_set_mask[wm_k - k0]) or ((w.end_sec - w.start_sec) > 0.35 and silent_frac > 0.60):
                    bad_words += 1
            if bad_words == 0:
                continue

        aligned_dicts, _ = align_words_to_envelope(
            tokens=tokens,
            word_weights=word_weights,
            env=env,
            start_sec=s0,
            end_sec=s1,
            min_pause_sec=min_pause_sec,
            probs=[prob_val] * len(tokens),
        )
        if aligned_dicts:
            seg.words = [
                WordTimestamp(
                    word=d["word"],
                    start_sec=d["start_sec"],
                    end_sec=d["end_sec"],
                    probability=d["probability"],
                )
                for d in aligned_dicts
            ]

    return audit_and_verify_transcript(transcript)


ChunkRunnerFn = Callable[
    [Path, str | None, float, str],
    tuple[str, float, list[TranscriptSegment], list[str]],
]

_ANOMALY_RECHECK_FLAGS = {
    "single_token_repeat_loop",
    "sparse_words_over_long_window",
    "possible_code_switch_phonetic_garble",
    "mixed_cyrillic_latin_token",
    "hallucination_repeat_or_phantom_cleaned",
}


def _chunk_has_signal(wav_path: Path) -> bool:
    try:
        _sr_c, _pcm_c = _load_wav_mono_float(wav_path)
        return len(_pcm_c) > 0 and float(np.max(np.abs(_pcm_c))) >= 1e-4
    except Exception:
        return True


def _verify_and_select_chunk_hypothesis(
    chunk_wav: Path,
    c0: float,
    lang: str,
    prob: float,
    segs: list[TranscriptSegment],
    probe_langs: list[str],
    core_langs: set[str],
    multi_chunk: bool,
    tag_prefix: str,
    run_chunk: ChunkRunnerFn,
) -> tuple[str, float, list[TranscriptSegment]]:
    """Evaluate a decoded chunk and re-decode across candidate languages if anomalous or low-confidence."""
    base_score = _chunk_quality_score(lang, segs)
    chunk_anomalous = any(bool(set(_detect_segment_anomalies(s)) & _ANOMALY_RECHECK_FLAGS) for s in segs)
    chunk_low_conf = any(s.low_confidence or s.confidence < 0.56 for s in segs) or chunk_anomalous or not segs
    outlier_lang = (
        (lang in _OUTLIER_LANG_CODES and (not multi_chunk or lang not in core_langs))
    )
    chosen_lang, chosen_prob, chosen_segs = lang, prob, segs

    if (outlier_lang or chunk_low_conf) and probe_langs:
        alts: list[tuple[float, str, float, list[TranscriptSegment]]] = []
        for alt_lang in probe_langs:
            if alt_lang == lang and segs:
                continue
            try:
                a_lang, a_prob, a_segs, _ = run_chunk(chunk_wav, alt_lang, c0, f"{tag_prefix}_{alt_lang}")
                alts.append((_chunk_quality_score(a_lang, a_segs), a_lang, a_prob, a_segs))
            except Exception:
                continue
        if alts:
            best_alt_score, best_alt_lang, best_alt_prob, best_alt_segs = max(alts, key=lambda x: x[0])
            alt_summary = [
                {
                    "language": al,
                    "text": " ".join(s.text for s in asegs),
                    "quality_score": round(asc, 3),
                }
                for asc, al, _, asegs in alts
                if asegs
            ]
            if best_alt_score > base_score + 0.06:
                if segs:
                    alt_summary.append(
                        {
                            "language": lang,
                            "text": " ".join(s.text for s in segs),
                            "quality_score": round(base_score, 3),
                        }
                    )
                chosen_lang = best_alt_lang
                chosen_prob = best_alt_prob
                chosen_segs = best_alt_segs
                for s in chosen_segs:
                    s.verification_status = "alternate_hypothesis_selected"
                    s.candidate_alternatives = alt_summary[:3]
            else:
                for s in chosen_segs:
                    s.candidate_alternatives = alt_summary[:3]

    return chosen_lang, chosen_prob, chosen_segs


def _run_multilingual_chunked_verification(
    wav_path: Path,
    duration_sec: float,
    engine_name: str,
    model_label: str,
    language: str | None,
    run_chunk: ChunkRunnerFn,
) -> TranscriptResult:
    """Execute pause-aware chunking + multilingual kk/ru/en verification identically across ASR backends."""
    chunks = _plan_wav_chunks(wav_path) if (language is None and duration_sec > 24.0) else []
    if len(chunks) <= 1:
        lang, lang_prob, segments, uncertain_ids = run_chunk(wav_path, language, 0.0, "main")
        if language is None and _chunk_has_signal(wav_path):
            lang, lang_prob, segments = _verify_and_select_chunk_hypothesis(
                chunk_wav=wav_path,
                c0=0.0,
                lang=lang,
                prob=lang_prob,
                segs=segments,
                probe_langs=["kk", "ru", "en"],
                core_langs={lang} if lang not in _OUTLIER_LANG_CODES and lang != "unknown" else set(),
                multi_chunk=False,
                tag_prefix="main",
                run_chunk=run_chunk,
            )
            uncertain_ids = []
            for idx_s, s in enumerate(segments):
                s.segment_id = f"seg_{idx_s:04d}"
                if s.low_confidence:
                    uncertain_ids.append(s.segment_id)
        res = TranscriptResult(
            engine=engine_name,
            model=model_label,
            detected_language=lang,
            language_probability=lang_prob,
            languages_observed=[lang] if lang != "unknown" else [],
            duration_sec=duration_sec,
            segments=segments,
            uncertain_segment_ids=uncertain_ids,
            verbatim_preserved=True,
        )
        return audit_and_verify_transcript(res)

    raw_chunks: list[tuple[int, float, float, Path, str, float, list[TranscriptSegment], list[str]]] = []
    lang_counts: dict[str, int] = {}
    for idx, (c0, c1) in enumerate(chunks):
        chunk_wav = wav_path.parent / f"asr_chunk_{idx:03d}.wav"
        _write_wav_slice(wav_path, chunk_wav, c0, c1)
        if not _chunk_has_signal(chunk_wav):
            continue
        lang, prob, segs, unc = run_chunk(chunk_wav, None, c0, f"{idx:03d}")
        if lang != "unknown" and segs:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        raw_chunks.append((idx, c0, c1, chunk_wav, lang, prob, segs, unc))

    core_langs = {k for k, cnt in lang_counts.items() if cnt >= max(2, len(chunks) // 5)}
    if not core_langs and lang_counts:
        core_langs = {max(lang_counts, key=lambda k: lang_counts[k])}

    if core_langs & {"kk", "ru"} or any(l in _OUTLIER_LANG_CODES for _, _, _, _, l, _, _, _ in raw_chunks):
        probe_langs = ["kk", "ru", "en"]
    else:
        probe_langs = [lg for lg in ("kk", "ru", "en") if lg in core_langs or len(core_langs) > 0]

    all_segments: list[TranscriptSegment] = []
    all_uncertain: list[str] = []
    lang_weights: dict[str, int] = {}
    weighted_probs: list[tuple[float, int]] = []

    for idx, c0, c1, chunk_wav, lang, prob, segs, _unc in raw_chunks:
        chosen_lang, chosen_prob, chosen_segs = _verify_and_select_chunk_hypothesis(
            chunk_wav=chunk_wav,
            c0=c0,
            lang=lang,
            prob=prob,
            segs=segs,
            probe_langs=probe_langs,
            core_langs=core_langs,
            multi_chunk=len(chunks) >= 3,
            tag_prefix=f"{idx:03d}",
            run_chunk=run_chunk,
        )
        for s in chosen_segs:
            s.segment_id = f"seg_{len(all_segments):04d}"
            if chosen_lang != "unknown":
                s.language = chosen_lang
            if s.low_confidence:
                all_uncertain.append(s.segment_id)
            all_segments.append(s)

        chunk_chars = sum(len(s.text) for s in chosen_segs) or 1
        if chosen_lang != "unknown" and chosen_segs:
            lang_weights[chosen_lang] = lang_weights.get(chosen_lang, 0) + chunk_chars
        weighted_probs.append((chosen_prob, chunk_chars))

    ordered_langs = [k for k, _ in sorted(lang_weights.items(), key=lambda kv: kv[1], reverse=True)]
    dominant_lang = ordered_langs[0] if ordered_langs else "unknown"
    total_chars = sum(w for _, w in weighted_probs) or 1
    avg_prob = round(sum(p * w for p, w in weighted_probs) / total_chars, 3)

    res = TranscriptResult(
        engine=engine_name,
        model=model_label,
        detected_language=dominant_lang,
        language_probability=avg_prob,
        languages_observed=ordered_langs,
        duration_sec=duration_sec,
        segments=all_segments,
        uncertain_segment_ids=all_uncertain,
        verbatim_preserved=True,
    )
    return audit_and_verify_transcript(res)


def _transcribe_with_whisper_cpp(
    wav_path: Path,
    duration_sec: float,
    whisper_cli: str,
    model_path: Path,
    language: str | None = None,
) -> TranscriptResult:
    def _run_chunk(chunk_wav: Path, lang: str | None, offset_sec: float, tag: str) -> tuple[str, float, list[TranscriptSegment], list[str]]:
        return _run_whisper_cpp_chunk(
            chunk_wav=chunk_wav,
            out_prefix=wav_path.parent / f"wcpp_out_{tag}",
            whisper_cli=whisper_cli,
            model_path=model_path,
            language=lang,
            offset_sec=offset_sec,
            start_seg_idx=0,
        )

    return _run_multilingual_chunked_verification(
        wav_path=wav_path,
        duration_sec=duration_sec,
        engine_name="whisper.cpp",
        model_label=model_path.name,
        language=language,
        run_chunk=_run_chunk,
    )


def _transcribe_with_mlx_whisper(
    wav_path: Path,
    duration_sec: float,
    model_name: str,
    language: str | None = None,
    mlx_module: Any | None = None,
) -> TranscriptResult:
    if mlx_module is None:
        import mlx_whisper as mlx_module  # type: ignore

    effective_mlx_model = model_name
    hf_large_v3_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--mlx-community--whisper-large-v3-mlx"
    if model_name == "mlx-community/whisper-small-mlx" and hf_large_v3_dir.exists():
        effective_mlx_model = "mlx-community/whisper-large-v3-mlx"

    def _run_chunk(chunk_wav: Path, lang: str | None, offset_sec: float, _tag: str) -> tuple[str, float, list[TranscriptSegment], list[str]]:
        kwargs: dict[str, Any] = {
            "path_or_hf_repo": effective_mlx_model,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.2,
            "no_speech_threshold": 0.55,
        }
        if lang:
            kwargs["language"] = lang
        result = mlx_module.transcribe(str(chunk_wav), **kwargs)
        detected_lang = str(result.get("language") or lang or "unknown")
        lang_prob = float(result.get("language_probability", 0.95 if detected_lang != "unknown" else 0.0))
        raw_segs = result.get("segments") or []
        segments, uncertain_ids = _convert_whisper_segments(raw_segs, detected_lang)
        _shift_segments_offset(segments, offset_sec)
        return detected_lang, round(lang_prob, 3), segments, uncertain_ids

    return _run_multilingual_chunked_verification(
        wav_path=wav_path,
        duration_sec=duration_sec,
        engine_name="mlx-whisper",
        model_label=effective_mlx_model,
        language=language,
        run_chunk=_run_chunk,
    )


def _transcribe_with_faster_whisper(
    wav_path: Path,
    duration_sec: float,
    model_name: str,
    language: str | None = None,
    whisper_model: Any | None = None,
    fw_size: str | None = None,
) -> TranscriptResult:
    resolved_size = fw_size or ("small" if "small" in model_name else "base")
    if whisper_model is None:
        from faster_whisper import WhisperModel  # type: ignore

        whisper_model = WhisperModel(resolved_size, device="cpu", compute_type="int8")

    def _run_chunk(chunk_wav: Path, lang: str | None, offset_sec: float, _tag: str) -> tuple[str, float, list[TranscriptSegment], list[str]]:
        seg_iter, info = whisper_model.transcribe(str(chunk_wav), language=lang, word_timestamps=True)
        raw_segs: list[dict[str, Any]] = []
        for s in seg_iter:
            raw_segs.append(
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                        for w in (s.words or [])
                    ],
                }
            )
        detected_lang = str(getattr(info, "language", None) or lang or "unknown")
        lang_prob = float(getattr(info, "language_probability", 0.9 if lang else 0.0))
        segments, uncertain_ids = _convert_whisper_segments(raw_segs, detected_lang)
        _shift_segments_offset(segments, offset_sec)
        return detected_lang, round(lang_prob, 3), segments, uncertain_ids

    return _run_multilingual_chunked_verification(
        wav_path=wav_path,
        duration_sec=duration_sec,
        engine_name="faster-whisper",
        model_label=resolved_size,
        language=language,
        run_chunk=_run_chunk,
    )


def transcribe_media(
    media_path: Path,
    duration_sec: float,
    model_name: str = "mlx-community/whisper-small-mlx",
    language: str | None = None,
) -> TranscriptResult:
    """Transcribe speech locally using mlx-whisper, whisper.cpp (Metal), or faster-whisper.

    Never fabricates missing speech; uncertain segments are explicitly flagged.
    """
    with tempfile.TemporaryDirectory(prefix="symposium_asr_") as tmpdir:
        wav_path = extract_mono_wav(media_path, Path(tmpdir) / "audio_16k.wav")
        raw_wav_path = extract_raw_mono_wav(media_path, Path(tmpdir) / "audio_raw_16k.wav")

        # 1. Try mlx-whisper (Apple Silicon Python MLX acceleration)
        try:
            tr = _transcribe_with_mlx_whisper(
                wav_path=wav_path,
                duration_sec=duration_sec,
                model_name=model_name,
                language=language,
            )
            return align_transcript_to_audio(tr, raw_wav_path, force_all=False)
        except ImportError:
            pass

        # 2. Try whisper.cpp (`whisper-cli` native Metal GPU binary)
        whisper_cli = shutil.which("whisper-cli")
        ggml_model = find_whisper_cpp_model(model_name)
        if whisper_cli and ggml_model:
            tr = _transcribe_with_whisper_cpp(
                wav_path=wav_path,
                duration_sec=duration_sec,
                whisper_cli=whisper_cli,
                model_path=ggml_model,
                language=language,
            )
            return align_transcript_to_audio(tr, raw_wav_path, force_all=False)

        # 3. Try faster-whisper (cross-platform CTranslate2 fallback)
        try:
            tr = _transcribe_with_faster_whisper(
                wav_path=wav_path,
                duration_sec=duration_sec,
                model_name=model_name,
                language=language,
            )
            return align_transcript_to_audio(tr, raw_wav_path, force_all=False)
        except ImportError as exc:
            raise RuntimeError(
                "No local Whisper backend available (install mlx-whisper, whisper-cpp + GGML model, or faster-whisper)."
            ) from exc


def extract_final_audio_word_timestamps(
    final_audio_wav: Path,
    languages: list[str] | None = None,
    model_hint: str = "mlx-community/whisper-small-mlx",
) -> list[dict[str, Any]]:
    """Decode actual spoken word occurrences & timestamps from final edited audio via available local ASR."""
    wav = Path(final_audio_wav)
    if not wav.exists():
        return []
    env = compute_audio_envelope(wav)
    if env is None or float(env["max_rms"]) < 5e-4:
        return []
    sr, samples = _load_wav_mono_float(wav)
    if sr <= 0 or len(samples) < int(0.1 * sr):
        return []
    duration_sec = float(len(samples)) / float(sr)

    clean_langs = [lg for lg in (languages or []) if lg and lg != "unknown"]
    single_lang = clean_langs[0] if len(set(clean_langs)) == 1 else None

    tr: TranscriptResult | None = None
    with tempfile.TemporaryDirectory(prefix="symposium_final_align_") as tmpdir:
        work_wav = Path(tmpdir) / "final_16k.wav"
        shutil.copy2(wav, work_wav)
        try:
            tr = _transcribe_with_mlx_whisper(work_wav, duration_sec, model_hint, language=single_lang)
        except Exception:
            tr = None
        if tr is None:
            whisper_cli = shutil.which("whisper-cli")
            ggml_model = find_whisper_cpp_model(model_hint)
            if whisper_cli and ggml_model:
                try:
                    tr = _transcribe_with_whisper_cpp(work_wav, duration_sec, whisper_cli, ggml_model, language=single_lang)
                except Exception:
                    tr = None
        if tr is None:
            try:
                tr = _transcribe_with_faster_whisper(work_wav, duration_sec, model_hint, language=single_lang)
            except Exception:
                tr = None

    if tr is None:
        return []
    words_out: list[dict[str, Any]] = []
    for seg in tr.segments:
        for w in seg.words:
            txt = w.word.strip()
            if not txt:
                continue
            words_out.append(
                {
                    "word": txt,
                    "start_sec": round(float(w.start_sec), 3),
                    "end_sec": round(max(float(w.start_sec) + 0.04, float(w.end_sec)), 3),
                    "probability": round(float(w.probability), 3),
                    "language": seg.language,
                }
            )
    words_out.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    return words_out
