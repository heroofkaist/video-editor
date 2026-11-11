"""Multilingual local transcription via mlx-whisper, whisper.cpp (whisper-cli Metal), or faster-whisper."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from symposium_studio.models.media import TranscriptResult, TranscriptSegment, WordTimestamp


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


def _convert_whisper_segments(
    raw_segments: list[dict[str, Any]],
    detected_language: str,
) -> tuple[list[TranscriptSegment], list[str]]:
    segments: list[TranscriptSegment] = []
    uncertain_ids: list[str] = []

    for idx, seg in enumerate(raw_segments):
        seg_id = f"seg_{idx:04d}"
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start_sec = round(float(seg.get("start", 0.0)), 3)
        end_sec = round(max(start_sec + 0.05, float(seg.get("end", start_sec + 0.5))), 3)

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
    preferred_names = (
        ["ggml-small.bin", "ggml-medium.bin", "ggml-base.bin", "ggml-tiny.bin"]
        if "small" in preferred_hint
        else ["ggml-base.bin", "ggml-small.bin", "ggml-tiny.bin"]
    )
    for d in search_dirs:
        if not d.exists():
            continue
        for fname in preferred_names:
            cand = d / fname
            if cand.is_file():
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
        raw = str(tok.get("text", ""))
        stripped = raw.strip()
        if not stripped or (stripped.startswith("[") and stripped.endswith("]")):
            continue
        offs = tok.get("offsets") or {}
        t0 = float(offs.get("from", int(seg_start * 1000))) / 1000.0
        t1 = float(offs.get("to", int(t0 * 1000))) / 1000.0
        p = float(tok.get("p", 0.75))
        probs_all.append(p)

        # New word boundary if leading whitespace or purely opening punctuation when buffer non-empty
        if raw.startswith(" ") and cur_text:
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


def _transcribe_with_whisper_cpp(
    wav_path: Path,
    duration_sec: float,
    whisper_cli: str,
    model_path: Path,
    language: str | None = None,
) -> TranscriptResult:
    out_prefix = wav_path.parent / "wcpp_out"
    cmd = [
        whisper_cli,
        "-m", str(model_path),
        "-f", str(wav_path),
        "-l", language if language else "auto",
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
    for idx, item in enumerate(payload.get("transcription") or []):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        offs = item.get("offsets") or {}
        s0 = round(float(offs.get("from", 0)) / 1000.0, 3)
        s1 = round(max(s0 + 0.05, float(offs.get("to", int((s0 + 0.5) * 1000))) / 1000.0), 3)
        words, seg_conf = _tokens_to_words(item.get("tokens") or [], s0, s1)
        low_conf = seg_conf < 0.45
        seg_id = f"seg_{idx:04d}"
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
                low_confidence_reason=f"low_token_confidence={seg_conf:.2f}" if low_conf else None,
                words=words,
            )
        )

    return TranscriptResult(
        engine="whisper.cpp",
        model=model_path.name,
        detected_language=detected_lang,
        language_probability=lang_prob,
        languages_observed=[detected_lang] if detected_lang != "unknown" else [],
        duration_sec=duration_sec,
        segments=segments,
        uncertain_segment_ids=uncertain_ids,
        verbatim_preserved=True,
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

        # 1. Try mlx-whisper (Apple Silicon Python MLX acceleration)
        try:
            import mlx_whisper  # type: ignore

            kwargs: dict[str, Any] = {
                "path_or_hf_repo": model_name,
                "word_timestamps": True,
                "condition_on_previous_text": True,
            }
            if language:
                kwargs["language"] = language
            result = mlx_whisper.transcribe(str(wav_path), **kwargs)
            detected_lang = str(result.get("language") or language or "unknown")
            raw_segs = result.get("segments") or []
            segments, uncertain_ids = _convert_whisper_segments(raw_segs, detected_lang)
            langs = sorted({s.language for s in segments} | ({detected_lang} if detected_lang != "unknown" else set()))
            return TranscriptResult(
                engine="mlx-whisper",
                model=model_name,
                detected_language=detected_lang,
                language_probability=0.95 if detected_lang != "unknown" else 0.0,
                languages_observed=langs,
                duration_sec=duration_sec,
                segments=segments,
                uncertain_segment_ids=uncertain_ids,
                verbatim_preserved=True,
            )
        except ImportError:
            pass

        # 2. Try whisper.cpp (`whisper-cli` native Metal GPU binary)
        whisper_cli = shutil.which("whisper-cli")
        ggml_model = find_whisper_cpp_model(model_name)
        if whisper_cli and ggml_model:
            return _transcribe_with_whisper_cpp(
                wav_path=wav_path,
                duration_sec=duration_sec,
                whisper_cli=whisper_cli,
                model_path=ggml_model,
                language=language,
            )

        # 3. Try faster-whisper (cross-platform CTranslate2 fallback)
        try:
            from faster_whisper import WhisperModel  # type: ignore

            fw_size = "small" if "small" in model_name else "base"
            model = WhisperModel(fw_size, device="cpu", compute_type="int8")
            seg_iter, info = model.transcribe(str(wav_path), language=language, word_timestamps=True)
            raw_segs = []
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
            detected_lang = str(getattr(info, "language", None) or language or "unknown")
            lang_prob = float(getattr(info, "language_probability", 0.0))
            segments, uncertain_ids = _convert_whisper_segments(raw_segs, detected_lang)
            return TranscriptResult(
                engine="faster-whisper",
                model=fw_size,
                detected_language=detected_lang,
                language_probability=round(lang_prob, 3),
                languages_observed=[detected_lang] if detected_lang != "unknown" else [],
                duration_sec=duration_sec,
                segments=segments,
                uncertain_segment_ids=uncertain_ids,
                verbatim_preserved=True,
            )
        except ImportError as exc:
            raise RuntimeError(
                "No local Whisper backend available (install mlx-whisper, whisper-cpp + GGML model, or faster-whisper)."
            ) from exc
