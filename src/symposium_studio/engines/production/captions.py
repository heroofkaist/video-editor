"""SRT and ASS kinetic/safe-zone subtitle generation from RenderTimeline."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from symposium_studio.engines.analysis.transcribe import (
    _syllable_word_weight,
    align_words_to_envelope,
    compute_audio_envelope,
    detect_speech_islands_in_window,
    extract_final_audio_word_timestamps,
)
from symposium_studio.models.timeline import CaptionCue, RenderTimeline, TargetPlatformFormat, VideoClipNode


def _fmt_srt_time(sec: float) -> str:
    ms_total = max(0, int(round(sec * 1000)))
    h = ms_total // 3_600_000
    m = (ms_total % 3_600_000) // 60_000
    s = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ass_time(sec: float) -> str:
    cs_total = max(0, int(round(sec * 100)))
    h = cs_total // 360_000
    m = (cs_total % 360_000) // 6000
    s = (cs_total % 6000) // 100
    cs = cs_total % 100
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def write_srt(timeline: RenderTimeline, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for idx, cue in enumerate(timeline.captions, start=1):
        txt = cue.text.strip()
        if cue.low_confidence:
            txt = f"[uncertain] {txt}"
        lines.append(f"{idx}\n{_fmt_srt_time(cue.start_sec)} --> {_fmt_srt_time(cue.end_sec)}\n{txt}\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _wrap_caption_tokens(items: list[tuple[str, int]], max_words: int, max_chars_per_line: int) -> list[list[tuple[str, int]]]:
    """Break caption words into at most 2 readable lines obeying both word count and char width limits."""
    if not items:
        return []
    lines: list[list[tuple[str, int]]] = [[]]
    cur_chars = 0
    for word_txt, dur_cs in items:
        cur_line = lines[-1]
        next_chars = cur_chars + (1 if cur_line else 0) + len(word_txt)
        need_break = bool(cur_line) and (
            len(cur_line) >= max_words or next_chars > max_chars_per_line
        )
        if need_break and len(lines) < 2:
            lines.append([])
            cur_chars = 0
            cur_line = lines[-1]
            next_chars = len(word_txt)
        cur_line.append((word_txt, dur_cs))
        cur_chars = next_chars
    return lines


def _format_karaoke_line(
    cue: CaptionCue,
    max_words: int,
    highlight_word: bool,
    max_chars_per_line: int = 42,
) -> str:
    words = cue.words
    if not words or not highlight_word:
        parts = [p for p in cue.text.strip().split() if p]
        wrapped = _wrap_caption_tokens([(p, 20) for p in parts], max_words=max_words, max_chars_per_line=max_chars_per_line)
        if wrapped:
            return "\\N".join(_escape_ass(" ".join(w for w, _ in ln)) for ln in wrapped)
        return _escape_ass(cue.text.strip())

    first_ws = float(words[0].get("start_sec", 0.0))
    last_we = float(words[-1].get("end_sec", 0.0))
    is_relative = cue.start_sec > 0.35 and first_ws < 0.20 and last_we <= (cue.end_sec - cue.start_sec) + 0.20
    origin = (
        cue.start_sec
        if (not is_relative and first_ws >= cue.start_sec - 0.06 and last_we <= cue.end_sec + 0.20)
        else 0.0
    )

    raw_items: list[tuple[str, int]] = []
    word_tags: list[str] = []
    cursor_cs = 0
    for w in words:
        word_txt = _escape_ass(str(w.get("word", "")).strip())
        if not word_txt:
            continue
        w_start_cs = max(cursor_cs, int(round((float(w.get("start_sec", 0.0)) - origin) * 100)))
        w_end_cs = max(w_start_cs + 8, int(round((float(w.get("end_sec", 0.2)) - origin) * 100)))
        wait_cs = w_start_cs - cursor_cs
        dur_cs = max(8, w_end_cs - w_start_cs)
        cursor_cs = w_start_cs + dur_cs
        prefix = f"{{\\k{wait_cs}}}" if wait_cs >= 2 else ""
        raw_items.append((word_txt, dur_cs))
        word_tags.append(f"{prefix}{{\\kf{dur_cs}}}{word_txt}")

    wrapped = _wrap_caption_tokens(raw_items, max_words=max_words, max_chars_per_line=max_chars_per_line)
    if not wrapped:
        return _escape_ass(cue.text.strip())
    lines_out: list[str] = []
    tag_idx = 0
    for ln in wrapped:
        ln_tokens: list[str] = []
        for _wt, _dur in ln:
            ln_tokens.append(word_tags[tag_idx])
            tag_idx += 1
        lines_out.append(" ".join(ln_tokens))
    return "\\N".join(lines_out)


def _match_cue_clip(cue: CaptionCue, clips: list[VideoClipNode]) -> VideoClipNode | None:
    if cue.clip_id:
        for c in clips:
            if c.clip_id == cue.clip_id:
                return c
    mid = 0.5 * (float(cue.start_sec) + float(cue.end_sec))
    best_clip: VideoClipNode | None = None
    best_ov = 0.0
    for c in clips:
        if c.timeline_start_sec - 0.08 <= mid <= c.timeline_end_sec + 0.08:
            return c
        ov = max(0.0, min(c.timeline_end_sec, cue.end_sec) - max(c.timeline_start_sec, cue.start_sec))
        if ov > best_ov:
            best_ov = ov
            best_clip = c
    return best_clip


_CYR_TO_LAT_TABLE = str.maketrans({
    "а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d",
    "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i", "і": "i", "й": "y",
    "к": "k", "қ": "k", "л": "l", "м": "m", "н": "n", "ң": "n", "о": "o",
    "ө": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u",
    "ү": "u", "ф": "f", "х": "h", "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "sh", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})


def _norm_word_token(tok: str) -> str:
    return re.sub(r"[^\w]+", "", tok.casefold()).strip()


def _word_match_similarity(expected_tok: str, asr_tok: str) -> float:
    e_norm = _norm_word_token(expected_tok)
    a_norm = _norm_word_token(asr_tok)
    if not e_norm or not a_norm:
        return 0.0
    if e_norm == a_norm:
        return 1.0
    sim_direct = difflib.SequenceMatcher(None, e_norm, a_norm).ratio()
    e_lat = e_norm.translate(_CYR_TO_LAT_TABLE)
    a_lat = a_norm.translate(_CYR_TO_LAT_TABLE)
    sim_lat = difflib.SequenceMatcher(None, e_lat, a_lat).ratio()
    # Cross-script phonetic check (e.g. Latin "test" vs Cyrillic "тест")
    best = max(sim_direct, sim_lat * 0.96)
    if (e_norm.startswith(a_norm) or a_norm.startswith(e_norm)) and min(len(e_norm), len(a_norm)) >= 3:
        best = max(best, 0.80)
    return best


def _normalize_monotonic_cues_and_words(captions: list[CaptionCue]) -> None:
    """Enforce strictly non-overlapping cues and strictly monotonic internal word timings in place."""
    captions.sort(key=lambda c: (c.start_sec, c.end_sec))
    prev_cue_end = 0.0
    for cue in captions:
        cue_s = max(prev_cue_end, float(cue.start_sec))
        if cue.words:
            w_cursor = cue_s
            for w in cue.words:
                orig_ws = float(w.get("start_sec", w_cursor))
                orig_we = float(w.get("end_sec", orig_ws + 0.08))
                orig_dur = max(0.04, min(1.5, orig_we - orig_ws))
                ws = max(w_cursor, orig_ws)
                we = max(ws + 0.04, orig_we if orig_ws >= w_cursor else ws + min(0.18, orig_dur))
                w["start_sec"] = round(ws, 3)
                w["end_sec"] = round(we, 3)
                w_cursor = float(w["end_sec"])
            cue.start_sec = round(float(cue.words[0]["start_sec"]), 3)
            cue.end_sec = round(max(cue.start_sec + 0.14, float(cue.words[-1]["end_sec"])), 3)
        else:
            cue.start_sec = round(cue_s, 3)
            cue.end_sec = round(max(cue.start_sec + 0.14, float(cue.end_sec)), 3)
        prev_cue_end = float(cue.end_sec)


def _speech_silence_threshold(env: dict[str, Any]) -> float:
    floor = float(env["floor"])
    max_rms = float(env["max_rms"])
    has_silent_baseline = floor <= 0.025 and floor < max_rms * 0.20
    return max(floor * 1.85, max_rms * 0.03) if has_silent_baseline else min(max_rms * 0.03, floor * 0.35)


def _tighten_asr_word_span(
    w: dict[str, Any],
    env: dict[str, Any] | None,
    clip_start: float | None = None,
    clip_end: float | None = None,
    min_pause_sec: float = 0.10,
) -> tuple[float, float, bool]:
    """Tighten pause-smeared ASR word spans onto actual speech islands before clip clamping."""
    ws_raw = float(w.get("start_sec", 0.0))
    we_raw = max(ws_raw + 0.04, float(w.get("end_sec", ws_raw + 0.04)))
    has_real_speech = env is not None and float(env["max_rms"]) >= 5e-4
    if has_real_speech and env is not None:
        silence_thresh = _speech_silence_threshold(env)
        if we_raw - ws_raw > 1.20:
            det_tok = detect_speech_islands_in_window(env, ws_raw, we_raw, min_pause_sec=min_pause_sec)
            if det_tok["islands_sec"] and float(det_tok["peak_rms"]) >= silence_thresh:
                ws_raw, we_raw = max(det_tok["islands_sec"], key=lambda iv: iv[1] - iv[0])
            else:
                return ws_raw, we_raw, False
        else:
            det_tok = detect_speech_islands_in_window(env, ws_raw, we_raw, min_pause_sec=min_pause_sec)
            if float(det_tok["peak_rms"]) < silence_thresh:
                return ws_raw, we_raw, False

    if clip_start is not None:
        ws_raw = max(clip_start, ws_raw)
    if clip_end is not None:
        we_raw = min(clip_end, max(ws_raw + 0.04, we_raw))
    return round(ws_raw, 3), round(max(ws_raw + 0.04, we_raw), 3), True


def _align_tokens_to_asr_words(
    expected_items: list[dict[str, Any]],
    asr_words: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    env: dict[str, Any] | None,
    min_pause_sec: float = 0.10,
) -> tuple[list[dict[str, Any]] | None, int]:
    """Align expected timeline tokens monotonically against ASR-transcribed words from final audio."""
    if not expected_items or not asr_words:
        return None, 0

    clip_dur = max(0.2, clip_end - clip_start)
    cand_asr: list[dict[str, Any]] = []
    for w in asr_words:
        if float(w.get("end_sec", 0.0)) < clip_start - 0.35 or float(w.get("start_sec", 0.0)) > clip_end + 0.35:
            continue
        if not _norm_word_token(str(w.get("word", ""))):
            continue
        ws_t, we_t, ok_tok = _tighten_asr_word_span(w, env, clip_start=clip_start, clip_end=clip_end, min_pause_sec=min_pause_sec)
        if not ok_tok:
            continue
        cleaned = dict(w)
        cleaned["start_sec"] = ws_t
        cleaned["end_sec"] = we_t
        cand_asr.append(cleaned)

    n = len(expected_items)
    m = len(cand_asr)
    if n == 0 or m == 0:
        return None, 0

    max_time_slack = min(2.4, max(1.35, 0.25 * clip_dur))
    dp: list[list[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
    choice: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]  # 1=skip_i, 2=skip_j, 3=match

    for i in range(1, n + 1):
        ei = expected_items[i - 1]
        e_txt = str(ei["word"])
        e_norm = _norm_word_token(e_txt)
        e_mid = 0.5 * (float(ei.get("orig_start", clip_start)) + float(ei.get("orig_end", clip_end)))
        for j in range(1, m + 1):
            aj = cand_asr[j - 1]
            a_mid = 0.5 * (float(aj["start_sec"]) + float(aj["end_sec"]))
            best_sc = dp[i - 1][j]
            act = 1
            if dp[i][j - 1] > best_sc:
                best_sc = dp[i][j - 1]
                act = 2
            dt = abs(e_mid - a_mid)
            if dt <= max_time_slack:
                a_norm = _norm_word_token(str(aj["word"]))
                sim = _word_match_similarity(e_txt, str(aj["word"]))
                min_tok_len = min(len(e_norm), len(a_norm))
                prob_a = float(aj.get("probability", 0.9))
                # Prevent tiny 1-2 char function words or very low-prob tokens from anchoring large jumps
                short_jump = min_tok_len <= 2 and (dt > 1.05 or sim < 0.85 or prob_a < 0.25)
                if sim >= 0.54 and not short_jump:
                    match_sc = dp[i - 1][j - 1] + (sim * 2.0) - min(0.75, 0.22 * dt)
                    if match_sc > best_sc:
                        best_sc = match_sc
                        act = 3
            dp[i][j] = best_sc
            choice[i][j] = act

    matches: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        act = choice[i][j]
        if act == 3:
            matches[i - 1] = j - 1
            i -= 1
            j -= 1
        elif act == 1:
            i -= 1
        else:
            j -= 1

    # Prune contradictory/compressing anchors that squeeze >=2 intervening words below physical speech duration
    sorted_m = sorted(matches.items())
    if len(sorted_m) >= 2:
        pruned: dict[int, int] = {}
        kept_list: list[tuple[int, int]] = []
        for idx_e, idx_a in sorted_m:
            aj = cand_asr[idx_a]
            ws = max(clip_start, min(clip_end - 0.04, float(aj["start_sec"])))
            if kept_list:
                prev_e, prev_a = kept_list[-1]
                prev_we = min(clip_end, max(clip_start + 0.04, float(cand_asr[prev_a]["end_sec"])))
                k_between = idx_e - prev_e - 1
                if k_between >= 2:
                    avail = ws - prev_we
                    orig_gap_span = float(expected_items[idx_e - 1]["orig_end"]) - float(expected_items[prev_e + 1]["orig_start"])
                    if avail < 0.065 * k_between and orig_gap_span >= 0.22 * k_between:
                        # Drop whichever of the two anchors deviates further from its expected timeline time
                        prev_dt = abs(0.5 * (float(expected_items[prev_e]["orig_start"]) + float(expected_items[prev_e]["orig_end"])) - float(cand_asr[prev_a]["start_sec"]))
                        curr_dt = abs(0.5 * (float(expected_items[idx_e]["orig_start"]) + float(expected_items[idx_e]["orig_end"])) - ws)
                        if curr_dt >= prev_dt:
                            continue
                        kept_list.pop()
            kept_list.append((idx_e, idx_a))
        pruned = dict(kept_list)
        matches = pruned

    matched_cnt = len(matches)
    if matched_cnt == 0 or (n >= 4 and matched_cnt < 2 and matched_cnt / n < 0.25):
        return None, 0

    aligned: list[dict[str, Any] | None] = [None] * n
    for idx_e, idx_a in sorted(matches.items()):
        aj = cand_asr[idx_a]
        ws = max(clip_start, min(clip_end - 0.04, float(aj["start_sec"])))
        we = min(clip_end, max(ws + 0.04, float(aj["end_sec"])))
        orig = expected_items[idx_e]
        aligned[idx_e] = {
            "word": str(orig["word"]),
            "start_sec": round(ws, 3),
            "end_sec": round(we, 3),
            "probability": round(float(orig.get("probability", aj.get("probability", 0.95))), 3),
            "alignment_source": "asr_word_match",
            "matched_asr_word": str(aj["word"]),
        }

    # Interpolate any unmatched words inside tight neighboring matched ASR anchor bounds + acoustic islands,
    # preserving natural pause boundaries from expected orig_start/orig_end so pauses never stretch single cues.
    idx = 0
    while idx < n:
        if aligned[idx] is not None:
            idx += 1
            continue
        run_end = idx
        while run_end < n and aligned[run_end] is None:
            run_end += 1
        left_t = float(aligned[idx - 1]["end_sec"]) if idx > 0 and aligned[idx - 1] is not None else max(
            clip_start, float(expected_items[idx].get("orig_start", clip_start)) - 0.20
        )
        right_t = float(aligned[run_end]["start_sec"]) if run_end < n and aligned[run_end] is not None else min(
            clip_end, float(expected_items[run_end - 1].get("orig_end", clip_end)) + 0.20
        )
        left_t = max(clip_start, left_t)
        right_t = min(clip_end, max(left_t + 0.04 * (run_end - idx), right_t))

        # Split unmatched run into natural speaking bursts across >=0.40s pauses in expected timeline timing
        bursts: list[tuple[int, int]] = []
        b_start = idx
        for k in range(idx + 1, run_end):
            pause_gap = float(expected_items[k]["orig_start"]) - float(expected_items[k - 1]["orig_end"])
            if pause_gap >= 0.40:
                bursts.append((b_start, k))
                b_start = k
        bursts.append((b_start, run_end))

        cursor_t = left_t
        for b_i, (bs, be) in enumerate(bursts):
            k_rem_after = run_end - be
            hard_max_t = right_t - 0.04 * k_rem_after
            first_orig_s = float(expected_items[bs].get("orig_start", cursor_t))
            last_orig_e = float(expected_items[be - 1].get("orig_end", hard_max_t))
            # If this burst is just one segment among multiple bursts or bounded by a distant anchor,
            # localize the acoustic window instead of stretching across the long silent pause.
            if len(bursts) > 1:
                win_s = max(cursor_t, first_orig_s - 0.55)
                next_burst_orig_s = float(expected_items[be]["orig_start"]) if b_i + 1 < len(bursts) else hard_max_t
                win_e = min(hard_max_t, max(win_s + 0.04 * (be - bs), min(last_orig_e + 0.55, 0.5 * (last_orig_e + next_burst_orig_s))))
            else:
                win_s = cursor_t
                # Also guard a single burst before a distant anchor after a long pause
                if run_end < n and (right_t - last_orig_e) > 1.25:
                    win_e = min(hard_max_t, max(win_s + 0.04 * (be - bs), last_orig_e + 0.55))
                else:
                    win_e = hard_max_t
            win_s = max(clip_start, min(win_s, hard_max_t - 0.04 * (be - bs)))
            win_e = min(clip_end, max(win_s + 0.04 * (be - bs), win_e))

            sub_tokens = [str(expected_items[k]["word"]) for k in range(bs, be)]
            sub_weights = [_syllable_word_weight(t) for t in sub_tokens]
            sub_probs = [float(expected_items[k].get("probability", 0.92)) for k in range(bs, be)]

            used_env = False
            if env is not None:
                det = detect_speech_islands_in_window(env, win_s, win_e, min_pause_sec=min_pause_sec)
                if det["islands_idx"]:
                    env_words, _ = align_words_to_envelope(
                        tokens=sub_tokens,
                        word_weights=sub_weights,
                        env=env,
                        start_sec=win_s,
                        end_sec=win_e,
                        min_pause_sec=min_pause_sec,
                        probs=sub_probs,
                    )
                    if len(env_words) == len(sub_tokens):
                        for k_off, ew in enumerate(env_words):
                            aligned[bs + k_off] = {
                                "word": sub_tokens[k_off],
                                "start_sec": round(float(ew["start_sec"]), 3),
                                "end_sec": round(float(ew["end_sec"]), 3),
                                "probability": round(float(ew["probability"]), 3),
                                "alignment_source": "asr_anchor_interpolated",
                            }
                        used_env = True
            if not used_env:
                tot_w = sum(sub_weights) or float(len(sub_tokens))
                cur_t = win_s
                span = max(0.04 * len(sub_tokens), win_e - win_s)
                for k_off, (tok, ww, pr) in enumerate(zip(sub_tokens, sub_weights, sub_probs)):
                    dur = span * (ww / tot_w)
                    we = min(clip_end, max(cur_t + 0.04, cur_t + dur))
                    aligned[bs + k_off] = {
                        "word": tok,
                        "start_sec": round(cur_t, 3),
                        "end_sec": round(we, 3),
                        "probability": round(pr, 3),
                        "alignment_source": "asr_anchor_interpolated",
                    }
                    cur_t = we
            cursor_t = float(aligned[be - 1]["end_sec"])  # type: ignore[index]
        idx = run_end

    # Guarantee strictly ascending word timestamps within [clip_start, clip_end]
    out: list[dict[str, Any]] = []
    prev_end = clip_start
    for rem_idx, item in enumerate(aligned):
        assert item is not None
        max_end_here = clip_end - 0.04 * (n - rem_idx - 1)
        ws = max(prev_end, min(max_end_here - 0.04, float(item["start_sec"])))
        we = min(max_end_here, max(ws + 0.04, float(item["end_sec"])))
        if we <= ws:
            ws = max(prev_end, max_end_here - 0.04)
            we = max_end_here
        item["start_sec"] = round(ws, 3)
        item["end_sec"] = round(we, 3)
        prev_end = item["end_sec"]
        out.append(item)
    return out, matched_cnt


def align_captions_to_final_audio(
    timeline: RenderTimeline,
    final_audio_wav: Path,
    min_pause_sec: float = 0.10,
    final_asr_words: list[dict[str, Any]] | None = None,
    use_asr_alignment: bool = True,
) -> RenderTimeline:
    """Align caption cue and word timestamps directly against final edited concatenated audio.

    Prefers genuine ASR word-timestamp DP matching against final edited audio pronunciation,
    refining unmatched inter-anchor words via acoustic speech islands, and gracefully falling back
    to acoustic envelope anchoring when final audio has no decodable speech tokens.
    """
    aligned_tl = timeline.model_copy(deep=True)
    if not aligned_tl.captions:
        return aligned_tl

    env = compute_audio_envelope(final_audio_wav)
    if env is None or float(env["max_rms"]) < 5e-4:
        return aligned_tl

    asr_words: list[dict[str, Any]] = list(final_asr_words or [])
    if not asr_words and use_asr_alignment:
        lang_hints = list(dict.fromkeys(
            m.language for m in aligned_tl.segment_mappings if m.language and m.language != "unknown"
        ))
        asr_words = extract_final_audio_word_timestamps(final_audio_wav, languages=lang_hints or None)

    clips = aligned_tl.clips
    clip_order = {c.clip_id: i for i, c in enumerate(clips)}

    # Group captions by host clip in chronological order
    grouped_indices: dict[int, list[int]] = {}
    unmatched_indices: list[int] = []
    for idx, cue in enumerate(aligned_tl.captions):
        host = _match_cue_clip(cue, clips)
        if host is not None and host.clip_id in clip_order:
            grouped_indices.setdefault(clip_order[host.clip_id], []).append(idx)
        else:
            unmatched_indices.append(idx)

    floor = float(env["floor"])
    max_rms = float(env["max_rms"])

    for c_idx in sorted(grouped_indices):
        clip = clips[c_idx]
        idx_list = sorted(grouped_indices[c_idx], key=lambda i: aligned_tl.captions[i].start_sec)

        # Stage 1: Try genuine ASR word-level monotone alignment across this clip's cues
        clip_cue_slices: list[tuple[int, list[dict[str, Any]]]] = []
        clip_flat_items: list[dict[str, Any]] = []
        for idx in idx_list:
            cue = aligned_tl.captions[idx]
            if cue.words:
                w_items = [dict(w) for w in cue.words if str(w.get("word", "")).strip()]
            else:
                toks = [t for t in cue.text.split() if t.strip()]
                step = max(0.04, (cue.end_sec - cue.start_sec) / max(1, len(toks)))
                w_items = [
                    {
                        "word": t,
                        "start_sec": round(cue.start_sec + wi * step, 3),
                        "end_sec": round(min(cue.end_sec, cue.start_sec + (wi + 1) * step), 3),
                    }
                    for wi, t in enumerate(toks)
                ]
            if not w_items:
                continue
            clip_cue_slices.append((idx, w_items))
            for wi in w_items:
                clip_flat_items.append(
                    {
                        "word": str(wi["word"]).strip(),
                        "orig_start": float(wi.get("start_sec", cue.start_sec)),
                        "orig_end": float(wi.get("end_sec", cue.end_sec)),
                        "probability": float(wi.get("probability", 0.95)),
                    }
                )

        if asr_words and clip_flat_items:
            asr_aligned_words, matched_cnt = _align_tokens_to_asr_words(
                expected_items=clip_flat_items,
                asr_words=asr_words,
                clip_start=float(clip.timeline_start_sec),
                clip_end=float(clip.timeline_end_sec),
                env=env,
                min_pause_sec=min_pause_sec,
            )
            if asr_aligned_words is not None and matched_cnt > 0 and len(asr_aligned_words) == len(clip_flat_items):
                cursor = 0
                for idx, orig_items in clip_cue_slices:
                    cue = aligned_tl.captions[idx]
                    n_w = len(orig_items)
                    seg_aligned = asr_aligned_words[cursor : cursor + n_w]
                    cursor += n_w
                    new_words: list[dict[str, Any]] = []
                    for orig_w, al_w in zip(orig_items, seg_aligned):
                        merged = dict(orig_w)
                        merged["word"] = str(orig_w.get("word", al_w["word"]))
                        merged["start_sec"] = round(float(al_w["start_sec"]), 3)
                        merged["end_sec"] = round(float(al_w["end_sec"]), 3)
                        merged["alignment_source"] = al_w.get("alignment_source", "asr_word_match")
                        new_words.append(merged)
                    cue.words = new_words
                    cue.start_sec = round(float(new_words[0]["start_sec"]), 3)
                    cue.end_sec = round(max(cue.start_sec + 0.14, float(new_words[-1]["end_sec"])), 3)
                    cue.aligned_to_final_audio = True
                continue

        # Stage 2: Acoustic speech-island fallback (for non-speech tones or zero ASR matches)
        # Partition cues into contiguous speech bursts separated by > 0.48s
        clusters: list[list[int]] = []
        for idx in idx_list:
            c = aligned_tl.captions[idx]
            if not clusters:
                clusters.append([idx])
            else:
                prev_c = aligned_tl.captions[clusters[-1][-1]]
                if c.start_sec - prev_c.end_sec <= 0.48:
                    clusters[-1].append(idx)
                else:
                    clusters.append([idx])

        prev_cluster_end = float(clip.timeline_start_sec)
        for cl_i, cl_indices in enumerate(clusters):
            first_c = aligned_tl.captions[cl_indices[0]]
            last_c = aligned_tl.captions[cl_indices[-1]]
            next_start = (
                float(aligned_tl.captions[clusters[cl_i + 1][0]].start_sec) - 0.02
                if cl_i + 1 < len(clusters)
                else float(clip.timeline_end_sec)
            )
            win_s = max(float(clip.timeline_start_sec), prev_cluster_end, float(first_c.start_sec) - 0.30)
            win_e = min(float(clip.timeline_end_sec), next_start, float(last_c.end_sec) + 0.30)
            if win_e <= win_s + 0.12:
                prev_cluster_end = max(prev_cluster_end, float(last_c.end_sec))
                continue

            det = detect_speech_islands_in_window(env, win_s, win_e, min_pause_sec=min_pause_sec)
            # Only re-anchor when real modulated speech islands exist (preserve flat silent/tone synthetic tests safely)
            if not det["islands_idx"] or float(det["peak_rms"]) < max(floor * 2.0, max_rms * 0.05):
                prev_cluster_end = max(prev_cluster_end, float(last_c.end_sec))
                continue

            cue_token_slices: list[tuple[int, list[dict[str, Any]]]] = []
            flat_tokens: list[str] = []
            flat_weights: list[float] = []
            flat_probs: list[float] = []
            for idx in cl_indices:
                cue = aligned_tl.captions[idx]
                if cue.words:
                    w_items = [dict(w) for w in cue.words if str(w.get("word", "")).strip()]
                else:
                    toks = [t for t in cue.text.split() if t.strip()]
                    w_items = [{"word": t} for t in toks]
                if not w_items:
                    continue
                cue_token_slices.append((idx, w_items))
                for wi in w_items:
                    tok = str(wi["word"]).strip()
                    flat_tokens.append(tok)
                    flat_weights.append(_syllable_word_weight(tok))
                    flat_probs.append(float(wi.get("probability", 0.95)))

            if not flat_tokens:
                continue

            aligned_words, _ = align_words_to_envelope(
                tokens=flat_tokens,
                word_weights=flat_weights,
                env=env,
                start_sec=win_s,
                end_sec=win_e,
                min_pause_sec=min_pause_sec,
                probs=flat_probs,
            )
            if len(aligned_words) != len(flat_tokens):
                continue

            cursor = 0
            for idx, orig_items in cue_token_slices:
                cue = aligned_tl.captions[idx]
                n_w = len(orig_items)
                seg_aligned = aligned_words[cursor : cursor + n_w]
                cursor += n_w
                new_words: list[dict[str, Any]] = []
                for orig_w, al_w in zip(orig_items, seg_aligned):
                    merged = dict(orig_w)
                    # Keep source timestamps intact if present; update ONLY output-relative timestamps
                    merged["word"] = str(orig_w.get("word", al_w["word"]))
                    merged["start_sec"] = round(float(al_w["start_sec"]), 3)
                    merged["end_sec"] = round(float(al_w["end_sec"]), 3)
                    merged["alignment_source"] = "acoustic_envelope"
                    new_words.append(merged)
                cue.words = new_words
                cue.start_sec = round(float(new_words[0]["start_sec"]), 3)
                cue.end_sec = round(max(cue.start_sec + 0.14, float(new_words[-1]["end_sec"])), 3)
                cue.aligned_to_final_audio = True
            prev_cluster_end = float(aligned_tl.captions[cl_indices[-1]].end_sec)

    # Guarantee strictly non-overlapping ascending cues and monotonic internal words after snapping
    _normalize_monotonic_cues_and_words(aligned_tl.captions)
    return aligned_tl


def validate_subtitle_integrity(
    timeline: RenderTimeline,
    final_audio_wav: Path | None = None,
    expected_segment_ids: list[str] | set[str] | None = None,
    final_asr_words: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Detect missing, duplicated, incorrectly ordered, unadjusted-source, and acoustically misaligned subtitles."""
    missing: list[dict[str, Any]] = []
    duplicated: list[dict[str, Any]] = []
    out_of_order: list[dict[str, Any]] = []
    misaligned: list[dict[str, Any]] = []
    warnings: list[str] = []

    cues = timeline.captions
    clips = timeline.clips
    clip_by_id = {c.clip_id: c for c in clips}
    clip_idx_by_id = {c.clip_id: i for i, c in enumerate(clips)}

    seen_ids: set[str] = set()
    prev_cue: CaptionCue | None = None
    prev_norm_txt = ""
    last_clip_idx = -1

    def _norm(txt: str) -> str:
        return re.sub(r"[^\w]+", " ", txt.casefold()).strip()

    env = compute_audio_envelope(final_audio_wav) if (final_audio_wav is not None and Path(final_audio_wav).exists()) else None
    has_real_speech = env is not None and float(env["max_rms"]) >= 5e-4

    cue_word_counts = [len([t for t in c.text.split() if t.strip()]) for c in cues]
    over_three = sum(1 for n in cue_word_counts if n > 3)

    for idx, cue in enumerate(cues):
        txt = cue.text.strip()
        ntxt = _norm(txt)
        if cue.cue_id in seen_ids:
            duplicated.append({"code": "DUPLICATE_CUE_ID", "cue_id": cue.cue_id, "time_sec": cue.start_sec})
        seen_ids.add(cue.cue_id)

        if not txt:
            missing.append({"code": "EMPTY_CUE_TEXT", "cue_id": cue.cue_id, "time_sec": cue.start_sec})
            continue

        if cue.end_sec <= cue.start_sec:
            out_of_order.append(
                {"code": "NON_POSITIVE_DURATION", "cue_id": cue.cue_id, "start_sec": cue.start_sec, "end_sec": cue.end_sec}
            )

        if prev_cue is not None:
            if cue.start_sec < prev_cue.start_sec - 0.005:
                out_of_order.append(
                    {
                        "code": "NON_MONOTONIC_CUE",
                        "cue_id": cue.cue_id,
                        "start_sec": cue.start_sec,
                        "prev_start_sec": prev_cue.start_sec,
                    }
                )
            elif cue.start_sec < prev_cue.end_sec - 0.045:
                out_of_order.append(
                    {
                        "code": "OVERLAPPING_CUES",
                        "cue_id": cue.cue_id,
                        "start_sec": cue.start_sec,
                        "prev_end_sec": prev_cue.end_sec,
                    }
                )

            gap_prev = cue.start_sec - prev_cue.end_sec
            toks_here = ntxt.split()
            if ntxt and ntxt == prev_norm_txt and (len(toks_here) >= 2 or gap_prev <= 0.65):
                duplicated.append(
                    {
                        "code": "DUPLICATE_CAPTION_TEXT",
                        "cue_id": cue.cue_id,
                        "prev_cue_id": prev_cue.cue_id,
                        "text": txt,
                        "time_sec": cue.start_sec,
                    }
                )

        # Verify internal word timing order and cue containment
        prev_we = -1.0
        for wi, w in enumerate(cue.words):
            ws = float(w.get("start_sec", 0.0))
            we = float(w.get("end_sec", 0.0))
            if we <= ws or (wi > 0 and ws < prev_we - 0.015):
                out_of_order.append(
                    {"code": "NON_MONOTONIC_WORDS", "cue_id": cue.cue_id, "word": w.get("word"), "start_sec": ws, "prev_end": prev_we}
                )
                break
            prev_we = max(prev_we, we)

        # Check clip ordering and unadjusted-source-timestamp leakage
        host_clip = clip_by_id.get(cue.clip_id or "") or _match_cue_clip(cue, clips)
        if cue.clip_id and cue.clip_id in clip_idx_by_id:
            ci = clip_idx_by_id[cue.clip_id]
            if ci < last_clip_idx:
                out_of_order.append(
                    {"code": "CLIP_ORDER_MISMATCH", "cue_id": cue.cue_id, "clip_id": cue.clip_id, "time_sec": cue.start_sec}
                )
            last_clip_idx = max(last_clip_idx, ci)

        if host_clip is not None:
            if cue.start_sec < host_clip.timeline_start_sec - 0.18 or cue.end_sec > host_clip.timeline_end_sec + 0.25:
                misaligned.append(
                    {
                        "code": "OUTSIDE_HOST_CLIP_BOUNDS",
                        "cue_id": cue.cue_id,
                        "clip_id": host_clip.clip_id,
                        "cue_window": [cue.start_sec, cue.end_sec],
                        "clip_window": [host_clip.timeline_start_sec, host_clip.timeline_end_sec],
                    }
                )
            if cue.source_start_sec is not None:
                expected_tl_s = host_clip.timeline_start_sec + (
                    float(cue.source_start_sec) - float(host_clip.source_start_sec)
                ) / max(0.25, float(host_clip.playback_speed))
                clip_dur = max(0.2, host_clip.timeline_end_sec - host_clip.timeline_start_sec)
                tol_unadj = max(1.45, 0.45 * clip_dur) if cue.aligned_to_final_audio else 0.65
                has_asr_match = any(w.get("alignment_source") == "asr_word_match" for w in cue.words)
                leaked_raw_src = (
                    abs(float(cue.start_sec) - float(cue.source_start_sec)) <= 0.12
                    and abs(expected_tl_s - float(cue.source_start_sec)) > 0.75
                    and (not cue.aligned_to_final_audio or not has_asr_match)
                )
                if leaked_raw_src or abs(float(cue.start_sec) - expected_tl_s) > tol_unadj:
                    misaligned.append(
                        {
                            "code": "UNADJUSTED_SOURCE_TIMESTAMP_REUSE",
                            "cue_id": cue.cue_id,
                            "clip_id": host_clip.clip_id,
                            "cue_start_sec": cue.start_sec,
                            "expected_timeline_start_sec": round(expected_tl_s, 3),
                            "source_start_sec": cue.source_start_sec,
                        }
                    )

        # Acoustic check against final edited audio
        if has_real_speech and env is not None:
            det = detect_speech_islands_in_window(env, cue.start_sec, cue.end_sec, min_pause_sec=0.10, pad_sec=0.06)
            floor = float(env["floor"])
            max_rms = float(env["max_rms"])
            has_silent_baseline = floor <= 0.025 and floor < max_rms * 0.20
            silence_thresh = max(floor * 1.85, max_rms * 0.03) if has_silent_baseline else min(max_rms * 0.03, floor * 0.35)
            if float(det["peak_rms"]) < silence_thresh:
                misaligned.append(
                    {
                        "code": "CUE_OVER_SILENCE_IN_FINAL_AUDIO",
                        "cue_id": cue.cue_id,
                        "text": txt,
                        "time_sec": cue.start_sec,
                        "peak_rms": round(float(det["peak_rms"]), 5),
                    }
                )

        prev_cue = cue
        prev_norm_txt = ntxt

    # Check missing captions across mapped source segments & speech-active clips
    cue_id_set = {c.cue_id for c in cues}
    mapped_seg_ids: set[str] = set()
    for m in timeline.segment_mappings:
        mapped_seg_ids.add(m.source_segment_id)
        has_linked = any(cid in cue_id_set for cid in m.cue_ids)
        has_overlap = any(
            min(c.end_sec, m.timeline_end_sec) - max(c.start_sec, m.timeline_start_sec) > 0.05
            for c in cues
        )
        if m.text_excerpt.strip() and not (has_linked or has_overlap):
            missing.append(
                {
                    "code": "MISSING_CAPTION_FOR_MAPPED_SEGMENT",
                    "source_segment_id": m.source_segment_id,
                    "clip_id": m.clip_id,
                    "timeline_window": [m.timeline_start_sec, m.timeline_end_sec],
                    "text_excerpt": m.text_excerpt,
                }
            )

    if expected_segment_ids is not None:
        cue_seg_ids = {c.source_segment_id for c in cues if c.source_segment_id}
        covered = mapped_seg_ids | cue_seg_ids
        for sid in expected_segment_ids:
            if sid not in covered:
                missing.append({"code": "EXPECTED_SEGMENT_UNCOVERED", "source_segment_id": sid})

    if has_real_speech and env is not None and cues and clips:
        for cl in clips:
            cl_dur = cl.timeline_end_sec - cl.timeline_start_sec
            if cl_dur < 1.5:
                continue
            has_cue = any(
                min(c.end_sec, cl.timeline_end_sec) - max(c.start_sec, cl.timeline_start_sec) > 0.08
                for c in cues
            )
            if not has_cue:
                det_cl = detect_speech_islands_in_window(env, cl.timeline_start_sec, cl.timeline_end_sec)
                speech_dur = sum(e - s for s, e in det_cl["islands_sec"])
                if speech_dur >= 0.70 and float(det_cl["peak_rms"]) >= max(float(env["floor"]) * 2.5, float(env["max_rms"]) * 0.12):
                    missing.append(
                        {
                            "code": "MISSING_CAPTION_ON_SPEECH_CLIP",
                            "clip_id": cl.clip_id,
                            "timeline_window": [cl.timeline_start_sec, cl.timeline_end_sec],
                            "detected_speech_sec": round(speech_dur, 2),
                        }
                    )

    # Summarize word-level provenance & verify drift when final ASR words are provided
    source_counts: dict[str, int] = {}
    total_word_count = 0
    for c in cues:
        for w in c.words:
            total_word_count += 1
            src = str(w.get("alignment_source") or "timeline_estimate")
            source_counts[src] = source_counts.get(src, 0) + 1

    word_drifts: list[float] = []
    if final_asr_words:
        # Compare ASR-matched cue words directly against matching final ASR word occurrences nearby,
        # using the exact same speech-island tightening & host clip boundary clamping as alignment.
        for c in cues:
            hc = clip_by_id.get(c.clip_id or "") or _match_cue_clip(c, clips)
            hc_s = float(hc.timeline_start_sec) if hc is not None else None
            hc_e = float(hc.timeline_end_sec) if hc is not None else None
            for w in c.words:
                w_txt = str(w.get("word", ""))
                ws = float(w.get("start_sec", c.start_sec))
                near_pairs: list[tuple[dict[str, Any], float]] = []
                for aw in final_asr_words:
                    if _word_match_similarity(w_txt, str(aw.get("word", ""))) < 0.65:
                        continue
                    aw_s, _aw_e, ok_tok = _tighten_asr_word_span(aw, env, clip_start=hc_s, clip_end=hc_e)
                    if not ok_tok:
                        aw_s = float(aw.get("start_sec", -99.0))
                    if abs(aw_s - ws) <= 1.25 or abs(float(aw.get("start_sec", -99.0)) - ws) <= 1.25:
                        near_pairs.append((aw, aw_s))
                if near_pairs:
                    best_aw, best_aw_s = min(near_pairs, key=lambda item: abs(item[1] - ws))
                    drift = abs(best_aw_s - ws)
                    word_drifts.append(drift)
                    if drift > 0.55 and w.get("alignment_source") == "asr_word_match":
                        misaligned.append(
                            {
                                "code": "ASR_WORD_DRIFT_EXCEEDED",
                                "cue_id": c.cue_id,
                                "word": w_txt,
                                "cue_word_start_sec": ws,
                                "asr_word_start_sec": best_aw_s,
                                "drift_sec": round(drift, 3),
                            }
                        )

    if over_three > 0:
        warnings.append(f"{over_three} caption cue(s) exceed 3 words (max={max(cue_word_counts, default=0)}).")

    all_errors = missing + duplicated + out_of_order + misaligned
    return {
        "passed": len(all_errors) == 0,
        "missing": missing,
        "duplicated": duplicated,
        "out_of_order": out_of_order,
        "misaligned": misaligned,
        "warnings": warnings,
        "error_count": len(all_errors),
        "cue_count": len(cues),
        "aligned_cue_count": sum(1 for c in cues if c.aligned_to_final_audio),
        "total_word_count": total_word_count,
        "asr_matched_word_count": source_counts.get("asr_word_match", 0),
        "alignment_sources": source_counts,
        "mean_word_onset_drift_sec": round(sum(word_drifts) / len(word_drifts), 3) if word_drifts else None,
        "max_word_onset_drift_sec": round(max(word_drifts), 3) if word_drifts else None,
        "word_count_gt_3": over_three,
        "max_words_per_cue": max(cue_word_counts, default=0),
    }


def check_caption_quality(
    timeline: RenderTimeline,
    final_audio_wav: Path | None = None,
) -> dict[str, Any]:
    """Validate subtitle cues for safe margins, readable line width, timing sanity, and structural integrity."""
    is_vertical = timeline.platform_format != TargetPlatformFormat.YOUTUBE_LONG or timeline.height > timeline.width
    cs = timeline.caption_settings
    margin_v = int(cs.safe_margin_bottom_px)
    min_safe_y = 240 if is_vertical else 70
    max_chars_line = 32 if is_vertical else 44

    issues: list[str] = []
    warnings: list[str] = []
    prev_end = -1.0
    max_observed_line_chars = 0

    for idx, cue in enumerate(timeline.captions):
        txt = cue.text.strip()
        if not txt:
            warnings.append(f"Empty caption cue at index {idx} ({cue.start_sec:.2f}s).")
            continue
        dur = cue.end_sec - cue.start_sec
        if dur < 0.25:
            warnings.append(f"Ultra-short caption cue ({dur:.2f}s) at {cue.start_sec:.2f}s: {txt[:30]}")
        if prev_end > cue.start_sec + 0.04:
            warnings.append(f"Overlapping caption cues around {cue.start_sec:.2f}s (prev end {prev_end:.2f}s).")
        prev_end = max(prev_end, cue.end_sec)

        # Estimate wrapped longest line length
        items = [(p, 20) for p in txt.split() if p]
        wrapped = _wrap_caption_tokens(items, max_words=cs.max_words_per_line, max_chars_per_line=max_chars_line)
        for ln in wrapped:
            ln_len = sum(len(w) for w, _ in ln) + max(0, len(ln) - 1)
            max_observed_line_chars = max(max_observed_line_chars, ln_len)
            if ln_len > max_chars_line + 14:
                warnings.append(f"Long subtitle line ({ln_len} chars) at {cue.start_sec:.2f}s.")

    if margin_v < min_safe_y:
        issues.append(f"Caption bottom safe margin {margin_v}px violates {min_safe_y}px platform safe zone.")

    integrity = validate_subtitle_integrity(timeline, final_audio_wav=final_audio_wav)
    if not integrity["passed"]:
        for cat in ("missing", "duplicated", "out_of_order", "misaligned"):
            for item in integrity[cat][:2]:
                issues.append(f"Subtitle integrity ({cat}): {item.get('code')} @ {item.get('time_sec', item.get('clip_id', ''))}")
    warnings.extend(integrity["warnings"])

    return {
        "passed": len(issues) == 0,
        "ok": len(issues) == 0,
        "cue_count": len(timeline.captions),
        "margin_bottom_px": margin_v,
        "max_chars_per_line_cap": max_chars_line,
        "max_observed_line_chars": max_observed_line_chars,
        "max_line_chars": max_observed_line_chars,
        "integrity": integrity,
        "issues": issues,
        "warnings": warnings,
    }


def write_ass(timeline: RenderTimeline, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cs = timeline.caption_settings
    font_size = max(24, int(round(cs.font_size * cs.scale_factor)))
    is_vertical = timeline.platform_format != TargetPlatformFormat.YOUTUBE_LONG or timeline.height > timeline.width
    # Respect platform safe zones (Shorts/TikTok/Reels put buttons/captions at bottom 240-300px)
    margin_v = max(cs.safe_margin_bottom_px, 280 if is_vertical else 85)
    margin_lr = 74 if is_vertical else 110
    max_chars_line = 30 if is_vertical else 44

    # PrimaryColour = vibrant neon yellow-gold karaoke highlight, SecondaryColour = crisp pure white before sweep
    outline_px = 4.8 if cs.max_words_per_line <= 4 else 3.6
    shadow_px = 2.4 if cs.max_words_per_line <= 4 else 1.8
    suppress_bracket_noise = timeline.style_profile in {"zoomer_vlog", "short_form_viral", "comedy"}
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {timeline.width}
PlayResY: {timeline.height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cs.font_name},{font_size},&H001CE8FF,&H00FFFFFF,&H00080808,&HA0000000,-1,0,0,0,100,100,0.4,0,1,{outline_px},{shadow_px},2,{margin_lr},{margin_lr},{margin_v},1
Style: Uncertain,{cs.font_name},{max(22, font_size - 4)},&H001CE8FF,&H00F8FAFC,&H000A0A0A,&H95000000,-1,0,0,0,100,100,0.2,0,1,{max(3.0, outline_px - 0.6)},{shadow_px},2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    for cue in timeline.captions:
        style_name = "Uncertain" if (cue.low_confidence and not suppress_bracket_noise) else "Default"
        body = _format_karaoke_line(
            cue,
            cs.max_words_per_line,
            cs.highlight_active_word,
            max_chars_per_line=max_chars_line,
        )
        if cue.low_confidence and not suppress_bracket_noise and cue.style_preset != "kinetic_caption_pop":
            body = f"[?] {body}"
        events.append(
            f"Dialogue: 0,{_fmt_ass_time(cue.start_sec)},{_fmt_ass_time(cue.end_sec)},{style_name},,0,0,0,,{body}"
        )

    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_path
