"""Editorial Hook Director, Ending Director, and Narrative Coherence Planner."""

from __future__ import annotations

import re
from typing import Any

from symposium_studio.engines.analysis.semantics import (
    CURIOSITY_MARKERS,
    DANGLING_STARTERS,
    INCOMPLETE_ENDINGS,
    PREMISE_SETUP_MARKERS,
    _tokenize,
)
from symposium_studio.models.media import HighlightProposal, MediaAnalysisBundle, TranscriptSegment

CLOSURE_MARKERS = (
    "хорошо",
    "рахмет",
    "жақсы",
    "ашамын",
    "көреміз",
    "туған күн",
    "вывод",
    "итог",
    "подписы",
    "коммент",
    "telegram",
    "conclusion",
    "lesson",
    "therefore",
    "remember",
    "result",
)


def _overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0.0:
        return 0.0
    shortest = max(0.25, min(a1 - a0, b1 - b0))
    return inter / shortest


def _segments_in_span(segments: list[TranscriptSegment], start_sec: float, end_sec: float) -> list[TranscriptSegment]:
    return [s for s in segments if s.end_sec > start_sec + 0.15 and s.start_sec < end_sec - 0.15]


def _tight_speech_start(seg: TranscriptSegment, default_start: float) -> float:
    """Trim leading silence cleanly to ~0.12s before first detected word when word timestamps exist."""
    if seg.words:
        first_w = seg.words[0]
        return round(max(0.0, max(seg.start_sec, first_w.start_sec - 0.12)), 3)
    return round(max(0.0, max(default_start, seg.start_sec - 0.08)), 3)


def _tight_speech_end(seg: TranscriptSegment, max_duration: float) -> float:
    """Preserve a clean natural breath tail (~0.30s after final word) without cutting mid-syllable."""
    end_w = seg.words[-1].end_sec if seg.words else seg.end_sec
    natural_end = max(seg.end_sec, end_w + 0.30)
    return round(min(max_duration, natural_end), 3)


def evaluate_hook_candidates(
    analysis: MediaAnalysisBundle,
    max_candidates: int = 6,
) -> list[dict[str, Any]]:
    """Score highlight and natural opening spans as candidates for the first 3-18 seconds."""
    segs = analysis.transcript.segments
    hls = analysis.highlights
    dur_total = max(1.0, analysis.probe.duration_sec)
    candidates: list[dict[str, Any]] = []
    seen_spans: list[tuple[float, float]] = []

    # 1. Explicit opening setup span (cover first self-contained premise up to ~24s if multi-segment)
    if segs:
        open_segs: list[TranscriptSegment] = []
        for s in segs:
            open_segs.append(s)
            span_d = open_segs[-1].end_sec - open_segs[0].start_sec
            txt_so_far = " ".join(x.text for x in open_segs)
            if span_d >= 11.0 and s.text.strip().endswith((".", "!", "?")) and (
                any(m in txt_so_far.lower() for m in PREMISE_SETUP_MARKERS) or span_d >= 16.0
            ):
                break
            if span_d >= 24.5:
                break
        if open_segs:
            s0 = _tight_speech_start(open_segs[0], open_segs[0].start_sec)
            e0 = min(dur_total, open_segs[-1].end_sec + 0.25)
            seen_spans.append((s0, e0))
            candidates.append(
                {
                    "highlight_id": hls[0].highlight_id if hls else "hl_open_premise",
                    "start_sec": round(s0, 3),
                    "end_sec": round(e0, 3),
                    "text": " ".join(x.text for x in open_segs),
                    "segs": open_segs,
                    "is_opening_position": True,
                }
            )

    # 2. Evaluate all semantic highlights as candidate hooks
    for h in hls:
        h_segs = _segments_in_span(segs, h.start_sec, h.end_sec)
        if not h_segs:
            continue
        s0 = _tight_speech_start(h_segs[0], h.start_sec)
        e0 = min(dur_total, h_segs[-1].end_sec + 0.20)
        if any(_overlap_ratio(s0, e0, a, b) >= 0.85 for a, b in seen_spans):
            # Still prefer premise-trimmed span if existing opening candidate exists
            continue
        seen_spans.append((s0, e0))
        candidates.append(
            {
                "highlight_id": h.highlight_id,
                "start_sec": round(s0, 3),
                "end_sec": round(e0, 3),
                "text": " ".join(x.text for x in h_segs),
                "segs": h_segs,
                "is_opening_position": h.start_sec < max(6.0, dur_total * 0.08),
            }
        )

    scored: list[dict[str, Any]] = []
    for cand in candidates:
        c_segs: list[TranscriptSegment] = cand["segs"]
        txt = cand["text"].strip()
        low = txt.lower()
        toks = _tokenize(txt)
        first_tok = toks[0] if toks else ""
        first_seg_txt = c_segs[0].text.strip()

        starts_lower = bool(first_seg_txt and first_seg_txt[0].islower())
        starts_comma = first_seg_txt.startswith((",", ";", ":", "-", "—"))
        dangling_start = first_tok in DANGLING_STARTERS or starts_lower or starts_comma

        has_premise = any(m in low for m in PREMISE_SETUP_MARKERS)
        curiosity_hits = sum(1 for m in CURIOSITY_MARKERS if m in low)
        mean_conf = sum(s.confidence for s in c_segs) / max(1, len(c_segs))
        first_conf = c_segs[0].confidence
        low_conf_frac = sum(1 for s in c_segs if s.low_confidence) / max(1, len(c_segs))

        # Visual stability & face visibility inside hook
        vis = [v for v in analysis.visual_samples if cand["start_sec"] <= v.time_sec <= cand["end_sec"]]
        face_ratio = sum(1 for v in vis if v.faces) / max(1, len(vis)) if vis else 0.75
        sharp_mean = sum(v.sharpness_score for v in vis) / max(1, len(vis)) if vis else 0.7

        premise_score = 0.96 if has_premise else (0.76 if cand["is_opening_position"] else 0.48)
        clean_start_score = 0.22 if dangling_start else (0.62 if c_segs[0].low_confidence else 0.94)
        curiosity_score = min(1.0, 0.35 + curiosity_hits * 0.14 + (0.18 if has_premise else 0.0))
        trust_score = max(0.15, min(1.0, mean_conf * 0.7 + first_conf * 0.3 - low_conf_frac * 0.25))
        visual_score = max(0.3, min(1.0, face_ratio * 0.65 + min(1.0, sharp_mean * 1.2) * 0.35))

        # Duration sweet spot (6-24s gives clear context on YouTube, tight hook before body)
        c_dur = max(0.5, cand["end_sec"] - cand["start_sec"])
        dur_fit = 0.95 if 6.0 <= c_dur <= 24.5 else (0.72 if c_dur >= 3.5 else 0.45)

        total_score = round(
            0.28 * premise_score
            + 0.26 * clean_start_score
            + 0.18 * curiosity_score
            + 0.16 * trust_score
            + 0.06 * visual_score
            + 0.06 * dur_fit,
            3,
        )

        issues: list[str] = []
        if dangling_start:
            issues.append(f"dangling_start('{first_tok or first_seg_txt[:15]}')")
        if c_segs[0].low_confidence:
            issues.append("low_confidence_opening_segment")
        if not has_premise and not cand["is_opening_position"]:
            issues.append("lacks_initial_viewer_premise")

        scored.append(
            {
                "highlight_id": cand["highlight_id"],
                "start_sec": cand["start_sec"],
                "end_sec": cand["end_sec"],
                "duration_sec": round(c_dur, 2),
                "hook_score": total_score,
                "clean_start": not dangling_start,
                "has_premise": has_premise,
                "trust_score": round(trust_score, 3),
                "issues": issues,
                "excerpt": txt[:140],
                "rationale": (
                    f"Hook score {total_score:.3f}: premise={premise_score:.2f}, clean_start={clean_start_score:.2f}, "
                    f"curiosity={curiosity_score:.2f}, asr_trust={trust_score:.2f}"
                ),
            }
        )

    scored.sort(key=lambda d: (d["clean_start"], d["hook_score"]), reverse=True)
    return scored[:max_candidates]


def select_best_hook(analysis: MediaAnalysisBundle) -> dict[str, Any]:
    """Choose the single strongest self-contained opening hook without fabricating speech."""
    ranked = evaluate_hook_candidates(analysis, max_candidates=6)
    if not ranked:
        return {
            "highlight_id": "hl_00",
            "start_sec": 0.0,
            "end_sec": min(12.0, analysis.probe.duration_sec),
            "duration_sec": min(12.0, analysis.probe.duration_sec),
            "hook_score": 0.5,
            "clean_start": True,
            "has_premise": False,
            "trust_score": 0.5,
            "issues": ["fallback_empty_transcript"],
            "excerpt": "",
            "rationale": "Fallback 0s opening because no transcript segments were available.",
            "alternatives": [],
        }
    best = dict(ranked[0])
    best["alternatives"] = ranked[1:]
    return best


def evaluate_ending_closure(
    segments: list[TranscriptSegment],
    clip_start_sec: float,
    clip_end_sec: float,
) -> dict[str, Any]:
    """Verify that a closing clip ends on a finished thought/sentence rather than cutting mid-clause."""
    span_segs = _segments_in_span(segments, clip_start_sec, clip_end_sec)
    if not span_segs:
        return {
            "closure_ok": True,
            "issues": [],
            "closure_type": "no_speech_span",
            "spoken_end_text": "",
        }
    last_seg = span_segs[-1]
    txt = last_seg.text.strip()
    toks = _tokenize(txt)
    last_tok = toks[-1] if toks else ""
    issues: list[str] = []

    # Check if clip_end_sec slices significantly before last segment / final word completes
    last_word_end = last_seg.words[-1].end_sec if last_seg.words else last_seg.end_sec
    if clip_end_sec + 0.20 < min(last_seg.end_sec, last_word_end):
        issues.append(
            f"Clip ends at {clip_end_sec:.2f}s before closing phrase '{txt[-32:]}' finishes ({last_word_end:.2f}s)."
        )

    ends_punct = txt.endswith((".", "!", "?"))
    dangling_end = (last_tok in INCOMPLETE_ENDINGS) or txt.endswith((",", "-", "—", ":"))
    if dangling_end:
        issues.append(f"Closing segment ends on incomplete connector '{last_tok or txt[-8:]}'.")

    has_closure_cue = any(m in " ".join(s.text.lower() for s in span_segs[-2:]) for m in CLOSURE_MARKERS)
    closure_type = (
        "strong_payoff_closure"
        if (ends_punct and has_closure_cue and not dangling_end)
        else ("clean_sentence_end" if (ends_punct and not dangling_end and not issues) else "incomplete_or_abrupt")
    )

    return {
        "closure_ok": len(issues) == 0 and not dangling_end,
        "issues": issues,
        "closure_type": closure_type,
        "spoken_end_text": txt,
        "last_segment_end_sec": last_seg.end_sec,
        "has_closure_cue": has_closure_cue,
    }


def snap_ending_boundary(
    analysis: MediaAnalysisBundle,
    proposed_start_sec: float,
    proposed_end_sec: float,
    max_extend_sec: float = 10.0,
) -> tuple[float, float, dict[str, Any]]:
    """Snap a beat/clip's end forward cleanly to finish the active sentence or closing payoff."""
    segs = analysis.transcript.segments
    dur_total = max(0.5, analysis.probe.duration_sec)
    if not segs:
        return proposed_start_sec, min(dur_total, proposed_end_sec), {
            "closure_ok": True,
            "issues": [],
            "closure_type": "no_transcript",
            "spoken_end_text": "",
        }

    span_segs = _segments_in_span(segs, proposed_start_sec, proposed_end_sec)
    if not span_segs:
        return proposed_start_sec, min(dur_total, proposed_end_sec), evaluate_ending_closure(
            segs, proposed_start_sec, proposed_end_sec
        )

    last_idx = next((i for i, s in enumerate(segs) if s.segment_id == span_segs[-1].segment_id), len(segs) - 1)
    # Walk forward at most max_extend_sec if ending cuts mid-sentence or on an incomplete word/phrase
    chosen_end = max(proposed_end_sec, _tight_speech_end(segs[last_idx], dur_total))
    cur_idx = last_idx
    while cur_idx < len(segs):
        cur_seg = segs[cur_idx]
        cur_txt = cur_seg.text.strip()
        cur_toks = _tokenize(cur_txt)
        ends_dangling = (cur_toks[-1] in INCOMPLETE_ENDINGS) if cur_toks else False
        ends_clean = cur_txt.endswith((".", "!", "?")) and not ends_dangling
        next_closure = False
        if cur_idx + 1 < len(segs):
            nxt = segs[cur_idx + 1]
            # Include short immediate closing remark ("Хорошо." / "Бір жыл екі ай как раз.") if right next to end
            if nxt.end_sec - proposed_end_sec <= max_extend_sec and (
                nxt.start_sec - cur_seg.end_sec <= 2.2
            ) and (
                len(nxt.text.split()) <= 6
                or any(m in nxt.text.lower() for m in CLOSURE_MARKERS)
                or not ends_clean
            ):
                next_closure = True
        if ends_clean and not next_closure:
            chosen_end = max(chosen_end, _tight_speech_end(cur_seg, dur_total))
            break
        if cur_idx + 1 < len(segs) and segs[cur_idx + 1].end_sec - proposed_end_sec <= max_extend_sec:
            cur_idx += 1
            chosen_end = _tight_speech_end(segs[cur_idx], dur_total)
        else:
            chosen_end = _tight_speech_end(cur_seg, dur_total)
            break

    # Trim clean start as well if span starts near segment onset
    first_seg = span_segs[0]
    snapped_start = _tight_speech_start(first_seg, proposed_start_sec) if abs(proposed_start_sec - first_seg.start_sec) <= 1.2 else proposed_start_sec
    snapped_end = round(min(dur_total, max(snapped_start + 1.0, chosen_end)), 3)
    report = evaluate_ending_closure(segs, snapped_start, snapped_end)
    return snapped_start, snapped_end, report


def select_closing_payoff(
    analysis: MediaAnalysisBundle,
    exclude_spans: list[tuple[float, float]],
) -> HighlightProposal | None:
    """Pick the strongest late-video closing payoff highlight that does not overlap already selected spans."""
    hls = analysis.highlights
    segs = analysis.transcript.segments
    dur_total = max(1.0, analysis.probe.duration_sec)
    scored: list[tuple[float, HighlightProposal]] = []

    for h in hls:
        if any(_overlap_ratio(h.start_sec, h.end_sec, a, b) >= 0.35 for a, b in exclude_spans):
            continue
        pos = h.start_sec / dur_total
        end_pos = h.end_sec / dur_total
        ev = evaluate_ending_closure(segs, h.start_sec, h.end_sec)
        tail_txt = (ev.get("spoken_end_text") or h.transcript_excerpt).lower()
        closure_bonus = 0.35 if (any(m in tail_txt for m in CLOSURE_MARKERS) or ev.get("has_closure_cue")) else 0.0
        late_bonus = 0.48 if pos >= 0.78 else (0.24 if pos >= 0.65 else (0.04 if pos >= 0.45 else -0.25))
        terminal_bonus = 0.22 if end_pos >= 0.88 else 0.0
        clean_bonus = 0.25 if ev["closure_ok"] else -0.25
        role_bonus = 0.14 if h.role == "payoff" else 0.0
        sc = h.scores.composite_score * 0.45 + closure_bonus + late_bonus + terminal_bonus + clean_bonus + role_bonus
        scored.append((sc, h))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def plan_coherent_beat_sequence(
    analysis: MediaAnalysisBundle,
    target_min_sec: float,
    target_max_sec: float,
    strategy: str = "hook_first",
) -> list[HighlightProposal]:
    """Assemble Hook -> chronological Body -> Intentional Payoff with zero duplicate/overlapping spans."""
    hls = list(analysis.highlights)
    if not hls:
        return []
    segs = analysis.transcript.segments
    dur_total = max(1.0, analysis.probe.duration_sec)

    hook_info = select_best_hook(analysis)
    hook_hl = next((h for h in hls if h.highlight_id == hook_info["highlight_id"]), hls[0])

    # Build a clean snapped copy of the opening hook proposal
    hook_start = hook_info["start_sec"]
    hook_end = hook_info["end_sec"]
    if strategy == "mechanism_proof":
        clean_hl = max(
            hls,
            key=lambda h: (h.scores.context_dependency >= 0.7, h.scores.visual_opportunity + h.scores.educational_value),
        )
        if clean_hl.start_sec <= hook_start + 5.0:
            hook_hl = clean_hl
            hook_start, hook_end = clean_hl.start_sec, clean_hl.end_sec

    # If chosen hook starts shortly after a clean opening premise (e.g. 12s after 0..12s setup), expand backward to include the premise seamlessly
    if 2.5 < hook_start <= 16.0 and segs:
        pre_segs = [s for s in segs if s.end_sec <= hook_start + 0.5]
        if pre_segs and not pre_segs[0].low_confidence and not (pre_segs[0].text.strip()[:1].islower()):
            hook_start = _tight_speech_start(pre_segs[0], pre_segs[0].start_sec)

    hook_snap = hook_hl.model_copy(
        update={
            "role": "hook",
            "start_sec": round(hook_start, 3),
            "end_sec": round(hook_end, 3),
            "duration_sec": round(max(1.0, hook_end - hook_start), 3),
        }
    )
    used_spans: list[tuple[float, float]] = [(hook_snap.start_sec, hook_snap.end_sec)]

    # Reserve closing payoff beat early so target duration budget doesn't chop the finale
    payoff_hl = select_closing_payoff(analysis, used_spans)
    payoff_snap: HighlightProposal | None = None
    payoff_dur = 0.0
    if payoff_hl is not None:
        ps0, ps1, _ = snap_ending_boundary(analysis, payoff_hl.start_sec, payoff_hl.end_sec, max_extend_sec=10.0)
        # Ensure payoff doesn't collide with hook
        if _overlap_ratio(ps0, ps1, hook_snap.start_sec, hook_snap.end_sec) < 0.25:
            payoff_snap = payoff_hl.model_copy(
                update={
                    "role": "payoff",
                    "start_sec": ps0,
                    "end_sec": ps1,
                    "duration_sec": round(max(1.0, ps1 - ps0), 3),
                }
            )
            payoff_dur = payoff_snap.duration_sec
            used_spans.append((ps0, ps1))

    # Select middle body beats strictly between hook and payoff (avoiding overlaps & backward jumps)
    body_budget = max(0.0, target_max_sec - hook_snap.duration_sec - payoff_dur)
    remaining = []
    for h in hls:
        if any(_overlap_ratio(h.start_sec, h.end_sec, a, b) >= 0.30 for a, b in used_spans):
            continue
        # Never place an earlier clip after a natural-start hook, and never place a post-payoff clip before payoff
        if hook_snap.start_sec < 20.0 and h.end_sec <= hook_snap.start_sec + 1.0:
            continue
        if payoff_snap is not None and h.start_sec >= payoff_snap.start_sec - 0.5:
            continue
        remaining.append(h)

    remaining_ranked = sorted(
        remaining,
        key=lambda h: h.scores.composite_score * 0.75 + h.scores.context_dependency * 0.15 + h.scores.visual_opportunity * 0.10,
        reverse=True,
    )

    chosen_body: list[HighlightProposal] = []
    acc_body = 0.0
    for h in remaining_ranked:
        if any(_overlap_ratio(h.start_sec, h.end_sec, x.start_sec, x.end_sec) >= 0.30 for x in chosen_body):
            continue
        cur_total = hook_snap.duration_sec + acc_body + payoff_dur
        if cur_total >= target_min_sec and acc_body + h.duration_sec > body_budget + 8.0:
            break
        if hook_snap.duration_sec + acc_body + payoff_dur + h.duration_sec <= target_max_sec + 10.0 or not chosen_body:
            h_segs = _segments_in_span(segs, h.start_sec, h.end_sec)
            hs0 = _tight_speech_start(h_segs[0], h.start_sec) if h_segs else h.start_sec
            hs1 = min(dur_total, _tight_speech_end(h_segs[-1], dur_total)) if h_segs else h.end_sec
            for ua, ub in used_spans + [(c.start_sec, c.end_sec) for c in chosen_body]:
                if hs0 < ua < hs1:
                    hs1 = min(hs1, ua - 0.05)
                if hs0 < ub < hs1:
                    hs0 = max(hs0, ub + 0.05)
            if hs1 - hs0 >= 3.5:
                chosen_body.append(
                    h.model_copy(
                        update={
                            "start_sec": round(hs0, 3),
                            "end_sec": round(hs1, 3),
                            "duration_sec": round(hs1 - hs0, 3),
                        }
                    )
                )
                acc_body += hs1 - hs0
        if hook_snap.duration_sec + acc_body + payoff_dur >= target_max_sec:
            break

    # Chronological narrative flow in middle body (never jump backward right before payoff)
    chosen_body.sort(key=lambda h: h.start_sec)

    final_seq: list[HighlightProposal] = [hook_snap] + chosen_body
    if payoff_snap is not None:
        final_seq.append(payoff_snap)

    # Final ending check: snap very last beat cleanly
    if final_seq:
        last_h = final_seq[-1]
        fs0, fs1, _ = snap_ending_boundary(analysis, last_h.start_sec, last_h.end_sec, max_extend_sec=8.0)
        if not any(_overlap_ratio(fs0, fs1, o.start_sec, o.end_sec) >= 0.40 for o in final_seq[:-1]):
            final_seq[-1] = last_h.model_copy(
                update={
                    "start_sec": fs0,
                    "end_sec": fs1,
                    "duration_sec": round(max(1.0, fs1 - fs0), 3),
                }
            )
    return final_seq
