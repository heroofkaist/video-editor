"""Semantic topic segmentation, repetition checks, and multi-factor highlight scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from symposium_studio.models.media import (
    HighlightProposal,
    HighlightScoreBreakdown,
    TopicCluster,
    TranscriptSegment,
    VisualContextSample,
)

DOMAIN_KEYWORDS: dict[str, tuple[str, str]] = {
    "physics": (
        "magnetic|flux|coil|induction|faraday|current|voltage|field|electron|quantum|energy|wave|force|gravity|momentum|entropy|laser",
        "em_induction_faraday",
    ),
    "electrical_engineering": (
        "circuit|resistor|capacitor|inductor|oscilloscope|signal|waveform|frequency|impedance|transistor|amplifier|pcb|microcontroller|pwm",
        "rlc_signal_flow",
    ),
    "programming": (
        "algorithm|tree|array|hash|graph|complexity|binary|recursion|cache|thread|compiler|python|memory|pointer|database|latency|api",
        "binary_tree_trace",
    ),
    "storytelling": (
        "mistake|lesson|failed|learned|story|week|month|year|first|surprised|built|launched|experiment|discovered|why|problem",
        "narrative_timeline",
    ),
}

CURIOSITY_MARKERS = (
    "why",
    "how",
    "secret",
    "mistake",
    "actually",
    "instead",
    "nobody",
    "surprising",
    "watch",
    "happens",
    "breaks",
    "problem",
    "question",
    "?",
)

FILLER_PHRASES = ("um", "uh", "you know", "basically", "actually um", "sort of", "like i said")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁәіңғүұқөһ가-힣0-9]+", text.lower())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _detect_domain(text: str) -> tuple[str, str | None, bool]:
    low = text.lower()
    best_domain = "storytelling"
    best_template = "narrative_timeline"
    best_hits = 0
    for domain, (pattern, template) in DOMAIN_KEYWORDS.items():
        hits = len(re.findall(rf"\b(?:{pattern})\b", low))
        if hits > best_hits:
            best_hits = hits
            best_domain = domain
            best_template = template
    is_tech = best_domain in {"physics", "electrical_engineering", "programming"} and best_hits > 0
    return best_domain, (best_template if best_hits > 0 else "kinetic_callout"), is_tech


def detect_repetitions(segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
    flagged: list[dict[str, Any]] = []
    for i in range(1, len(segments)):
        prev_toks = set(_tokenize(segments[i - 1].text))
        cur_toks = set(_tokenize(segments[i].text))
        sim = _jaccard(prev_toks, cur_toks)
        if sim >= 0.72 and len(cur_toks) >= 4:
            flagged.append(
                {
                    "type": "adjacent_repetition",
                    "segment_id": segments[i].segment_id,
                    "previous_segment_id": segments[i - 1].segment_id,
                    "similarity": round(sim, 3),
                    "start_sec": segments[i].start_sec,
                    "end_sec": segments[i].end_sec,
                    "text": segments[i].text,
                }
            )
    return flagged


def segment_topics(segments: list[TranscriptSegment], target_cluster_sec: float = 28.0) -> list[TopicCluster]:
    if not segments:
        return []

    clusters: list[list[TranscriptSegment]] = [[segments[0]]]
    for seg in segments[1:]:
        cur_group = clusters[-1]
        cur_dur = cur_group[-1].end_sec - cur_group[0].start_sec
        gap = seg.start_sec - cur_group[-1].end_sec
        prev_words = set(_tokenize(" ".join(s.text for s in cur_group[-2:])))
        cur_words = set(_tokenize(seg.text))
        sim = _jaccard(prev_words, cur_words)

        should_split = (cur_dur >= target_cluster_sec and sim < 0.22) or gap > 2.2 or cur_dur >= target_cluster_sec * 1.5
        if should_split:
            clusters.append([seg])
        else:
            cur_group.append(seg)

    out: list[TopicCluster] = []
    for idx, grp in enumerate(clusters):
        combined = " ".join(s.text for s in grp)
        toks = [t for t in _tokenize(combined) if len(t) > 3]
        top_kw = [w for w, _ in Counter(toks).most_common(5)]
        domain, _, is_tech = _detect_domain(combined)
        first_sentence = grp[0].text.strip()
        title = first_sentence[:64] + ("..." if len(first_sentence) > 64 else "")
        out.append(
            TopicCluster(
                topic_id=f"topic_{idx:02d}",
                title=title,
                start_sec=grp[0].start_sec,
                end_sec=grp[-1].end_sec,
                keywords=top_kw,
                summary=combined[:220],
                is_technical=is_tech,
                domain=domain,
            )
        )
    return out


def score_highlights(
    segments: list[TranscriptSegment],
    topics: list[TopicCluster],
    visual_samples: list[VisualContextSample],
    duration_sec: float,
) -> list[HighlightProposal]:
    if not segments:
        return []

    # Build sliding/topic-aligned candidate windows (10-28s)
    windows: list[list[TranscriptSegment]] = []
    buf: list[TranscriptSegment] = []
    for seg in segments:
        buf.append(seg)
        span = buf[-1].end_sec - buf[0].start_sec
        if span >= 12.0:
            windows.append(list(buf))
            buf = [seg] if len(buf) > 2 else []
    if buf:
        if windows and (buf[-1].end_sec - buf[0].start_sec) < 6.0:
            windows[-1].extend(buf)
        else:
            windows.append(buf)

    all_tokens = Counter(_tokenize(" ".join(s.text for s in segments)))
    total_toks = max(1, sum(all_tokens.values()))

    proposals: list[HighlightProposal] = []
    for idx, win in enumerate(windows):
        start_s = win[0].start_sec
        end_s = win[-1].end_sec
        dur = max(0.5, end_s - start_s)
        text = " ".join(s.text for s in win)
        toks = _tokenize(text)
        unique_ratio = len(set(toks)) / max(1, len(toks))

        # 1. Information density: unique content words per second
        words_per_sec = len(toks) / dur
        info_density = min(1.0, (unique_ratio * 0.55) + min(1.0, words_per_sec / 3.2) * 0.45)

        # 2. Novelty: inverse frequency of tokens across transcript
        avg_rarity = sum(1.0 - min(0.95, all_tokens[t] / total_toks * 12.0) for t in toks) / max(1, len(toks))
        novelty = max(0.2, min(1.0, avg_rarity))

        # 3. Story relevance: early hook or structural transition or high topic overlap
        pos_ratio = start_s / max(1.0, duration_sec)
        story_rel = 0.85 if pos_ratio < 0.15 or pos_ratio > 0.8 else 0.7

        # 4. Audience curiosity: questions, contrast words, hook cues
        low_txt = text.lower()
        curiosity_hits = sum(1 for m in CURIOSITY_MARKERS if m in low_txt)
        curiosity = min(1.0, 0.35 + curiosity_hits * 0.15)

        # 5. Educational value: domain terminology & causal structure ("because", "therefore", "means", "equation")
        domain, suggested_template, is_tech = _detect_domain(text)
        causal_hits = len(re.findall(r"\b(because|therefore|means|causes|results|formula|law|step|when)\b", low_txt))
        edu_val = min(1.0, (0.55 if is_tech else 0.35) + causal_hits * 0.12)

        # 6. Emotional intensity: punctuation/emphasis + speech rate + visual motion
        vis_in_win = [v for v in visual_samples if start_s <= v.time_sec <= end_s]
        mean_motion = sum(v.motion_score for v in vis_in_win) / max(1, len(vis_in_win))
        em_hits = text.count("!") + len(re.findall(r"\b(amazing|crazy|huge|critical|never|always|breakthrough)\b", low_txt))
        emotional = min(1.0, 0.35 + em_hits * 0.15 + min(0.3, mean_motion * 3.0))

        # 7. Visual opportunity: concrete physics / EE / coding / timeline concepts that benefit from original animation
        visual_opp = 0.88 if is_tech else (0.68 if any(k in low_txt for k in ("timeline", "step", "percent", "chart", "compare")) else 0.45)

        # 8. Context dependency (1.0 = completely self-contained, lower if starts with dangling connector)
        first_word = toks[0] if toks else ""
        dangling = first_word in {"and", "so", "but", "then", "it", "they", "that", "this"}
        context_indep = 0.55 if dangling else 0.9

        composite = round(
            0.16 * info_density
            + 0.12 * novelty
            + 0.14 * story_rel
            + 0.16 * curiosity
            + 0.14 * edu_val
            + 0.10 * emotional
            + 0.12 * visual_opp
            + 0.06 * context_indep,
            3,
        )

        if idx == 0:
            role = "hook"
        elif is_tech and visual_opp >= 0.8:
            role = "core_explanation"
        elif idx == len(windows) - 1:
            role = "payoff"
        elif curiosity >= 0.65:
            role = "short_candidate"
        else:
            role = "turning_point"

        snippet = text.strip()
        title = snippet[:58] + ("..." if len(snippet) > 58 else "")
        proposals.append(
            HighlightProposal(
                highlight_id=f"hl_{idx:02d}",
                start_sec=round(start_s, 3),
                end_sec=round(end_s, 3),
                duration_sec=round(dur, 3),
                role=role,
                title=title,
                transcript_excerpt=snippet,
                scores=HighlightScoreBreakdown(
                    information_density=round(info_density, 3),
                    novelty=round(novelty, 3),
                    story_relevance=round(story_rel, 3),
                    audience_curiosity=round(curiosity, 3),
                    educational_value=round(edu_val, 3),
                    emotional_intensity=round(emotional, 3),
                    visual_opportunity=round(visual_opp, 3),
                    context_dependency=round(context_indep, 3),
                    composite_score=composite,
                ),
                visual_opportunity_type=suggested_template,
                suggested_animation_domain=domain,
                rationale=(
                    f"Role={role}; domain={domain}; curiosity={curiosity:.2f}; "
                    f"visual_opportunity={visual_opp:.2f}; self_contained={context_indep:.2f}"
                ),
                kept=True,
            )
        )

    return proposals
