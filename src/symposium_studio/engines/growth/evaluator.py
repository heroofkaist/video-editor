"""Growth Objective Evaluator, Relevance Engine, and 11-Factor Content Performance Predictor."""

from __future__ import annotations

import re
import statistics
import uuid
from typing import Any

from symposium_studio.models.growth import (
    CREATOR_CORE_AUDIENCE,
    CandidateEditEvaluation,
    ContentExperimentRecord,
    CreativeDirectorGateReview,
    GrowthObjectiveProposal,
    PerformancePredictionScorecard,
    RelevanceAssessment,
)
from symposium_studio.models.media import HighlightProposal, MediaAnalysisBundle
from symposium_studio.models.research import ResearchReport, ViralContentItem


def _clamp01(v: float) -> float:
    return round(max(0.05, min(0.99, float(v))), 3)


def assess_topic_relevance(
    analysis: MediaAnalysisBundle,
    research_report: ResearchReport,
    research_items: list[ViralContentItem] | None = None,
) -> RelevanceAssessment:
    """Evaluate topic timeliness, cross-niche momentum, and evergreen audience relevance."""
    keywords = {k.lower() for t in analysis.topics for k in t.keywords if len(k) > 2}
    transcript_low = " ".join(s.text.lower() for s in analysis.transcript.segments)

    verified_findings = [f for f in research_report.findings if f.get("views") is not None]
    pool_items = list(research_items or [])
    matched_trends: list[dict[str, Any]] = []

    for f in research_report.findings:
        hay = f"{f.get('title', '')} {f.get('opening_hook', '')} {f.get('narrative_structure', '')}".lower()
        kw_hits = sum(1 for k in keywords if k in hay)
        ratio = float(f.get("audience_normalized_ratio") or 1.0)
        if kw_hits > 0 or ratio >= 2.0:
            matched_trends.append(
                {
                    "item_id": f.get("item_id"),
                    "category": f.get("category"),
                    "platform": f.get("platform"),
                    "title": f.get("title"),
                    "audience_normalized_ratio": f.get("audience_normalized_ratio"),
                    "views_per_day": f.get("views_per_day"),
                    "relevance_reason": (
                        f"Matched {kw_hits} topic keyword(s)"
                        if kw_hits > 0
                        else f"Cross-niche breakout ({ratio}x baseline) adaptable to explanation pacing"
                    ),
                }
            )

    if not matched_trends and pool_items:
        for it in pool_items[:5]:
            matched_trends.append(
                {
                    "item_id": it.item_id,
                    "category": it.category.value,
                    "platform": it.platform.value,
                    "title": it.title,
                    "audience_normalized_ratio": it.metrics.audience_normalized_ratio,
                    "views_per_day": it.metrics.views_per_day,
                    "relevance_reason": "Cross-niche reference format from research catalog",
                }
            )

    timely_cues = bool(
        re.search(r"\b(today|new|recent|2025|2026|breakthrough|benchmark|release|latest|bug|fail)\b", transcript_low)
    )
    is_tech_principle = any(t.is_technical for t in analysis.topics)
    if timely_cues and is_tech_principle:
        timeliness = "timely_anchor_to_evergreen"
    elif timely_cues:
        timeliness = "timely"
    else:
        timeliness = "evergreen"

    ratios = [float(f["audience_normalized_ratio"]) for f in verified_findings if f.get("audience_normalized_ratio")]
    median_ratio = statistics.median(ratios) if ratios else 1.0
    kw_density = min(1.0, len(matched_trends) / 6.0)
    momentum = _clamp01(0.42 + min(0.35, (median_ratio - 1.0) * 0.08) + kw_density * 0.22)

    kw_str = ", ".join(sorted(keywords)[:5]) or "first-principles mechanism"
    why_today = (
        f"Audiences across {len({m['category'] for m in matched_trends[:6]}) or 1} researched niche(s) engage strongly when "
        f"concrete anomaly demonstrations precede theory; anchoring [{kw_str}] in observable proof makes the concept immediately actionable."
    )
    evergreen_note = (
        "Core physical/computational derivation remains useful for students and engineers over years if stripped of ephemeral filler."
        if is_tech_principle
        else "Clear problem -> resolution storytelling retains shelf life beyond short news cycles."
    )
    limitations: list[str] = []
    if not verified_findings:
        limitations.append(
            "Zero live metriced external videos present during relevance check; momentum rests on baseline structural priors until live RSS discovery runs."
        )
    return RelevanceAssessment(
        topic_timeliness=timeliness,
        momentum_score=momentum,
        matched_trends=matched_trends[:6],
        why_viewer_cares_today=why_today,
        long_term_relevance_note=evergreen_note,
        limitations=limitations,
    )


def _prior_experiment_lift(
    strategy_name: str,
    editing_style: str,
    prior_experiments: list[ContentExperimentRecord],
) -> tuple[float, list[str]]:
    """Compute bounded empirical lift from creator's prior published experiments."""
    observed = [
        e for e in prior_experiments if e.ratio_vs_creator_median is not None and e.first_7d_views is not None
    ]
    if not observed:
        return 0.0, []

    matched = [
        e for e in observed if e.strategy_name == strategy_name or e.editing_style == editing_style
    ]
    if not matched:
        return 0.0, []

    ratios = [float(e.ratio_vs_creator_median) for e in matched if e.ratio_vs_creator_median is not None]
    avg_ratio = sum(ratios) / len(ratios)
    if avg_ratio >= 1.25:
        lift = round(min(0.10, (avg_ratio - 1.0) * 0.09), 3)
        note = f"Historical lift +{lift:.3f}: {len(matched)} prior experiment(s) averaged {avg_ratio:.2f}x creator median."
        return lift, [note]
    if avg_ratio < 0.85:
        penalty = round(max(-0.08, (avg_ratio - 1.0) * 0.12), 3)
        note = f"Historical caution {penalty:.3f}: {len(matched)} prior experiment(s) averaged {avg_ratio:.2f}x creator median."
        return penalty, [note]
    return 0.0, [f"Historical neutral baseline across {len(matched)} comparable experiment(s) ({avg_ratio:.2f}x median)."]


def _select_candidate_beats(
    hls: list[HighlightProposal],
    mode: str,
    uncertain_spans: list[tuple[float, float]],
    target_max_sec: float,
) -> list[HighlightProposal]:
    if not hls:
        return []
    curiosity_sorted = sorted(
        hls,
        key=lambda h: (h.scores.audience_curiosity + h.scores.emotional_intensity, h.scores.composite_score),
        reverse=True,
    )
    visual_sorted = sorted(
        hls,
        key=lambda h: (h.scores.educational_value + h.scores.visual_opportunity, h.scores.composite_score),
        reverse=True,
    )

    def _is_clean(h: HighlightProposal) -> bool:
        return not any(u_end > h.start_sec and u_start < h.end_sec for u_start, u_end in uncertain_spans)

    if mode == "hook_first":
        seed = [curiosity_sorted[0]]
        pool = sorted(hls, key=lambda h: h.scores.composite_score, reverse=True)
        cap = target_max_sec
    elif mode == "mechanism_proof":
        clean_first = [h for h in visual_sorted if _is_clean(h)] or visual_sorted
        seed = [clean_first[0]]
        pool = clean_first[1:] + sorted(hls, key=lambda h: h.start_sec)
        cap = min(95.0, target_max_sec + 10.0)
    else:  # tight_viral
        hook_pick = max(hls[: min(3, len(hls))], key=lambda h: h.scores.audience_curiosity)
        seed = [hook_pick]
        pool = [h for h in sorted(hls, key=lambda h: h.scores.composite_score, reverse=True) if h.duration_sec <= 28.0] or list(hls)
        cap = min(62.0, target_max_sec)

    chosen = list(seed)
    seen = {h.highlight_id for h in chosen}
    total = sum(h.duration_sec for h in chosen)
    for h in pool:
        if h.highlight_id in seen:
            continue
        if total + h.duration_sec > cap and len(chosen) >= 2:
            continue
        chosen.append(h)
        seen.add(h.highlight_id)
        total += h.duration_sec
        if len(chosen) >= 6:
            break
    return chosen


def evaluate_candidate_edits(
    analysis: MediaAnalysisBundle,
    research_report: ResearchReport,
    relevance: RelevanceAssessment,
    style_name: str,
    prior_experiments: list[ContentExperimentRecord] | None = None,
    target_max_sec: float = 90.0,
) -> list[CandidateEditEvaluation]:
    """Generate 3 distinct candidate edit versions, score across all 11 growth dimensions, and rank them."""
    hls = analysis.highlights
    segs = analysis.transcript.segments
    unc_ids = set(analysis.transcript.uncertain_segment_ids)
    uncertain_spans = [
        (s.start_sec, s.end_sec)
        for s in segs
        if s.segment_id in unc_ids or s.low_confidence
    ]
    long_pause_count = sum(
        1 for a, b in zip(segs, segs[1:]) if (b.start_sec - a.end_sec) >= 1.0
    )
    has_tech = any(t.is_technical for t in analysis.topics)
    has_faces = any(vs.faces for vs in analysis.visual_samples)
    verified_research_n = sum(1 for f in research_report.findings if f.get("views") is not None)
    prior_list = list(prior_experiments or [])

    blueprints = [
        ("Hook-First Curiosity Paradox", style_name or "educational_explainer", "hook_first"),
        ("Mechanism & Visual Proof", "educational_explainer", "mechanism_proof"),
        ("Tight Viral Story Compression", "high_energy_youtube", "tight_viral"),
    ]

    # Base empirical uncertainty shrinks only as live research + historical creator experiments grow
    observed_exp_n = sum(1 for e in prior_list if e.first_7d_views is not None)
    base_uncertainty = round(
        max(0.06, 0.24 - min(0.09, verified_research_n * 0.005) - min(0.09, observed_exp_n * 0.015)),
        3,
    )

    candidates: list[CandidateEditEvaluation] = []
    for idx, (s_name, st_profile, mode) in enumerate(blueprints):
        beats = _select_candidate_beats(hls, mode, uncertain_spans, target_max_sec=target_max_sec)
        dur = round(sum(b.duration_sec for b in beats), 2) if beats else round(min(target_max_sec, analysis.probe.duration_sec), 2)
        first = beats[0] if beats else None

        avg_info = sum(b.scores.information_density for b in beats) / max(1, len(beats)) if beats else 0.55
        avg_vis = sum(b.scores.visual_opportunity for b in beats) / max(1, len(beats)) if beats else 0.55
        avg_edu = sum(b.scores.educational_value for b in beats) / max(1, len(beats)) if beats else 0.55
        avg_emo = sum(b.scores.emotional_intensity for b in beats) / max(1, len(beats)) if beats else 0.55

        hook_boost = 0.09 if mode == "hook_first" else (0.07 if mode == "tight_viral" else 0.02)
        clarity_boost = 0.08 if mode == "mechanism_proof" else (0.04 if mode == "hook_first" else 0.01)
        compression_bonus = 0.07 if mode == "tight_viral" else (0.04 if mode == "hook_first" else 0.0)
        unc_overlap_cnt = sum(
            1 for b in beats if any(ue > b.start_sec and us < b.end_sec for us, ue in uncertain_spans)
        )

        sc = PerformancePredictionScorecard(
            hook_strength=_clamp01((first.scores.audience_curiosity if first else 0.55) + hook_boost),
            topic_relevance=_clamp01(relevance.momentum_score),
            clarity=_clamp01(avg_edu + clarity_boost - unc_overlap_cnt * 0.06),
            information_density=_clamp01(avg_info + (0.04 if has_tech else 0.0)),
            curiosity=_clamp01((first.scores.audience_curiosity if first else 0.55) + (0.06 if "?" in (first.transcript_excerpt if first else "") else 0.0)),
            emotional_engagement=_clamp01(avg_emo + (0.06 if mode != "mechanism_proof" else 0.02)),
            narrative_progression=_clamp01(0.68 + (0.12 if len(beats) >= 3 else 0.04) - max(0, long_pause_count - 4) * 0.02 + compression_bonus * 0.5),
            visual_quality=_clamp01(avg_vis * 0.7 + (0.22 if has_faces else 0.12) + 0.08),
            audience_compatibility=_clamp01(0.76 + (0.12 if has_tech else 0.04) + (0.05 if mode == "hook_first" else 0.02)),
            shareability=_clamp01((relevance.momentum_score * 0.45) + (first.scores.audience_curiosity if first else 0.55) * 0.35 + compression_bonus),
            estimated_long_term_relevance=_clamp01(0.78 if relevance.topic_timeliness != "timely" else 0.58 + avg_edu * 0.12),
        )

        weighted = (
            sc.hook_strength * 0.14
            + sc.topic_relevance * 0.11
            + sc.clarity * 0.11
            + sc.information_density * 0.08
            + sc.curiosity * 0.11
            + sc.emotional_engagement * 0.07
            + sc.narrative_progression * 0.10
            + sc.visual_quality * 0.08
            + sc.audience_compatibility * 0.08
            + sc.shareability * 0.06
            + sc.estimated_long_term_relevance * 0.06
        )
        lift, lift_notes = _prior_experiment_lift(s_name, st_profile, prior_list)
        comp = _clamp01(weighted + lift)
        low_ci = round(max(0.05, comp - base_uncertainty), 3)
        high_ci = round(min(0.99, comp + base_uncertainty), 3)

        rationale = [
            f"Hook strength {sc.hook_strength:.2f} anchored at {first.highlight_id if first else 'opening frame'} ({dur:.1f}s edit).",
            f"Topic relevance {sc.topic_relevance:.2f} ({relevance.topic_timeliness}); {len(beats)} beats avoiding {unc_overlap_cnt} uncertain ASR overlap(s).",
            *lift_notes,
        ]
        if verified_research_n == 0:
            rationale.append("Wider prediction uncertainty (+/- %.2f) due to zero live metriced research items." % base_uncertainty)
        if observed_exp_n == 0:
            rationale.append("No published creator experiment baseline yet; prediction will tighten as actual 7d metrics arrive.")

        candidates.append(
            CandidateEditEvaluation(
                candidate_id=f"cand_{idx + 1}_{uuid.uuid4().hex[:5]}",
                strategy_name=s_name,
                editing_style=st_profile,
                beat_highlight_ids=[b.highlight_id for b in beats],
                target_duration_sec=dur,
                scorecard=sc,
                composite_score=comp,
                historical_lift_applied=lift,
                confidence_interval={"low": low_ci, "high": high_ci, "uncertainty": base_uncertainty},
                evidence_rationale=rationale,
                selected=False,
            )
        )

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    if candidates:
        candidates[0].selected = True
    return candidates


def build_growth_objective_proposal(
    analysis: MediaAnalysisBundle,
    relevance: RelevanceAssessment,
    best_candidate: CandidateEditEvaluation,
    chosen_template: str,
) -> GrowthObjectiveProposal:
    """Build Section-7 Growth Objective Evaluator brief aligned with creator audience & funnel goals."""
    tech_domains = sorted({t.domain for t in analysis.topics if t.is_technical})
    keywords = list(dict.fromkeys(k for t in analysis.topics for k in t.keywords))[:6]
    segs = analysis.transcript.segments
    pauses = [round(b.start_sec - a.end_sec, 2) for a, b in zip(segs, segs[1:]) if (b.start_sec - a.end_sec) >= 1.0]
    unc_n = len(analysis.transcript.uncertain_segment_ids)

    audiences = list(CREATOR_CORE_AUDIENCE)
    if "broader_curious_learners" not in audiences:
        audiences.append("curious_ambitious_general_audience")

    first_hl = next((h for h in analysis.highlights if h.highlight_id in best_candidate.beat_highlight_ids), None)
    hook_excerpt = (first_hl.transcript_excerpt[:140] if first_hl else "Observable demonstration hook")

    risks: list[str] = []
    if pauses:
        risks.append(f"{len(pauses)} source dead-air pause(s) >= 1.0s (max {max(pauses):.2f}s) threaten first-30s retention unless trimmed.")
    if unc_n:
        risks.append(f"{unc_n} low-confidence ASR span(s) require manual caption verification or clean alternative beat selection.")
    if best_candidate.target_duration_sec > 85.0:
        risks.append("Longer assembly (>85s) raises mid-roll drop risk if mechanism visual lags exposition.")
    if not risks:
        risks.append("Ensure opening hook visual payoff lands inside 0.0-3.0s without introductory channel boilerplate.")

    return GrowthObjectiveProposal(
        target_audience=audiences,
        accessible_entry_hook=f"Concrete paradox before formalism ({', '.join(tech_domains) or 'applied STEM'}) so broader curious viewers follow before technical depth.",
        content_hypothesis=(
            f"Applying '{best_candidate.strategy_name}' with procedural `{chosen_template}` simulation on [{', '.join(keywords[:4]) or 'core topic'}] "
            f"will raise first-7-day qualified views toward 1.5x creator median while converting deep learners to Telegram notes."
        ),
        relevance_assessment=relevance,
        expected_viewer_value=(
            f"Viewers gain intuitive mechanism clarity + verifiable mental model of {', '.join(keywords[:3]) or 'the system'} "
            "plus reproducible reference parameters via Telegram."
        ),
        selected_hook_strategy=f"{best_candidate.strategy_name}: open on `{first_hl.highlight_id if first_hl else 'seg_0'}` — \"{hook_excerpt}\"",
        retention_risk_analysis=risks,
        distribution_strategy={
            "short_form_discovery": "Deploy 3 non-overlapping 9:16 Shorts (Hook, Mechanism, Payoff) across YouTube Shorts, TikTok, Instagram Reels with safe-zone captions.",
            "youtube_long": "Publish cohesive 16:9 YouTube deep dive anchored by chapters, original visualization, and clean thumbnail contrast.",
            "threads_bridge": "Post natural-voice observation & open technical question (<=280 chars) linking the Telegram discussion note.",
            "telegram_community": "Deliver technical deep-dive breakdown, benchmark parameters, poll, and tagged funnel tracking.",
        },
        measurement_plan={
            "targets": {
                "initial": "Technically verified, engaging multi-platform assets with authorized measurement capture.",
                "intermediate": "1.5x (+50%) improvement in first-7-day qualified views vs creator's recent comparable platform median.",
                "stretch": "2.0x creator's historical median performance sustained across comparable sample sizes.",
            },
            "tracked_metrics_by_cohort": [
                "qualified_organic_views_7d",
                "authorized_ctr_and_retention",
                "avg_view_duration_sec_and_watch_time",
                "shares_and_meaningful_comments",
                "new_subscribers_or_followers",
                "telegram_funnel_clicks_and_joins",
            ],
            "platform_isolation_policy": "Evaluate YouTube long-form, Shorts, TikTok, Instagram Reels, and Threads separately against same-platform medians.",
        },
    )


def creative_director_gate_review(
    proposal: GrowthObjectiveProposal,
    candidates: list[CandidateEditEvaluation],
) -> CreativeDirectorGateReview:
    """Creative Director pre-approval audit against all Section-7 Growth Objective criteria."""
    best = candidates[0] if candidates else None
    checks: list[str] = []
    ok = True

    required_pairs = [
        ("target_audience", bool(proposal.target_audience)),
        ("content_hypothesis", bool(proposal.content_hypothesis.strip())),
        ("relevance_assessment", bool(proposal.relevance_assessment.why_viewer_cares_today.strip())),
        ("expected_viewer_value", bool(proposal.expected_viewer_value.strip())),
        ("selected_hook_strategy", bool(proposal.selected_hook_strategy.strip())),
        ("retention_risk_analysis", len(proposal.retention_risk_analysis) >= 1),
        ("distribution_strategy", len(proposal.distribution_strategy) >= 3),
        ("measurement_plan", "targets" in proposal.measurement_plan),
    ]
    for label, passed in required_pairs:
        checks.append(f"{label}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            ok = False

    if not best or not best.selected:
        ok = False
        checks.append("candidate_rank_1_selected: FAIL")
    else:
        checks.append(
            f"candidate_rank_1_selected: PASS ({best.strategy_name}, score={best.composite_score:.3f} +/- {best.confidence_interval.get('uncertainty', 0.2):.2f})"
        )

    return CreativeDirectorGateReview(
        approved=ok,
        checked_criteria=checks,
        selected_candidate_id=best.candidate_id if best else "none",
        scorecard_rank=1,
        revision_notes=(
            f"Approved candidate `{best.strategy_name}` after evaluating {len(candidates)} versions against creator audience & growth targets."
            if ok and best
            else "Candidate blocked pending complete Growth Objective criteria."
        ),
    )
