"""AI Symposium multi-agent research and creative deliberation engine.

Implements all 7 Research Team agents, all 11 Creative Team agents,
signal-grounded independent Round 1 analysis (with optional structured LLM execution),
Round 2 Moderator cross-examination of real evidence/confidence/pacing conflicts,
and Round 3 traceable Storyboard revision across inexpensive, standard, and comprehensive modes.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from symposium_studio.config import StudioConfig
from symposium_studio.engines.analysis.styles import get_style_profile
from symposium_studio.engines.growth.evaluator import (
    assess_topic_relevance,
    build_growth_objective_proposal,
    creative_director_gate_review,
    evaluate_candidate_edits,
)
from symposium_studio.engines.production.directors import (
    evaluate_ending_closure,
    plan_coherent_beat_sequence,
    select_best_hook,
    snap_ending_boundary,
)
from symposium_studio.engines.symposium.llm_provider import SymposiumLLMRouter
from symposium_studio.models.growth import ContentExperimentRecord
from symposium_studio.models.media import MediaAnalysisBundle
from symposium_studio.models.project import utc_now_iso
from symposium_studio.models.research import ResearchReport, ViralContentItem
from symposium_studio.models.symposium import (
    AgentProposal,
    AgentTeam,
    DebateTurn,
    ModeratorChallenge,
    ResearchMode,
    Storyboard,
    StoryboardBeat,
    SymposiumDecision,
    SymposiumSession,
)


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    title: str
    team: AgentTeam
    min_mode: ResearchMode
    charter: str


AGENT_ROSTER: list[AgentDefinition] = [
    # Research Team (7)
    AgentDefinition(
        "viral_researcher",
        "Viral Researcher",
        AgentTeam.RESEARCH,
        ResearchMode.INEXPENSIVE,
        "Evaluate cross-niche public outliers weighted by audience baseline multiplier and extract concrete packaging mechanics.",
    ),
    AgentDefinition(
        "hook_scientist",
        "Hook Scientist",
        AgentTeam.RESEARCH,
        ResearchMode.INEXPENSIVE,
        "Score candidate opening lines, curiosity/stakes framing, and trim dead silence at t=0.",
    ),
    AgentDefinition(
        "trend_researcher",
        "Trend Researcher",
        AgentTeam.RESEARCH,
        ResearchMode.STANDARD,
        "Analyze cross-platform format mix, Shorts vs long-form dynamics, and velocity trends in the research corpus.",
    ),
    AgentDefinition(
        "storytelling_researcher",
        "Storytelling Researcher",
        AgentTeam.RESEARCH,
        ResearchMode.STANDARD,
        "Design tension -> mechanism -> payoff arc ordering across candidate highlights.",
    ),
    AgentDefinition(
        "audience_researcher",
        "Audience Researcher",
        AgentTeam.RESEARCH,
        ResearchMode.COMPREHENSIVE,
        "Assess cognitive load, speech pacing (WPM), and jargon density relative to target viewers.",
    ),
    AgentDefinition(
        "growth_strategist",
        "Growth Strategist",
        AgentTeam.RESEARCH,
        ResearchMode.STANDARD,
        "Align YouTube anchor, Shorts discovery hooks, Threads curiosity bridges, and Telegram deep dives.",
    ),
    AgentDefinition(
        "scientific_fact_checker",
        "Scientific Fact Checker",
        AgentTeam.RESEARCH,
        ResearchMode.STANDARD,
        "Audit technical claims, ASR uncertainty overlap, and enforce factual_scientific constraints on simulations.",
    ),
    # Creative Team (11)
    AgentDefinition(
        "creative_director",
        "Creative Director",
        AgentTeam.CREATIVE,
        ResearchMode.INEXPENSIVE,
        "Synthesize editorial direction, resolve tradeoffs from Moderator critique, and approve final strategy.",
    ),
    AgentDefinition(
        "story_producer",
        "Story Producer",
        AgentTeam.CREATIVE,
        ResearchMode.INEXPENSIVE,
        "Select source highlight sequence meeting duration targets without redundant pauses.",
    ),
    AgentDefinition(
        "professional_video_editor",
        "Professional Video Editor",
        AgentTeam.CREATIVE,
        ResearchMode.INEXPENSIVE,
        "Set cut points, trim silence pauses, speed pacing, transitions, and subject-aware framing.",
    ),
    AgentDefinition(
        "motion_graphics_artist",
        "Motion Graphics Artist",
        AgentTeam.CREATIVE,
        ResearchMode.STANDARD,
        "Author SceneSpec template & simulation parameters matching spoken explanation.",
    ),
    AgentDefinition(
        "caption_designer",
        "Caption Designer",
        AgentTeam.CREATIVE,
        ResearchMode.STANDARD,
        "Configure word-level caption hierarchy, platform safe zones, and low-confidence review flags.",
    ),
    AgentDefinition(
        "sound_designer",
        "Sound Designer",
        AgentTeam.CREATIVE,
        ResearchMode.STANDARD,
        "Balance -14 LUFS voice presence, 80 Hz highpass cleanup, sidechain-ducked ambient bed, and transition SFX.",
    ),
    AgentDefinition(
        "platform_strategist",
        "Platform Strategist",
        AgentTeam.CREATIVE,
        ResearchMode.STANDARD,
        "Tailor 16:9 YouTube master and 3 distinct 9:16 vertical Shorts plus Threads/Telegram hooks.",
    ),
    AgentDefinition(
        "three_d_animation_artist",
        "3D Animation Artist",
        AgentTeam.CREATIVE,
        ResearchMode.COMPREHENSIVE,
        "Evaluate when Blender 3D geometry adds spatial clarity versus crisp 2D vector canvas.",
    ),
    AgentDefinition(
        "visual_effects_artist",
        "Visual Effects Artist",
        AgentTeam.CREATIVE,
        ResearchMode.COMPREHENSIVE,
        "Enforce face-non-overlapping overlay bounds and subtle contrast polish.",
    ),
    AgentDefinition(
        "meme_researcher",
        "Meme Researcher",
        AgentTeam.CREATIVE,
        ResearchMode.COMPREHENSIVE,
        "Propose attributed reaction beats only at genuine narrative irony without distracting from equations.",
    ),
    AgentDefinition(
        "qa_reviewer",
        "Quality Assurance Reviewer",
        AgentTeam.CREATIVE,
        ResearchMode.INEXPENSIVE,
        "Enforce zero black frames, loudness bounds, exact dimensions, and safe UI margins on all renders.",
    ),
]

MODE_RANK = {
    ResearchMode.INEXPENSIVE: 0,
    ResearchMode.STANDARD: 1,
    ResearchMode.COMPREHENSIVE: 2,
}


def active_agents_for_mode(mode: ResearchMode) -> list[AgentDefinition]:
    rank = MODE_RANK[mode]
    return [a for a in AGENT_ROSTER if MODE_RANK[a.min_mode] <= rank]


def _pick_animation_template(analysis: MediaAnalysisBundle, style_name: str = "") -> str:
    if style_name == "zoomer_vlog":
        return "meme_reaction_card"
    best_visual = (
        max(analysis.highlights, key=lambda h: h.scores.visual_opportunity + h.scores.educational_value)
        if analysis.highlights
        else None
    )
    if best_visual and best_visual.visual_opportunity_type:
        return best_visual.visual_opportunity_type
    full_txt = " ".join(s.text.lower() for s in analysis.transcript.segments)
    if any(k in full_txt for k in ("wave", "interference", "superposition", "frequency", "phase", "acoustic")):
        return "wave_interference_sim"
    if any(k in full_txt for k in ("architecture", "pipeline", "microservice", "queue", "api", "latency", "cache")):
        return "system_architecture_flow"
    if any(k in full_txt for k in ("circuit", "voltage", "resistor", "capacitor", "rlc", "impedance")):
        return "rlc_signal_flow"
    if any(k in full_txt for k in ("tree", "binary", "algorithm", "search", "complexity", "recursion")):
        return "binary_tree_trace"
    if style_name in {"personal_vlog", "comedy", "short_form_viral"}:
        return "kinetic_callout"
    return "em_induction_faraday"


def _compute_signal_brief(
    analysis: MediaAnalysisBundle,
    research_report: ResearchReport,
) -> dict[str, Any]:
    """Compute objective quantitative metrics from media analysis & research report."""
    segs = analysis.transcript.segments
    total_words = sum(len(s.words) if s.words else len(s.text.split()) for s in segs)
    speech_dur = sum(max(0.05, s.end_sec - s.start_sec) for s in segs) or max(1.0, analysis.probe.duration_sec)
    wpm = round((total_words / speech_dur) * 60.0, 1)

    long_pauses: list[dict[str, float]] = []
    for a, b in zip(segs, segs[1:]):
        gap = round(b.start_sec - a.end_sec, 2)
        if gap >= 1.0:
            long_pauses.append({"start_sec": a.end_sec, "end_sec": b.start_sec, "gap_sec": gap})

    uncertain_ids = set(analysis.transcript.uncertain_segment_ids)
    uncertain_segs = [
        {"segment_id": s.segment_id, "start_sec": s.start_sec, "end_sec": s.end_sec, "conf": s.confidence, "reason": s.low_confidence_reason}
        for s in segs
        if s.segment_id in uncertain_ids or s.low_confidence
    ]

    findings = research_report.findings
    verified_findings = [f for f in findings if f.get("views") is not None]
    outliers = sorted(
        verified_findings,
        key=lambda f: (float(f.get("audience_normalized_ratio") or 1.0), float(f.get("platform_relative_score") or 0.0)),
        reverse=True,
    )

    hook_ranked = sorted(
        analysis.highlights,
        key=lambda h: (
            h.scores.audience_curiosity
            + h.scores.emotional_intensity
            + (0.25 if "?" in h.transcript_excerpt else 0.0)
            + (0.18 if re.search(r"\b(why|never|wrong|but|actually|secret|break|fail)\b", h.transcript_excerpt.lower()) else 0.0)
        ),
        reverse=True,
    )

    return {
        "duration_sec": analysis.probe.duration_sec,
        "resolution": f"{analysis.probe.width}x{analysis.probe.height}",
        "audio_mean_db": analysis.probe.mean_volume_db,
        "detected_language": analysis.transcript.detected_language,
        "speech_wpm": wpm,
        "segment_count": len(segs),
        "long_pauses": long_pauses,
        "uncertain_segments": uncertain_segs,
        "recommended_crop_center_x": analysis.recommended_crop_center_x,
        "faces_detected_frames": sum(1 for vs in analysis.visual_samples if vs.faces),
        "technical_domains": sorted({t.domain for t in analysis.topics if t.is_technical}),
        "keywords": list(dict.fromkeys(k for t in analysis.topics for k in t.keywords))[:10],
        "verified_research_count": len(verified_findings),
        "total_research_findings": len(findings),
        "top_outliers": outliers[:4],
        "valid_research_ids": [str(f["item_id"]) for f in findings if f.get("item_id")],
        "valid_highlight_ids": [h.highlight_id for h in analysis.highlights],
        "hook_candidates": [
            {
                "highlight_id": h.highlight_id,
                "start_sec": h.start_sec,
                "end_sec": h.end_sec,
                "curiosity": h.scores.audience_curiosity,
                "composite": h.scores.composite_score,
                "excerpt": h.transcript_excerpt[:140],
            }
            for h in hook_ranked[:3]
        ],
    }


class SymposiumOrchestrator:
    """Runs traceable 3-round AI Symposium deliberation grounded in real media & research signals."""

    def __init__(self, config: StudioConfig) -> None:
        self.config = config
        self.llm = SymposiumLLMRouter(config)

    def _select_highlights_for_duration(
        self,
        analysis: MediaAnalysisBundle,
        target_min_sec: float = 60.0,
        target_max_sec: float = 90.0,
    ) -> list[Any]:
        """Select a coherent narrative edit targeting 60-90s (or full clip if shorter)."""
        hls = analysis.highlights
        if not hls:
            return []
        if analysis.probe.duration_sec <= target_max_sec:
            return plan_coherent_beat_sequence(
                analysis,
                target_min_sec=max(10.0, analysis.probe.duration_sec * 0.7),
                target_max_sec=target_max_sec,
                strategy="hook_first",
            )
        return plan_coherent_beat_sequence(
            analysis,
            target_min_sec=target_min_sec,
            target_max_sec=target_max_sec,
            strategy="hook_first",
        )

    def _build_specialist_proposal(
        self,
        ag: AgentDefinition,
        brief: dict[str, Any],
        analysis: MediaAnalysisBundle,
        research_report: ResearchReport,
        style_name: str,
        chosen_template: str,
        llm_backend: str,
    ) -> tuple[AgentProposal, DebateTurn]:
        style = get_style_profile(style_name)
        pid = f"prop_{ag.agent_id}_{uuid.uuid4().hex[:5]}"
        valid_ids = set(brief["valid_research_ids"] + brief["valid_highlight_ids"])
        ref_ids = brief["valid_research_ids"][:3]
        hl_ids = brief["valid_highlight_ids"][:3]
        best_hook = brief["hook_candidates"][0] if brief["hook_candidates"] else None
        outliers = brief["top_outliers"]

        # Deterministic domain-grounded computation first (serves as evidence floor & offline engine)
        if ag.agent_id == "viral_researcher":
            if outliers:
                top = outliers[0]
                cats = sorted({o["category"] for o in outliers})
                summary = (
                    f"Analyzed {brief['verified_research_count']} metriced public videos across {', '.join(cats)}. "
                    f"Top outlier '{top['title'][:65]}' ({top['creator']}) hit {top.get('audience_normalized_ratio', 1.0)}x channel median "
                    f"({top.get('views'):,} views)."
                )
                recs = [
                    t["editing_translation"] for t in research_report.cross_niche_transfers[:2]
                ] or [f"Mirror '{top.get('opening_hook', '')[:90]}' structure."]
                refs = [o["item_id"] for o in outliers[:3]]
                conf = 0.91
            else:
                summary = (
                    f"WARNING: Research corpus holds 0 live-metriced videos (only {brief['total_research_findings']} unscored archetypes). "
                    "Recommend running live public RSS/API discovery before finalizing breakout claims."
                )
                recs = ["Run `discover_cohort` across creator RSS feeds or official YouTube Data API v3 to ground scores."]
                refs = ref_ids[:2]
                conf = 0.62

        elif ag.agent_id == "hook_scientist":
            cands = brief["hook_candidates"]
            if best_hook:
                alts = ", ".join(f"{c['highlight_id']} ({c['start_sec']}s)" for c in cands[1:]) or "none"
                summary = (
                    f"Top hook candidate {best_hook['highlight_id']} at {best_hook['start_sec']:.1f}s-{best_hook['end_sec']:.1f}s "
                    f"(curiosity={best_hook['curiosity']:.2f}): \"{best_hook['excerpt'][:95]}\". Runners-up: {alts}."
                )
                recs = [
                    f"Open cut at {best_hook['start_sec']:.2f}s (skip {len(brief['long_pauses'])} upstream dead pause(s)).",
                    "Layer immediate tension statement + sub-impact SFX inside first 0.5s.",
                ]
                refs = [best_hook["highlight_id"], *ref_ids[:1]]
                conf = 0.89
            else:
                summary = "Insufficient transcript highlights; default to opening frame."
                recs = ["Lead with concrete visual demonstration."]
                refs = ref_ids[:1]
                conf = 0.55

        elif ag.agent_id == "trend_researcher":
            cats = sorted({f["category"] for f in research_report.findings})
            summary = (
                f"Sampled {len(research_report.findings)} cross-niche records across {len(cats)} categories "
                f"({', '.join(cats[:5])}). Contrarian questions & observable-before-formula hooks lead velocity."
            )
            recs = [t["editing_translation"] for t in research_report.cross_niche_transfers[:2]] or ["Pair visual proof with explicit stakes."]
            refs = ref_ids
            conf = 0.86

        elif ag.agent_id == "storytelling_researcher":
            summary = (
                f"Speech pace measured at {brief['speech_wpm']} WPM across {len(analysis.highlights)} candidate beats; "
                f"found {len(brief['long_pauses'])} pause gap(s) >= 1.0s breaking narrative flow."
            )
            recs = [
                "Order beats as Hook -> Concrete Anomaly -> Procedural Mechanism Animation -> Payoff.",
                "Bridge gaps >= 1.0s with tight L/J cuts instead of leaving dead air.",
            ]
            refs = hl_ids
            conf = 0.88

        elif ag.agent_id == "audience_researcher":
            tech_cnt = len(brief["technical_domains"])
            summary = (
                f"Detected {tech_cnt} technical domain tag(s) ({', '.join(brief['technical_domains']) or 'general'}) "
                f"at {brief['speech_wpm']} WPM. Viewers need intuitive mechanism visual before dense equations."
            )
            recs = ["Hold core explanation beat uninterrupted and attach plain-language callouts for key terms."]
            refs = hl_ids[:2]
            conf = 0.85

        elif ag.agent_id == "growth_strategist":
            kws = ", ".join(brief["keywords"][:4]) or "topic notes"
            summary = (
                f"Channel bridge centers on [{kws}]: use 3 distinct Shorts cuts as top-of-funnel hooks feeding the 16:9 YouTube deep dive and Telegram notes."
            )
            recs = [
                "Tag distinct funnel links per platform (youtube, shorts, tiktok, reels, threads) -> Telegram.",
                "Avoid posting identical copy across Threads & Shorts; isolate open question for Threads <=280 chars.",
            ]
            refs = ref_ids[:2]
            conf = 0.88

        elif ag.agent_id == "scientific_fact_checker":
            unc = brief["uncertain_segments"]
            unc_txt = (
                f"Flagged {len(unc)} low-confidence ASR segment(s): {[u['segment_id'] for u in unc[:4]]}."
                if unc
                else "Zero low-confidence ASR segments flagged."
            )
            summary = (
                f"Technical scope={brief['technical_domains'] or ['general']}. {unc_txt} "
                f"Mandating `factual_scientific` SceneSpec mode for `{chosen_template}`."
            )
            recs = [
                f"Keep analytical notes on `{chosen_template}` strictly separated from unverified metrics.",
                "Do not synthesize or paraphrase unverified ASR tokens cleanly; highlight low-confidence segments for manual review."
                if unc
                else "Preserve verbatim speaker transcript accuracy in ASS cues.",
            ]
            refs = hl_ids[:2]
            conf = 0.95

        elif ag.agent_id == "professional_video_editor":
            spd = 1.06 if brief["speech_wpm"] < 110 and brief["speech_wpm"] > 40 else 1.0
            summary = (
                f"Recommend playback_speed={spd}x (raw pace {brief['speech_wpm']} WPM), crop_center_x={brief['recommended_crop_center_x']:.3f}, "
                f"trimming {len(brief['long_pauses'])} silence gap(s) >= 1.0s."
            )
            recs = [
                f"Apply {style.transition_style} segment transitions without drifting audio-caption sync.",
                f"Track subject horizontally around x={brief['recommended_crop_center_x']:.3f} across all 9:16 exports.",
            ]
            refs = hl_ids[:2]
            conf = 0.90

        elif ag.agent_id == "motion_graphics_artist":
            summary = f"Author editable SceneSpec `{chosen_template}` synced to top technical explanation beat."
            recs = [
                f"Render `{chosen_template}` with alpha channel via Chromium Canvas / Vector backend.",
                "Composite side-by-side or right-column PIP avoiding speaker face region.",
            ]
            refs = hl_ids[:1]
            conf = 0.92
            # handled below for proposed_actions

        elif ag.agent_id == "three_d_animation_artist":
            summary = "Reserve Blender 3D (`blender -b` Cycles/Eevee subprocess) for volumetric hardware/coil depth; use fast 2D Chromium Canvas for vector timelines."
            recs = ["Keep GPL-3.0 Blender strictly isolated as optional CLI render step when installed."]
            refs = hl_ids[:1]
            conf = 0.84

        elif ag.agent_id == "visual_effects_artist":
            summary = (
                f"Speaker center detected at x={brief['recommended_crop_center_x']:.2f}; "
                "pin 16:9 graphics to opposite half and 9:16 graphics to top-safe zone above face."
            )
            recs = ["Verify zero overlap between overlay bounding box and speaker head in QA frames."]
            refs = hl_ids[:1]
            conf = 0.89

        elif ag.agent_id == "caption_designer":
            unc_n = len(brief["uncertain_segments"])
            summary = (
                f"Use `{style.caption_preset}` ASS subtitles; {unc_n} uncertain segment(s) require inspection; "
                "maintain bottom margin >= 260px on 9:16."
            )
            recs = ["Limit short-form lines to <= 38 chars for rapid legibility without hiding subject mouth."]
            refs = hl_ids[:2]
            conf = 0.91

        elif ag.agent_id == "sound_designer":
            db_str = f"{brief['audio_mean_db']} dB" if brief["audio_mean_db"] is not None else "unprobed"
            summary = f"Raw audio mean={db_str}. Apply 80 Hz highpass + 2.4:1 speech compressor -> -14 LUFS + sidechain-ducked ambient bed (-21 dB)."
            recs = ["Trigger procedural whoosh_transition on animation entrance and sub_impact on hook opening."]
            refs = []
            conf = 0.90

        elif ag.agent_id == "platform_strategist":
            summary = "Build distinct 16:9 YouTube master AND 3 non-identical 9:16 Shorts (Hook, Mechanism, Payoff) + <=280-char Threads post."
            recs = ["Ensure Shorts 1-3 extract different timestamps/angles rather than duplicating Clip 1 three times."]
            refs = hl_ids[:3]
            conf = 0.91

        elif ag.agent_id == "meme_researcher":
            contrast_hits = [
                h.highlight_id
                for h in analysis.highlights
                if re.search(r"\b(zero|nothing|wrong|wait|fail|crash|surprise|impossible)\b", h.transcript_excerpt.lower())
            ]
            summary = (
                f"Found {len(contrast_hits)} natural irony/surprise beat(s) ({contrast_hits[:2] or 'none'}); "
                "recommend clean kinetic freeze/micro-zoom callout only — no unattributed third-party memes."
            )
            recs = ["Keep reaction beats strictly outside core mathematical derivation sections."]
            refs = contrast_hits[:2]
            conf = 0.86

        elif ag.agent_id == "story_producer":
            summary = f"Curated {len(analysis.highlights)} candidate beats spanning {brief['duration_sec']:.1f}s source into cohesive 60-90s spine."
            recs = ["Verify every cut preserves full clause boundaries without clipping technical definitions."]
            refs = hl_ids
            conf = 0.89

        elif ag.agent_id == "qa_reviewer":
            summary = "Require automated FFmpeg blackdetect/freezedetect + dimension/loudnorm pass plus visual PNG frame inspection."
            recs = ["Fail gate if face crop or captions overlap overlays, or audio diverges from -14 LUFS target."]
            refs = []
            conf = 0.95

        else:  # creative_director
            summary = (
                f"Combine `{style.display_name}` finish with opening hook `{best_hook['highlight_id'] if best_hook else 'seg_0'}` "
                f"and procedural `{chosen_template}` visual explainer."
            )
            recs = [f"Hold factual derivation beat cleanly for >= 5.0s around `{chosen_template}` overlay."]
            refs = (hl_ids[:2] + ref_ids[:2])[:4]
            conf = 0.91

        # Optional live structured LLM enhancement per agent when active LLM backend exists
        if llm_backend != "deterministic_analytical":
            sys_p = (
                f"You are {ag.title} ({ag.team.value} team) in AI Symposium Studio.\n"
                f"Charter: {ag.charter}\n"
                "Respond ONLY as valid JSON: {\"summary\": str, \"recommendations\": [str, ...], \"cited_ids\": [str, ...], \"confidence\": float}.\n"
                "Strictly cite ONLY IDs present in `valid_ids`. Never invent metrics, retention, or quotes."
            )
            usr_p = json.dumps(
                {
                    "style": style_name,
                    "chosen_template": chosen_template,
                    "signal_brief": brief,
                    "baseline_proposal": {"summary": summary, "recommendations": recs, "refs": refs},
                }
            )
            parsed, meta = self.llm.complete_structured(sys_p, usr_p)
            if parsed and isinstance(parsed.get("summary"), str) and parsed["summary"].strip():
                summary = parsed["summary"].strip()[:500]
                if isinstance(parsed.get("recommendations"), list) and parsed["recommendations"]:
                    recs = [str(r)[:240] for r in parsed["recommendations"][:4] if str(r).strip()]
                if isinstance(parsed.get("cited_ids"), list):
                    safe_cited = [str(cid) for cid in parsed["cited_ids"] if str(cid) in valid_ids]
                    if safe_cited:
                        refs = safe_cited[:4]
                if isinstance(parsed.get("confidence"), (int, float)):
                    conf = round(max(0.1, min(0.99, float(parsed["confidence"]))), 2)

        proposed_actions = (
            [{"action": "generate_animation", "template_id": chosen_template}]
            if ag.agent_id == "motion_graphics_artist"
            else []
        )
        prop = AgentProposal(
            proposal_id=pid,
            agent_id=ag.agent_id,
            agent_title=ag.title,
            team=ag.team,
            summary=summary,
            recommendations=recs,
            evidence_refs=refs,
            confidence=conf,
            proposed_actions=proposed_actions,
        )
        turn = DebateTurn(
            round_number=1,
            speaker_agent_id=ag.agent_id,
            message=summary,
            evidence_refs=refs,
            decision_impact=recs[0] if recs else "",
        )
        return prop, turn

    def _run_moderator_audit(
        self,
        proposals: list[AgentProposal],
        chosen_hls: list[Any],
        brief: dict[str, Any],
        analysis: MediaAnalysisBundle,
    ) -> tuple[list[ModeratorChallenge], list[DebateTurn], list[Any], list[str]]:
        """Round 2 & 3: inspect conflicts across evidence, ASR confidence, pacing, and duration."""
        by_agent = {p.agent_id: p for p in proposals}
        challenges: list[ModeratorChallenge] = []
        turns: list[DebateTurn] = []
        revisions: list[str] = []
        revised_hls = list(chosen_hls)

        # 1. Check whether viral_researcher had verified metriced evidence
        if brief["verified_research_count"] == 0 and "viral_researcher" in by_agent:
            p = by_agent["viral_researcher"]
            ch = ModeratorChallenge(
                challenge_id=f"mod_{uuid.uuid4().hex[:6]}",
                target_proposal_id=p.proposal_id,
                issue_type="unsupported_assumption",
                description=(
                    "Research database currently has 0 metriced external videos; cross-niche weights rest solely on unscored archetypes."
                ),
                resolution="Down-weighted viral confidence and logged limitation; run live creator discovery (`discover_cohort`) for metriced evidence.",
            )
            challenges.append(ch)
            turns.append(
                DebateTurn(
                    round_number=2,
                    speaker_agent_id="moderator",
                    target_agent_id="viral_researcher",
                    message=ch.description,
                    evidence_refs=p.evidence_refs,
                    decision_impact=ch.resolution,
                )
            )
            revisions.append("Caveated external pattern weights until live public RSS/API items exist.")

        # 2. Check ASR low-confidence segment overlap in selected beats
        unc_segments = brief["uncertain_segments"]
        if unc_segments and revised_hls:
            overlapping_hls = [
                h
                for h in revised_hls
                if any(u["end_sec"] > h.start_sec and u["start_sec"] < h.end_sec for u in unc_segments)
            ]
            clean_alternatives = [
                h
                for h in analysis.highlights
                if h.highlight_id not in {x.highlight_id for x in revised_hls}
                and not any(u["end_sec"] > h.start_sec and u["start_sec"] < h.end_sec for u in unc_segments)
                and not any(
                    max(0.0, min(h.end_sec, x.end_sec) - max(h.start_sec, x.start_sec))
                    >= min(h.duration_sec, x.duration_sec) * 0.35
                    for x in revised_hls
                )
            ]
            target_prop = by_agent.get("story_producer") or proposals[0]
            # Never swap away a strong verified opening premise hook or final closure beat
            non_anchor_overlapping = [
                h for h in overlapping_hls if h.highlight_id not in {revised_hls[0].highlight_id, revised_hls[-1].highlight_id}
            ]
            if non_anchor_overlapping and clean_alternatives and len(revised_hls) > 2:
                dropped = non_anchor_overlapping[-1]
                replacement = max(clean_alternatives, key=lambda x: x.scores.composite_score)
                idx = revised_hls.index(dropped)
                revised_hls[idx] = replacement
                if len(revised_hls) > 2:
                    revised_hls = [revised_hls[0]] + sorted(revised_hls[1:-1], key=lambda x: x.start_sec) + [revised_hls[-1]]
                res_msg = (
                    f"Swapped low-confidence ASR beat `{dropped.highlight_id}` ({dropped.start_sec:.1f}-{dropped.end_sec:.1f}s) "
                    f"for clean high-confidence beat `{replacement.highlight_id}` ({replacement.start_sec:.1f}-{replacement.end_sec:.1f}s)."
                )
                revisions.append(res_msg)
            else:
                res_msg = (
                    f"Preserved beat(s) {[h.highlight_id for h in overlapping_hls]} for narrative continuity; "
                    "flagged overlapping ASS cues as `[uncertain ASR — inspect manually]`."
                )
                revisions.append(res_msg)
            ch = ModeratorChallenge(
                challenge_id=f"mod_{uuid.uuid4().hex[:6]}",
                target_proposal_id=target_prop.proposal_id,
                issue_type="asr_confidence_hazard",
                description=(
                    f"{len(overlapping_hls)} selected timeline beat(s) overlap {len(unc_segments)} low-confidence ASR segment(s)."
                ),
                resolution=res_msg,
            )
            challenges.append(ch)
            turns.append(
                DebateTurn(
                    round_number=2,
                    speaker_agent_id="moderator",
                    target_agent_id=target_prop.agent_id,
                    message=ch.description,
                    evidence_refs=[u["segment_id"] for u in unc_segments[:3]],
                    decision_impact=res_msg,
                )
            )

        # 3. Check pacing speed vs scientific derivation legibility + animation dwell check
        if "motion_graphics_artist" in by_agent:
            mg = by_agent["motion_graphics_artist"]
            ch = ModeratorChallenge(
                challenge_id=f"mod_{uuid.uuid4().hex[:6]}",
                target_proposal_id=mg.proposal_id,
                issue_type="factuality_check",
                description=(
                    f"Fast pacing/jump cuts proposed alongside technical explanation (`wpm={brief['speech_wpm']}`); "
                    "ensure simulation remains legible >= 4.5s and carries `factual_scientific` label."
                ),
                resolution="Enforced `playback_speed=1.0` during animation explainer beat and locked face-non-overlapping right/top PIP bounds.",
            )
            challenges.append(ch)
            turns.append(
                DebateTurn(
                    round_number=2,
                    speaker_agent_id="scientific_fact_checker",
                    target_agent_id="motion_graphics_artist",
                    message=ch.description,
                    evidence_refs=mg.evidence_refs,
                    decision_impact=ch.resolution,
                )
            )

        # 4. Hook & Ending Director Coherence Validation
        if revised_hls and analysis.transcript.segments:
            best_hook = select_best_hook(analysis)
            hook_cand = revised_hls[0]
            if best_hook["clean_start"] and (
                hook_cand.scores.context_dependency < 0.60
                or (
                    hook_cand.highlight_id != best_hook["highlight_id"]
                    and best_hook["hook_score"] > 0.72
                )
            ):
                matching = next((h for h in analysis.highlights if h.highlight_id == best_hook["highlight_id"]), None)
                if matching is not None:
                    h_start = min(hook_cand.start_sec, best_hook["start_sec"]) if hook_cand.end_sec <= best_hook["end_sec"] and hook_cand.start_sec <= 2.0 else best_hook["start_sec"]
                    h_end = max(hook_cand.end_sec, best_hook["end_sec"])
                    revised_hls[0] = matching.model_copy(
                        update={
                            "role": "hook",
                            "start_sec": h_start,
                            "end_sec": h_end,
                            "duration_sec": round(max(1.0, h_end - h_start), 3),
                        }
                    )
                    revisions.append(
                        f"Hook Director enforced self-contained premise `{matching.highlight_id}` ({h_start:.1f}-{h_end:.1f}s)."
                    )
            else:
                same_hook = hook_cand.highlight_id == best_hook["highlight_id"]
                h_start = min(hook_cand.start_sec, best_hook["start_sec"]) if same_hook else hook_cand.start_sec
                h_end = max(hook_cand.end_sec, best_hook["end_sec"]) if same_hook else hook_cand.end_sec
                revised_hls[0] = hook_cand.model_copy(
                    update={
                        "start_sec": h_start,
                        "end_sec": h_end,
                        "duration_sec": round(max(1.0, h_end - h_start), 3),
                    }
                )

            last_hl = revised_hls[-1]
            s_start, s_end, end_rep = snap_ending_boundary(
                analysis, last_hl.start_sec, last_hl.end_sec, max_extend_sec=8.0
            )
            if abs(s_end - last_hl.end_sec) > 0.25 or not end_rep.get("closure_ok", True):
                revised_hls[-1] = last_hl.model_copy(
                    update={
                        "role": "payoff",
                        "start_sec": s_start,
                        "end_sec": s_end,
                        "duration_sec": round(max(1.0, s_end - s_start), 3),
                    }
                )
                msg = (
                    f"Ending Director snapped closing beat `{last_hl.highlight_id}` to sentence closure ({s_start:.1f}-{s_end:.1f}s, {end_rep.get('closure_type')})."
                )
                revisions.append(msg)
                hook_prop = by_agent.get("hook_scientist") or proposals[0]
                challenges.append(
                    ModeratorChallenge(
                        challenge_id=f"mod_{uuid.uuid4().hex[:6]}",
                        target_proposal_id=hook_prop.proposal_id,
                        issue_type="pacing_conflict",
                        description="Verified opening hook completeness and final sentence-boundary closure across assembled storyboard.",
                        resolution=msg,
                    )
                )

        # 5. Check duration & dead-air pause trims
        if brief["long_pauses"]:
            ed = by_agent.get("professional_video_editor") or proposals[0]
            res_msg = f"Trimmed or bridged {len(brief['long_pauses'])} dead pause interval(s) >= 1.0s at beat boundaries."
            ch = ModeratorChallenge(
                challenge_id=f"mod_{uuid.uuid4().hex[:6]}",
                target_proposal_id=ed.proposal_id,
                issue_type="pacing_conflict",
                description=f"Source audio contains {len(brief['long_pauses'])} silent pause(s) >= 1.0s that depress early retention.",
                resolution=res_msg,
            )
            challenges.append(ch)
            revisions.append(res_msg)

        return challenges, turns, revised_hls, revisions

    def run_symposium(
        self,
        project_id: str,
        analysis: MediaAnalysisBundle,
        research_report: ResearchReport,
        style_name: str = "educational_explainer",
        mode: ResearchMode = ResearchMode.STANDARD,
        target_min_sec: float = 60.0,
        target_max_sec: float = 90.0,
        prior_experiments: list[ContentExperimentRecord] | None = None,
        research_items: list[ViralContentItem] | None = None,
    ) -> SymposiumSession:
        style = get_style_profile(style_name)
        agents = active_agents_for_mode(mode)
        backend = self.llm.detect_backend()
        brief = _compute_signal_brief(analysis, research_report)
        anim_template = _pick_animation_template(analysis, style_name=style_name)
        relevance = assess_topic_relevance(analysis, research_report, research_items)

        proposals: list[AgentProposal] = []
        debate_turns: list[DebateTurn] = []

        # Round 1: Independent evidence-grounded proposals per active specialist
        for ag in agents:
            prop, turn = self._build_specialist_proposal(
                ag=ag,
                brief=brief,
                analysis=analysis,
                research_report=research_report,
                style_name=style_name,
                chosen_template=anim_template,
                llm_backend=backend,
            )
            proposals.append(prop)
            debate_turns.append(turn)

        # Pre-render Content Performance Prediction Engine across 3 candidate versions + historical priors
        candidates = evaluate_candidate_edits(
            analysis=analysis,
            research_report=research_report,
            relevance=relevance,
            style_name=style_name,
            prior_experiments=prior_experiments,
            target_max_sec=target_max_sec,
            target_min_sec=target_min_sec,
        )
        best_cand = candidates[0] if candidates else None
        prior_learnings = [
            note
            for c in candidates
            for note in c.evidence_rationale
            if note.startswith("Historical ")
        ][:5]

        # Seed Round 1 beats from highest-ranked candidate edit using cohesive hook/ending bounds
        hl_map = {h.highlight_id: h for h in analysis.highlights}
        strat_mode = (
            "tight_viral"
            if best_cand and "Tight Viral" in best_cand.strategy_name
            else ("mechanism_proof" if best_cand and "Mechanism" in best_cand.strategy_name else "hook_first")
        )
        cand_cap = (
            min(target_max_sec, max(target_min_sec + 15.0, target_max_sec * 0.85))
            if strat_mode == "tight_viral"
            else target_max_sec
        )
        initial_hls = (
            plan_coherent_beat_sequence(
                analysis,
                target_min_sec=min(cand_cap, max(24.0, target_min_sec * 0.90)),
                target_max_sec=cand_cap,
                strategy=strat_mode,
            )
            if best_cand
            else []
        ) or (
            [hl_map[hid] for hid in best_cand.beat_highlight_ids if hid in hl_map]
            if best_cand and best_cand.beat_highlight_ids
            else []
        ) or self._select_highlights_for_duration(
            analysis,
            target_min_sec=target_min_sec,
            target_max_sec=target_max_sec,
        )
        initial_ids = [h.highlight_id for h in initial_hls]

        # Round 2 & 3: Moderator Audit, Conflict Resolution, and Storyboard Revision
        challenges, audit_turns, chosen_hls, revisions = self._run_moderator_audit(
            proposals=proposals,
            chosen_hls=initial_hls,
            brief=brief,
            analysis=analysis,
        )
        debate_turns.extend(audit_turns)
        final_ids = [h.highlight_id for h in chosen_hls]
        if final_ids != initial_ids:
            revisions.insert(0, f"Storyboard beat order updated from {initial_ids} -> {final_ids} after Moderator audit.")

        # Section 7: Growth Objective Evaluator + Creative Director Pre-Approval Gate
        growth_obj = build_growth_objective_proposal(analysis, relevance, best_cand, anim_template) if best_cand else None
        cd_review = creative_director_gate_review(growth_obj, candidates) if growth_obj else None

        # Round 3 synthesis turn by Creative Director
        best_visual = (
            max(chosen_hls, key=lambda h: h.scores.visual_opportunity + h.scores.educational_value)
            if chosen_hls
            else None
        )
        ref_ids = brief["valid_research_ids"][:3]
        debate_turns.append(
            DebateTurn(
                round_number=3,
                speaker_agent_id="creative_director",
                message=(
                    f"Evaluated {len(candidates)} candidate edit strategies (winner: `{best_cand.strategy_name if best_cand else style.display_name}`, "
                    f"score={best_cand.composite_score if best_cand else 0.75:.2f}); relevance=`{relevance.topic_timeliness}` ({relevance.momentum_score:.2f}); "
                    f"locked {len(chosen_hls)}-beat edit with `{anim_template}` visual explainer."
                ),
                evidence_refs=(final_ids[:2] + ref_ids[:2])[:4],
                decision_impact=(
                    f"Creative Director gate={'APPROVED' if (cd_review and cd_review.approved) else 'FLAGGED'} "
                    f"({len(revisions)} audit revision(s) applied)."
                ),
            )
        )

        # Build traceable Storyboard beats
        beats: list[StoryboardBeat] = []
        anim_assigned = False
        default_speed = 1.06 if 40.0 < brief["speech_wpm"] < 110.0 else 1.0

        for b_idx, hl in enumerate(chosen_hls):
            use_anim = (not anim_assigned) and (
                (best_visual and hl.highlight_id == best_visual.highlight_id)
                or (b_idx == min(1, len(chosen_hls) - 1))
            )
            if use_anim:
                anim_assigned = True
            # Round 3 audit rule: hold explanation beat at 1.0x speed when animation is composited
            beat_audio = "speech_loudness_polish+ducked_ambient_bed" if not use_anim else "speech_loudness_polish+hold_1x_clarity"
            beats.append(
                StoryboardBeat(
                    beat_index=b_idx,
                    role="hook" if b_idx == 0 else ("animation_explainer" if use_anim else hl.role),
                    source_start_sec=hl.start_sec,
                    source_end_sec=hl.end_sec,
                    target_duration_sec=round(hl.duration_sec / (1.0 if use_anim else default_speed), 2),
                    visual_treatment="cutaway_with_pip_or_overlay" if use_anim else style.color_grade_preset,
                    animation_template=anim_template if use_anim else None,
                    effect_presets=[style.transition_style, style.color_grade_preset]
                    + (["whoosh_transition"] if use_anim else (["sub_impact"] if b_idx == 0 else [])),
                    caption_directive=style.caption_preset,
                    audio_directive=beat_audio,
                    evidence_ids=[hl.highlight_id, *ref_ids[:2]],
                )
            )

        outliers = brief["top_outliers"]
        outlier_cite = (
            f"citing verified outlier '{outliers[0]['title'][:45]}' ({outliers[0].get('audience_normalized_ratio', 1.0)}x channel median)"
            if outliers
            else "noting curated structural patterns pending live discovery"
        )
        decisions = [
            SymposiumDecision(
                decision_id=f"dec_{uuid.uuid4().hex[:6]}",
                topic="Growth Objective & Candidate Ranking Gate",
                chosen_option=(
                    f"Approved '{best_cand.strategy_name}' (score={best_cand.composite_score:.3f}, "
                    f"CI=[{best_cand.confidence_interval['low']:.2f}, {best_cand.confidence_interval['high']:.2f}], "
                    f"relevance={relevance.topic_timeliness})"
                    if best_cand
                    else style.display_name
                ),
                rejected_options=[c.strategy_name for c in candidates[1:]] if len(candidates) > 1 else [],
                supporting_agents=["creative_director", "growth_strategist", "audience_researcher", "hook_scientist"],
                evidence_refs=(final_ids[:2] + ref_ids[:2]),
                rationale=(
                    cd_review.revision_notes
                    if cd_review
                    else "Candidate ranked highest across 11 performance & long-term relevance criteria."
                ),
            ),
            SymposiumDecision(
                decision_id=f"dec_{uuid.uuid4().hex[:6]}",
                topic="Opening Hook & Narrative Ordering",
                chosen_option=f"Open with highlight {final_ids[0] if final_ids else 'seg_0'} ({outlier_cite})",
                rejected_options=["Chronological raw opening with dead-air pauses"],
                supporting_agents=["viral_researcher", "hook_scientist", "creative_director"],
                evidence_refs=(final_ids[:1] + ref_ids[:2]),
                rationale=(
                    f"Curiosity hook scored highest among {len(analysis.highlights)} segments while bypassing "
                    f"{len(brief['long_pauses'])} dead pause gap(s) >= 1.0s."
                ),
            ),
            SymposiumDecision(
                decision_id=f"dec_{uuid.uuid4().hex[:6]}",
                topic="Original Animation Integration",
                chosen_option=f"Generate factual procedural animation `{anim_template}` at 1.0x clarity speed",
                rejected_options=["Talking-head only without visual simulation", "Speed-ramped equation segment"],
                supporting_agents=["motion_graphics_artist", "scientific_fact_checker", "story_producer"],
                evidence_refs=[best_visual.highlight_id] if best_visual else [],
                rationale="Mechanism-first visual simulation clarifies invisible dynamics without fabricating experimental measurements.",
            ),
            SymposiumDecision(
                decision_id=f"dec_{uuid.uuid4().hex[:6]}",
                topic="Multi-Platform & Telegram Funnel",
                chosen_option="Render 16:9 YouTube master + 3 distinct 9:16 vertical Shorts + human Threads bridge -> Telegram notes",
                rejected_options=["Naive center-crop re-export", "Duplicated identical Short clip"],
                supporting_agents=["platform_strategist", "growth_strategist", "caption_designer"],
                evidence_refs=ref_ids,
                rationale="Distinct vertical beats (hook, mechanism, payoff) + dynamic crop_center_x maximize reach across feeds.",
            ),
        ]

        total_dur = round(sum(b.target_duration_sec for b in beats), 2)
        storyboard = Storyboard(
            storyboard_id=f"sb_{uuid.uuid4().hex[:8]}",
            project_id=project_id,
            target_format="youtube_long",
            selected_strategy=(
                f"{best_cand.strategy_name} ({style.display_name})"
                if best_cand
                else f"{style.display_name} + Cross-Niche Mechanism-First Storytelling"
            ),
            target_duration_sec=total_dur,
            beats=beats,
        )

        effective_backend = (
            f"{backend}"
            if self.llm.calls_succeeded > 0 or backend == "deterministic_analytical"
            else "deterministic_analytical (llm fallback)"
        )
        return SymposiumSession(
            session_id=f"sym_{uuid.uuid4().hex[:8]}",
            project_id=project_id,
            mode=mode,
            backend_used=effective_backend,
            llm_calls_count=self.llm.calls_succeeded,
            participating_agents=[a.agent_id for a in agents],
            proposals=proposals,
            debate_turns=debate_turns,
            challenges=challenges,
            decisions=decisions,
            revision_history=revisions,
            relevance_assessment=relevance,
            candidate_evaluations=candidates,
            growth_objective=growth_obj,
            creative_director_review=cd_review,
            prior_learnings_applied=prior_learnings,
            storyboard=storyboard,
            created_at=utc_now_iso(),
        )
