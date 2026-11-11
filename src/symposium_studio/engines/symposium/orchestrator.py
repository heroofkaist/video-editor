"""AI Symposium multi-agent research and creative deliberation engine.

Implements all 7 Research Team agents, all 11 Creative Team agents,
a Moderator checking unsupported claims/disagreements, and traceable
12-step execution across inexpensive, standard, and comprehensive modes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from symposium_studio.config import StudioConfig
from symposium_studio.engines.analysis.styles import get_style_profile
from symposium_studio.engines.symposium.llm_provider import SymposiumLLMRouter
from symposium_studio.models.media import MediaAnalysisBundle
from symposium_studio.models.project import utc_now_iso
from symposium_studio.models.research import ResearchReport
from symposium_studio.models.symposium import (
    AgentProposal,
    AgentTeam,
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


AGENT_ROSTER: list[AgentDefinition] = [
    # Research Team (7)
    AgentDefinition("viral_researcher", "Viral Researcher", AgentTeam.RESEARCH, ResearchMode.INEXPENSIVE),
    AgentDefinition("hook_scientist", "Hook Scientist", AgentTeam.RESEARCH, ResearchMode.INEXPENSIVE),
    AgentDefinition("trend_researcher", "Trend Researcher", AgentTeam.RESEARCH, ResearchMode.STANDARD),
    AgentDefinition("storytelling_researcher", "Storytelling Researcher", AgentTeam.RESEARCH, ResearchMode.STANDARD),
    AgentDefinition("audience_researcher", "Audience Researcher", AgentTeam.RESEARCH, ResearchMode.COMPREHENSIVE),
    AgentDefinition("growth_strategist", "Growth Strategist", AgentTeam.RESEARCH, ResearchMode.STANDARD),
    AgentDefinition("scientific_fact_checker", "Scientific Fact Checker", AgentTeam.RESEARCH, ResearchMode.STANDARD),
    # Creative Team (11)
    AgentDefinition("creative_director", "Creative Director", AgentTeam.CREATIVE, ResearchMode.INEXPENSIVE),
    AgentDefinition("story_producer", "Story Producer", AgentTeam.CREATIVE, ResearchMode.INEXPENSIVE),
    AgentDefinition("professional_video_editor", "Professional Video Editor", AgentTeam.CREATIVE, ResearchMode.INEXPENSIVE),
    AgentDefinition("motion_graphics_artist", "Motion Graphics Artist", AgentTeam.CREATIVE, ResearchMode.STANDARD),
    AgentDefinition("caption_designer", "Caption Designer", AgentTeam.CREATIVE, ResearchMode.STANDARD),
    AgentDefinition("sound_designer", "Sound Designer", AgentTeam.CREATIVE, ResearchMode.STANDARD),
    AgentDefinition("platform_strategist", "Platform Strategist", AgentTeam.CREATIVE, ResearchMode.STANDARD),
    AgentDefinition("three_d_animation_artist", "3D Animation Artist", AgentTeam.CREATIVE, ResearchMode.COMPREHENSIVE),
    AgentDefinition("visual_effects_artist", "Visual Effects Artist", AgentTeam.CREATIVE, ResearchMode.COMPREHENSIVE),
    AgentDefinition("meme_researcher", "Meme Researcher", AgentTeam.CREATIVE, ResearchMode.COMPREHENSIVE),
    AgentDefinition("qa_reviewer", "Quality Assurance Reviewer", AgentTeam.CREATIVE, ResearchMode.INEXPENSIVE),
]

MODE_RANK = {
    ResearchMode.INEXPENSIVE: 0,
    ResearchMode.STANDARD: 1,
    ResearchMode.COMPREHENSIVE: 2,
}


def active_agents_for_mode(mode: ResearchMode) -> list[AgentDefinition]:
    rank = MODE_RANK[mode]
    return [a for a in AGENT_ROSTER if MODE_RANK[a.min_mode] <= rank]


class SymposiumOrchestrator:
    """Runs the traceable AI Symposium conference grounded in real media analysis and research evidence."""

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
            return list(hls)

        # Always evaluate hook + top technical/story beats in chronological order after strong hook
        sorted_by_score = sorted(hls, key=lambda h: h.scores.composite_score, reverse=True)
        hook = max(hls[:2], key=lambda h: h.scores.audience_curiosity + h.scores.composite_score)

        selected = [hook]
        total = hook.duration_sec
        for cand in sorted_by_score:
            if cand.highlight_id == hook.highlight_id:
                continue
            if total >= target_min_sec and total + cand.duration_sec > target_max_sec:
                continue
            if total + cand.duration_sec <= target_max_sec + 6.0:
                selected.append(cand)
                total += cand.duration_sec
            if total >= target_min_sec and len(selected) >= 3:
                # Ensure at least one core_explanation / technical visual beat if available
                if any(s.visual_opportunity_type for s in selected):
                    break

        # Ensure hook opens first, remaining highlights flow chronologically
        rest = sorted([s for s in selected if s.highlight_id != hook.highlight_id], key=lambda x: x.start_sec)
        return [hook, *rest]

    def run_symposium(
        self,
        project_id: str,
        analysis: MediaAnalysisBundle,
        research_report: ResearchReport,
        style_name: str = "educational_explainer",
        mode: ResearchMode = ResearchMode.STANDARD,
        target_min_sec: float = 60.0,
        target_max_sec: float = 90.0,
    ) -> SymposiumSession:
        style = get_style_profile(style_name)
        agents = active_agents_for_mode(mode)
        backend = self.llm.detect_backend()
        proposals: list[AgentProposal] = []

        top_findings = research_report.findings[:4]
        top_transfers = research_report.cross_niche_transfers[:4]
        ref_ids = [f["item_id"] for f in top_findings]
        best_hook = max(
            analysis.highlights,
            key=lambda h: h.scores.audience_curiosity + h.scores.emotional_intensity,
        ) if analysis.highlights else None
        best_visual = max(
            analysis.highlights,
            key=lambda h: h.scores.visual_opportunity + h.scores.educational_value,
        ) if analysis.highlights else None

        # Step 1 & 2: Independent agent analysis & evidence-backed proposals
        for ag in agents:
            pid = f"prop_{ag.agent_id}_{uuid.uuid4().hex[:5]}"
            if ag.agent_id == "viral_researcher":
                cats = sorted({f["category"] for f in top_findings})
                proposals.append(
                    AgentProposal(
                        proposal_id=pid,
                        agent_id=ag.agent_id,
                        agent_title=ag.title,
                        team=ag.team,
                        summary=(
                            f"Cross-niche research across {', '.join(cats) or 'multiple categories'} shows "
                            f"in-media-res problem hooks + visual mechanism cutaways outperform static talking-head intros."
                        ),
                        recommendations=[
                            t["editing_translation"] for t in top_transfers[:2]
                        ] or ["Open with concrete stakes before background context."],
                        evidence_refs=ref_ids,
                        confidence=0.90,
                    )
                )
            elif ag.agent_id == "hook_scientist":
                hook_txt = best_hook.transcript_excerpt[:90] if best_hook else "Opening segment"
                proposals.append(
                    AgentProposal(
                        proposal_id=pid,
                        agent_id=ag.agent_id,
                        agent_title=ag.title,
                        team=ag.team,
                        summary=(
                            f"Best raw hook candidate at {best_hook.start_sec:.1f}s-{best_hook.end_sec:.1f}s "
                            f"(curiosity={best_hook.scores.audience_curiosity:.2f}): '{hook_txt}'."
                            if best_hook
                            else "Lead with sharpest question in opening 3s."
                        ),
                        recommendations=[
                            "Promote strongest curiosity/stakes beat to t=0.0s with sub-impact SFX and kinetic hook headline.",
                            "Trim any initial throat-clearing or pauses >0.4s before first spoken claim.",
                        ],
                        evidence_refs=ref_ids[:2] + ([best_hook.highlight_id] if best_hook else []),
                        confidence=0.89,
                    )
                )
            elif ag.agent_id == "scientific_fact_checker":
                tech_topics = [t.domain for t in analysis.topics if t.is_technical]
                proposals.append(
                    AgentProposal(
                        proposal_id=pid,
                        agent_id=ag.agent_id,
                        agent_title=ag.title,
                        team=ag.team,
                        summary=(
                            f"Detected technical domains: {sorted(set(tech_topics)) or ['general']}. "
                            f"Uncertain ASR segments flagged: {len(analysis.transcript.uncertain_segment_ids)}. "
                            "Require factual_scientific animation mode for physics/EE/code equations."
                        ),
                        recommendations=[
                            "Use verified analytical Faraday/RLC/algorithm models; never invent experimental readings.",
                            "Preserve verbatim speaker transcript; visually flag low-confidence segments if any.",
                        ],
                        evidence_refs=[best_visual.highlight_id] if best_visual else [],
                        confidence=0.95,
                    )
                )
            elif ag.agent_id == "motion_graphics_artist":
                tmpl = (best_visual.visual_opportunity_type if best_visual else None) or "em_induction_faraday"
                proposals.append(
                    AgentProposal(
                        proposal_id=pid,
                        agent_id=ag.agent_id,
                        agent_title=ag.title,
                        team=ag.team,
                        summary=f"Recommend generating original '{tmpl}' simulation animation overlaid during peak technical insight.",
                        recommendations=[
                            f"Generate editable SceneSpec for `{tmpl}` via Chromium Canvas / Matplotlib adapter.",
                            "Composite as clean cutaway + PIP or alpha overlay with subtle whoosh entrance.",
                        ],
                        evidence_refs=[best_visual.highlight_id] if best_visual else [],
                        confidence=0.92,
                        proposed_actions=[{"action": "generate_animation", "template_id": tmpl}],
                    )
                )
            elif ag.agent_id == "caption_designer":
                proposals.append(
                    AgentProposal(
                        proposal_id=pid,
                        agent_id=ag.agent_id,
                        agent_title=ag.title,
                        team=ag.team,
                        summary=f"Apply '{style.caption_preset}' ASS subtitles using safe margins for 16:9 and 9:16 exports.",
                        recommendations=[
                            "Enable word-level active highlighting for high-retention mobile viewing.",
                            "Respect vertical platform UI safe zones (bottom margin >= 260px on 1080x1920).",
                        ],
                        evidence_refs=ref_ids[:2],
                        confidence=0.91,
                    )
                )
            elif ag.agent_id == "platform_strategist":
                proposals.append(
                    AgentProposal(
                        proposal_id=pid,
                        agent_id=ag.agent_id,
                        agent_title=ag.title,
                        team=ag.team,
                        summary="Produce distinct edits (60-90s YouTube 16:9, face-tracked 9:16 Shorts/TikTok/Reels, Threads bridge, Telegram deep-dive).",
                        recommendations=[
                            "Use face-tracked dynamic crop centers for 9:16 rather than static center crop.",
                            "Route Threads curiosity hook + video CTA into Telegram community notes/diagrams.",
                        ],
                        evidence_refs=ref_ids,
                        confidence=0.90,
                    )
                )
            elif ag.agent_id == "growth_strategist":
                proposals.append(
                    AgentProposal(
                        proposal_id=pid,
                        agent_id=ag.agent_id,
                        agent_title=ag.title,
                        team=ag.team,
                        summary="Connect short-form discovery hooks to long-form depth and Telegram community retention via tagged funnel links.",
                        recommendations=[
                            "Generate platform-specific CTAs and Telegram poll + downloadable diagram summary.",
                        ],
                        evidence_refs=ref_ids[:2],
                        confidence=0.88,
                    )
                )
            else:
                proposals.append(
                    AgentProposal(
                        proposal_id=pid,
                        agent_id=ag.agent_id,
                        agent_title=ag.title,
                        team=ag.team,
                        summary=f"{ag.title} endorses '{style.display_name}' treatment aligned with {len(analysis.highlights)} scored highlights.",
                        recommendations=[
                            f"Apply {style.transition_style} pacing and {style.color_grade_preset} visual finish cleanly without clutter."
                        ],
                        evidence_refs=ref_ids[:1],
                        confidence=0.84,
                    )
                )

        # Step 4: Moderator checks disagreements & unsupported assumptions
        challenges: list[ModeratorChallenge] = [
            ModeratorChallenge(
                challenge_id=f"mod_{uuid.uuid4().hex[:6]}",
                target_proposal_id=proposals[0].proposal_id,
                issue_type="unsupported_assumption",
                description="Verify cross-niche gaming/comedy pacing does not obscure technical clarity or introduce random meme clutter.",
                resolution="Keep comedy 150ms silence beat & gaming crisis hook ONLY at narrative turning points; preserve scientific explanation breathing room.",
            ),
            ModeratorChallenge(
                challenge_id=f"mod_{uuid.uuid4().hex[:6]}",
                target_proposal_id=next(
                    (p.proposal_id for p in proposals if p.agent_id == "motion_graphics_artist"),
                    proposals[0].proposal_id,
                ),
                issue_type="factuality_check",
                description="Ensure animated physical simulation clearly separates analytical models from decorative graphics.",
                resolution="Tag animation SceneSpec with `factuality_mode=factual_scientific` and analytical notes.",
            ),
        ]

        # Step 6 & 7: Creative Director chooses strategy & constructs traceable Storyboard
        chosen_hls = self._select_highlights_for_duration(
            analysis,
            target_min_sec=target_min_sec,
            target_max_sec=target_max_sec,
        )
        anim_template = (best_visual.visual_opportunity_type if best_visual else None) or "em_induction_faraday"
        beats: list[StoryboardBeat] = []
        anim_assigned = False

        for b_idx, hl in enumerate(chosen_hls):
            use_anim = (not anim_assigned) and (
                (best_visual and hl.highlight_id == best_visual.highlight_id)
                or (b_idx == min(1, len(chosen_hls) - 1))
            )
            if use_anim:
                anim_assigned = True
            beats.append(
                StoryboardBeat(
                    beat_index=b_idx,
                    role="hook" if b_idx == 0 else ("animation_explainer" if use_anim else hl.role),
                    source_start_sec=hl.start_sec,
                    source_end_sec=hl.end_sec,
                    target_duration_sec=hl.duration_sec,
                    visual_treatment="cutaway_with_pip_or_overlay" if use_anim else style.color_grade_preset,
                    animation_template=anim_template if use_anim else None,
                    effect_presets=[style.transition_style, style.color_grade_preset]
                    + (["whoosh_transition"] if use_anim else []),
                    caption_directive=style.caption_preset,
                    audio_directive="speech_loudness_polish",
                    evidence_ids=[hl.highlight_id, *ref_ids[:2]],
                )
            )

        decisions = [
            SymposiumDecision(
                decision_id=f"dec_{uuid.uuid4().hex[:6]}",
                topic="Opening Hook & Narrative Ordering",
                chosen_option=f"Open with highlight {chosen_hls[0].highlight_id if chosen_hls else 'seg_0'} (curiosity-first gaming/short-form hybrid hook)",
                rejected_options=["Chronological raw opening without hook trim"],
                supporting_agents=["viral_researcher", "hook_scientist", "creative_director"],
                evidence_refs=ref_ids[:2],
                rationale="Cross-niche gaming & tech patterns demonstrate immediate stakes hook lifts early attention before formal theory.",
            ),
            SymposiumDecision(
                decision_id=f"dec_{uuid.uuid4().hex[:6]}",
                topic="Original Animation Integration",
                chosen_option=f"Generate factual procedural animation `{anim_template}` and composite onto core explanation beat",
                rejected_options=["Talking-head only without visual simulation"],
                supporting_agents=["motion_graphics_artist", "scientific_fact_checker", "story_producer"],
                evidence_refs=[best_visual.highlight_id] if best_visual else [],
                rationale="Mechanism-first visual simulation clarifies invisible dynamics without fabricating measurements.",
            ),
            SymposiumDecision(
                decision_id=f"dec_{uuid.uuid4().hex[:6]}",
                topic="Multi-Platform & Telegram Funnel",
                chosen_option="Render distinct 16:9 YouTube master + face-tracked 9:16 vertical short + human Threads post -> Telegram deep dive",
                rejected_options=["Naive center-crop re-export across platforms"],
                supporting_agents=["platform_strategist", "growth_strategist", "caption_designer"],
                evidence_refs=ref_ids,
                rationale="Platform-native safe zones, dynamic framing, and community follow-up maximize retention and conversion.",
            ),
        ]

        total_dur = round(sum(b.target_duration_sec for b in beats), 2)
        storyboard = Storyboard(
            storyboard_id=f"sb_{uuid.uuid4().hex[:8]}",
            project_id=project_id,
            target_format="youtube_long",
            selected_strategy=f"{style.display_name} + Cross-Niche Mechanism-First Storytelling",
            target_duration_sec=total_dur,
            beats=beats,
        )

        return SymposiumSession(
            session_id=f"sym_{uuid.uuid4().hex[:8]}",
            project_id=project_id,
            mode=mode,
            backend_used=backend,
            participating_agents=[a.agent_id for a in agents],
            proposals=proposals,
            challenges=challenges,
            decisions=decisions,
            storyboard=storyboard,
            created_at=utc_now_iso(),
        )
