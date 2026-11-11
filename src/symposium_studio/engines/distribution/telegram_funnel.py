"""Community Growth Engine: YouTube/Shorts/TikTok/IG/Threads -> Telegram community funnel."""

from __future__ import annotations

import uuid

from symposium_studio.models.media import MediaAnalysisBundle
from symposium_studio.models.social import FunnelLink, TelegramCampaign, TelegramPoll
from symposium_studio.storage.db import StudioDatabase


def build_telegram_campaign(
    project_id: str,
    project_name: str,
    analysis: MediaAnalysisBundle,
    target_channel: str = "@symposium_lab",
    db: StudioDatabase | None = None,
) -> TelegramCampaign:
    handle = target_channel.lstrip("@")
    slug = project_id.replace("proj_", "")
    keywords = []
    for t in analysis.topics:
        keywords.extend(t.keywords)
    kw = list(dict.fromkeys(keywords))[:6]
    topic_str = ", ".join(kw[:3]) if kw else project_name

    # Tagged referral links per upstream platform
    funnel_links: list[FunnelLink] = []
    platform_ctas: dict[str, str] = {}
    for plat in ("youtube", "youtube_shorts", "tiktok", "instagram_reels", "threads"):
        link_id = f"lnk_{plat}_{slug}"
        tagged = f"https://t.me/{handle}?start={slug}_{plat}"
        funnel_links.append(
            FunnelLink(
                link_id=link_id,
                source_platform=plat,
                campaign_slug=f"{slug}_{plat}",
                target_telegram_channel=target_channel,
                tagged_url=tagged,
            )
        )
        if db is not None:
            db.upsert_funnel_link(project_id, funnel_links[-1])

        if plat == "youtube":
            platform_ctas[plat] = f"Get the diagrams, equations & follow-up notes in Telegram: {tagged}"
        elif plat == "threads":
            platform_ctas[plat] = f"Sharing the full annotated diagram + discussion thread in {target_channel} ({tagged})"
        else:
            platform_ctas[plat] = f"Full diagrams & deep-dive notes in our Telegram ({target_channel})"

    first_hl = analysis.highlights[0].transcript_excerpt[:140] if analysis.highlights else project_name
    tech_topics = [t for t in analysis.topics if t.is_technical]
    deep_note = (
        f"Technical Follow-Up Note — {project_name}\n\n"
        f"Core focus: {topic_str}\n"
        f"Key excerpt from the recording: \"{first_hl}\"\n\n"
        + (
            "Topics covered in this session:\n"
            + "\n".join(f"- {t.title} ({t.start_sec:.1f}s–{t.end_sec:.1f}s)" for t in tech_topics[:4])
            if tech_topics
            else "Structured breakdown notes and visual specifications attached in project workspace."
        )
    )

    lead_post = (
        f"New breakdown live: {project_name}\n\n"
        f"In the video, we walked through {topic_str}. "
        f"Here in Telegram I'm posting the underlying visual spec, step-by-step equations/logic, "
        f"and the edge case that most quick clips skip.\n\n"
        f"Question for the lab: Which step felt least intuitive on first watch? Reply below."
    )

    poll = TelegramPoll(
        question=f"What should we break down next after '{project_name[:42]}'?",
        options=[
            f"Deeper math / mechanism on {kw[0] if kw else 'this topic'}",
            "Live bench test & failure modes",
            "How the animated diagram was generated",
            "Shorter 60s intuitive intuition clips",
        ],
    )

    return TelegramCampaign(
        campaign_id=f"tg_{uuid.uuid4().hex[:8]}",
        project_id=project_id,
        target_channel=target_channel,
        lead_post=lead_post,
        technical_deep_dive_note=deep_note,
        discussion_questions=[
            f"What part of {topic_str} would you test differently in practice?",
            "Should the next follow-up focus on intuition, real measurements, or simulation code?",
        ],
        resource_checklist=[
            "Annotated visual SceneSpec JSON (editable simulation parameters)",
            "Timestamped summary notes & source-of-truth OTIO timeline breakdown",
            "Follow-up discussion thread + community poll",
        ],
        mini_series_outline=[
            f"Part 1 (Short/Reel): 45s intuitive hook on {topic_str}",
            f"Part 2 (YouTube): Complete visual mechanism explanation",
            f"Part 3 (Telegram): Open Q&A, raw equations, and next experiment vote",
        ],
        poll=poll,
        suggested_posting_windows_utc=["14:00-16:00 UTC", "18:30-20:30 UTC"],
        funnel_links=funnel_links,
        platform_ctas=platform_ctas,
    )
