"""Content Repurposing Engine: YouTube metadata, Shorts/TikTok/Reels copy, and natural Threads posts."""

from __future__ import annotations

import re
import uuid

from symposium_studio.models.media import MediaAnalysisBundle
from symposium_studio.models.social import (
    ShortFormCopy,
    SocialRepurposingBundle,
    TelegramCampaign,
    ThreadsPost,
    ThreadsStyle,
    YouTubePackage,
)
from symposium_studio.models.timeline import RenderTimeline


def _fmt_chapter_ts(sec: float) -> str:
    total = max(0, int(sec))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _extract_grounded_sentences(analysis: MediaAnalysisBundle, max_n: int = 5) -> list[str]:
    sents: list[str] = []
    for hl in analysis.highlights:
        first = re.split(r"(?<=[.!?])\s+", hl.transcript_excerpt.strip())[0].strip()
        if first and first not in sents:
            sents.append(first)
        if len(sents) >= max_n:
            break
    if not sents:
        for seg in analysis.transcript.segments[:max_n]:
            if seg.text.strip():
                sents.append(seg.text.strip())
    return sents


def _trim_to_limit(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    clipped = clean[: max_chars - 3].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "..."


def generate_threads_posts(
    analysis: MediaAnalysisBundle,
    topic_label: str,
    telegram_channel: str,
    max_chars: int = 280,
    styles: list[ThreadsStyle] | None = None,
) -> list[ThreadsPost]:
    """Generate concise, human-voiced Threads posts grounded strictly in transcript facts."""
    facts = _extract_grounded_sentences(analysis, max_n=4)
    lead_fact = facts[0] if facts else f"Working through {topic_label} today."
    second_fact = facts[1] if len(facts) > 1 else lead_fact
    chan_ref = f"@{telegram_channel.lstrip('@')}"
    styles_to_build = styles or list(ThreadsStyle)

    templates: dict[ThreadsStyle, str] = {
        ThreadsStyle.PROVOCATIVE: (
            f"Most explanations skip the exact step where things actually happen: {lead_fact} "
            f"Curious where people still get tripped up on this. Full diagram notes in {chan_ref}."
        ),
        ThreadsStyle.EDUCATIONAL: (
            f"Quick breakdown from today's build on {topic_label}: {lead_fact} "
            f"Once you see {second_fact.lower()}, the math clicks. Extra notes in {chan_ref}."
        ),
        ThreadsStyle.PERSONAL: (
            f"Spent time testing {topic_label} on camera today. "
            f"The part that stood out most: {lead_fact} "
            f"Shared the raw sketch in {chan_ref}."
        ),
        ThreadsStyle.STORYTELLING: (
            f"Started recording thinking one angle would explain {topic_label}, then realized the real turning point was simpler: "
            f"{lead_fact} Full walkthrough + notes in {chan_ref}."
        ),
        ThreadsStyle.HUMOROUS: (
            f"Me trying to explain {topic_label} without a diagram vs. finally animating it: {lead_fact} "
            f"Sending the clean visual to {chan_ref} so nobody suffers."
        ),
        ThreadsStyle.TECHNICAL: (
            f"{topic_label}: {lead_fact} "
            f"Key point: {second_fact} "
            f"Putting the equations, trace parameters, and follow-up breakdown in {chan_ref}."
        ),
    }

    cap_limit = min(500, max(120, max_chars))
    posts: list[ThreadsPost] = []
    for st in styles_to_build:
        body = _trim_to_limit(templates[st], cap_limit)
        posts.append(
            ThreadsPost(
                post_id=f"thr_{st.value}_{uuid.uuid4().hex[:6]}",
                style=st,
                text=body,
                char_count=len(body),
                max_chars=cap_limit,
                cta_to_telegram=f"Join {chan_ref} for the underlying diagrams and follow-up notes.",
                grounded_facts=facts[:2],
            )
        )
    return posts


def build_social_repurposing_bundle(
    project_id: str,
    project_name: str,
    analysis: MediaAnalysisBundle,
    youtube_timeline: RenderTimeline,
    telegram_campaign: TelegramCampaign,
    thumbnail_path: str | None = None,
    short_video_paths: dict[str, list[str]] | None = None,
    threads_max_chars: int = 280,
) -> SocialRepurposingBundle:
    facts = _extract_grounded_sentences(analysis, max_n=5)
    all_kw = []
    for t in analysis.topics:
        all_kw.extend(t.keywords)
    unique_kw = list(dict.fromkeys(k for k in all_kw if len(k) > 2))[:10]
    topic_label = ", ".join(unique_kw[:3]) if unique_kw else project_name

    # Chapters from real YouTube timeline clips
    chapters: list[dict[str, str]] = []
    for idx, c in enumerate(youtube_timeline.clips):
        chapters.append(
            {
                "timestamp": _fmt_chapter_ts(c.timeline_start_sec),
                "title": (c.label or f"Section {idx + 1}").replace("[Hook Boosted] ", "")[:60],
            }
        )

    lead_claim = facts[0] if facts else project_name
    yt_titles = [
        f"Why {topic_label.title()} Actually Works Like This (Visual Breakdown)",
        f"{project_name}: The Core Mechanism Explained Clearly",
        f"Watch What Happens When {topic_label.title()} Changes",
        f"How {project_name} Works Step-by-Step (With Original Simulation)",
        f"The Real Reason Behind {topic_label.title()}",
    ]

    chapter_lines = "\n".join(f"{ch['timestamp']} - {ch['title']}" for ch in chapters)
    tg_cta = telegram_campaign.platform_ctas.get("youtube", f"Join Telegram: {telegram_campaign.target_channel}")
    yt_desc = (
        f"{lead_claim}\n\n"
        f"In this breakdown of {project_name}, we examine the underlying mechanism with factual visual models.\n\n"
        f"Chapters:\n{chapter_lines}\n\n"
        f"Community & Technical Notes:\n{tg_cta}\n"
    )

    thumb_concepts = [
        {
            "concept": "Visual Mechanism Contrast",
            "visual": "High-contrast split frame: speaker reaction left + glowing coil/flux diagram right",
            "overlay_text": "WHY IT FLOWS",
        },
        {
            "concept": "Zero-vs-Motion Curiosity",
            "visual": "Close-up simulation frame showing 0 V static vs active current pulse",
            "overlay_text": "STOPPED = ZERO",
        },
        {
            "concept": "Minimal Editorial Proof",
            "visual": "Clean dark navy canvas + single core equation + warm presenter rim light",
            "overlay_text": topic_label.upper()[:18],
        },
    ]

    yt_pkg = YouTubePackage(
        title_candidates=yt_titles,
        description=yt_desc,
        chapters=chapters,
        tags=unique_kw or [project_name.lower()],
        thumbnail_concepts=thumb_concepts,
        thumbnail_path=thumbnail_path,
    )

    short_paths = short_video_paths or {}
    shorts_copy: list[ShortFormCopy] = []
    for plat, default_cta_key in (
        ("youtube_shorts", "youtube_shorts"),
        ("tiktok", "tiktok"),
        ("instagram_reels", "instagram_reels"),
    ):
        vids = short_paths.get(plat, [None])
        for v_i, v_path in enumerate(vids):
            hook = facts[v_i % len(facts)] if facts else project_name
            cta = telegram_campaign.platform_ctas.get(default_cta_key, "")
            if plat == "tiktok":
                cap = f"{_trim_to_limit(hook, 120)} {cta}"
            elif plat == "instagram_reels":
                cap = f"{_trim_to_limit(hook, 150)}\n\n{cta}"
            else:
                cap = f"{_trim_to_limit(hook, 95)} | {cta}"
            shorts_copy.append(
                ShortFormCopy(
                    platform=plat,
                    variant_id=f"{plat}_{v_i + 1}",
                    hook_line=_trim_to_limit(hook, 85),
                    caption=cap.strip(),
                    hashtags=[f"#{re.sub(r'[^a-z0-9]', '', k.lower())}" for k in unique_kw[:5] if k],
                    cta_to_telegram=cta,
                    video_path=v_path,
                )
            )

    threads_posts = generate_threads_posts(
        analysis=analysis,
        topic_label=topic_label,
        telegram_channel=telegram_campaign.target_channel,
        max_chars=threads_max_chars,
    )

    summary = f"{project_name}: " + " ".join(facts[:2])
    follow_ups = [
        f"Deep-dive follow-up testing edge cases of {topic_label}",
        "Interactive Q&A breakdown responding to Telegram community poll answers",
        "Behind-the-scenes build log showing animation & math verification setup",
    ]

    return SocialRepurposingBundle(
        project_id=project_id,
        youtube=yt_pkg,
        shorts_copy=shorts_copy,
        threads_posts=threads_posts,
        short_summary=_trim_to_limit(summary, 320),
        follow_up_ideas=follow_ups,
        telegram_campaign=telegram_campaign,
    )
