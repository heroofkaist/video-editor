"""Content Repurposing Engine: YouTube metadata, Shorts/TikTok/Reels copy, and natural Threads posts."""

from __future__ import annotations

import re
import uuid

from symposium_studio.engines.analysis.semantics import TOPIC_STOPWORDS
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


def _is_kk_ru(analysis: MediaAnalysisBundle) -> bool:
    if (analysis.transcript.detected_language or "").lower() in {"kk", "ru"}:
        return True
    sample_txt = " ".join(s.text for s in analysis.transcript.segments[:8])
    return len(re.findall(r"[а-яёәіңғүұқөһ]", sample_txt.lower())) >= 12


def _extract_grounded_sentences(analysis: MediaAnalysisBundle, max_n: int = 5) -> list[str]:
    sents: list[str] = []
    for hl in analysis.highlights:
        cleaned = re.sub(r"\[uncertain\]", "", hl.transcript_excerpt or "").strip()
        if not cleaned:
            continue
        first = re.split(r"(?<=[.!?])\s+", cleaned)[0].strip()
        if first and len(first) >= 8 and first not in sents:
            sents.append(first)
        if len(sents) >= max_n:
            break
    if not sents:
        for seg in analysis.transcript.segments[:max_n]:
            txt = re.sub(r"\[uncertain\]", "", seg.text or "").strip()
            if txt:
                sents.append(txt)
    return sents


def _trim_to_limit(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    clipped = clean[: max_chars - 3].rsplit(" ", 1)[0].rstrip(" ,;:")
    return (clipped or clean[: max_chars - 3]) + "..."


def generate_threads_posts(
    analysis: MediaAnalysisBundle,
    topic_label: str,
    telegram_channel: str,
    max_chars: int = 280,
    styles: list[ThreadsStyle] | None = None,
) -> list[ThreadsPost]:
    """Generate concise, human-voiced Threads posts grounded strictly in transcript facts."""
    facts = _extract_grounded_sentences(analysis, max_n=4)
    kk_ru = _is_kk_ru(analysis)
    lead_fact = facts[0] if facts else (f"Разбираем тему: {topic_label}." if kk_ru else f"Working through {topic_label} today.")
    second_fact = facts[1] if len(facts) > 1 else lead_fact
    chan_ref = f"@{telegram_channel.lstrip('@')}"
    styles_to_build = styles or list(ThreadsStyle)

    if kk_ru:
        templates: dict[ThreadsStyle, str] = {
            ThreadsStyle.PROVOCATIVE: (
                f"Осы сәт көпшілікке таныс шығар: «{lead_fact}» "
                f"А через пару лет вообще всё иначе ощущается. Продолжение и заметки в {chan_ref}."
            ),
            ThreadsStyle.EDUCATIONAL: (
                f"Коротко по теме ({topic_label}): «{lead_fact}» "
                f"Дальше ключевой акцент: «{second_fact}». Детали и разбор — в {chan_ref}."
            ),
            ThreadsStyle.PERSONAL: (
                f"Записал честное видео без фильтра: «{lead_fact}» "
                f"Интересно сравнить, что получится через два года. Закулисье — в {chan_ref}."
            ),
            ThreadsStyle.STORYTELLING: (
                f"Начинал запись с одной мысли: «{lead_fact}» "
                f"А в итоге получился целый слепок времени и целей. Полный выпуск и обсуждение в {chan_ref}."
            ),
            ThreadsStyle.HUMOROUS: (
                f"Ожидание vs реальность первого семестра: «{lead_fact}» "
                f"Кто тоже через это проходил? Заходите обсудить в {chan_ref}."
            ),
            ThreadsStyle.TECHNICAL: (
                f"{topic_label}: «{lead_fact}» "
                f"Второй ориентир: «{second_fact}». Фиксируем прогресс и инсайты в {chan_ref}."
            ),
        }
        cta_line = f"Присоединяйтесь к {chan_ref} — там продолжение истории и обсуждение."
    else:
        templates = {
            ThreadsStyle.PROVOCATIVE: (
                f"Most explanations skip the exact step where things actually happen: {lead_fact} "
                f"Curious where people still get tripped up on this. Full diagram notes in {chan_ref}."
            ),
            ThreadsStyle.EDUCATIONAL: (
                f"Quick breakdown from today's build on {topic_label}: {lead_fact} "
                f"Once you see {second_fact.lower()}, it clicks. Extra notes in {chan_ref}."
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
                f"Me trying to explain {topic_label} cleanly vs. actually showing it step by step: {lead_fact} "
                f"Sending the visual notes to {chan_ref} so nobody suffers."
            ),
            ThreadsStyle.TECHNICAL: (
                f"{topic_label}: {lead_fact} "
                f"Key point: {second_fact} "
                f"Putting trace parameters and follow-up breakdown in {chan_ref}."
            ),
        }
        cta_line = f"Join {chan_ref} for the underlying diagrams and follow-up notes."

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
                cta_to_telegram=cta_line,
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
    kk_ru = _is_kk_ru(analysis)
    all_kw = []
    for t in analysis.topics:
        all_kw.extend(t.keywords)
    unique_kw = [
        k for k in dict.fromkeys(k.strip() for k in all_kw)
        if len(k) > 2 and k.lower() not in TOPIC_STOPWORDS
    ][:10]
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
    second_claim = facts[1] if len(facts) > 1 else lead_claim
    lead_hook = _trim_to_limit(lead_claim, 54).rstrip(".")
    chapter_lines = "\n".join(f"{ch['timestamp']} - {ch['title']}" for ch in chapters)
    tg_cta = telegram_campaign.platform_ctas.get("youtube", f"Join Telegram: {telegram_campaign.target_channel}")

    if kk_ru:
        yt_titles = [
            f"{project_name}: Ожидания vs Реальность через 2 года",
            f"«{lead_hook}» — честная капсула времени себе в будущее",
            f"Первый семестр, цели и реальность: {topic_label}",
            f"Запись себе через 2 года: GPA, стажировка, дисциплина",
            f"Что я сказал себе на 1 курсе: открою через 2 года",
        ]
        yt_desc = (
            f"{lead_claim}\n{second_claim}\n\n"
            f"Честная запись и разбор главных целей: {topic_label}.\n\n"
            f"Таймкоды:\n{chapter_lines}\n\n"
            f"Telegram-сообщество и обсуждение:\n{tg_cta}\n"
        )
        thumb_concepts = [
            {
                "concept": "Ожидание vs Реальность",
                "visual": "Контрастный кадр говорящего с реакцией + подпись про проверку через 2 года",
                "overlay_text": "ЧЕРЕЗ 2 ГОДА?",
            },
            {
                "concept": "Личный челлендж и цели",
                "visual": "Крупный план спикера, слева крупный ключевой вопрос или ориентир из речи",
                "overlay_text": "1 КУРС VS БУДУЩЕЕ",
            },
            {
                "concept": "Минималистичная капсула времени",
                "visual": "Чистый темный акцентный бейдж сверху + живой кадр автора",
                "overlay_text": _trim_to_limit(topic_label.upper(), 18),
            },
        ]
        follow_ups = [
            "Обновление через семестр: что реально получилось из поставленных целей",
            "Опрос в Telegram: какие ошибки 1 курса оказались самыми важными",
            f"Разбор ключевых тем выпуска ({topic_label}) в формате короткого треда",
        ]
    else:
        is_tech = any(t.is_technical for t in analysis.topics)
        yt_titles = [
            f"Why {topic_label.title()} Actually Works Like This (Visual Breakdown)" if is_tech else f"{project_name}: Honest Breakdown & Story",
            f"{project_name}: The Core Mechanism Explained Clearly" if is_tech else f"\"{lead_hook}\" — Full Story & Takeaways",
            f"Watch What Happens When {topic_label.title()} Changes" if is_tech else f"Reality Check on {topic_label.title()}",
            f"How {project_name} Works Step-by-Step" if is_tech else f"Inside {project_name}: What Actually Matters",
            f"The Real Reason Behind {topic_label.title()}",
        ]
        yt_desc = (
            f"{lead_claim}\n\n"
            f"In this video on {project_name}, we break down the core story and key takeaways.\n\n"
            f"Chapters:\n{chapter_lines}\n\n"
            f"Community & Notes:\n{tg_cta}\n"
        )
        thumb_concepts = [
            {
                "concept": "Visual Mechanism Contrast" if is_tech else "Expectation vs Reality",
                "visual": "High-contrast split frame: speaker reaction left + crisp visual graphic right",
                "overlay_text": "WHY IT WORKS" if is_tech else "THE REAL TEST",
            },
            {
                "concept": "Core Question Curiosity",
                "visual": "Close-up presenter reaction frame with crisp single-phrase visual callout",
                "overlay_text": "WHAT HAPPENED?",
            },
            {
                "concept": "Minimal Editorial Proof",
                "visual": "Clean dark navy canvas + single core callout + warm presenter rim light",
                "overlay_text": topic_label.upper()[:18],
            },
        ]
        follow_ups = [
            f"Deep-dive follow-up unpacking next steps on {topic_label}",
            "Interactive Q&A breakdown responding to Telegram community poll answers",
            "Behind-the-scenes reflection & progress checkpoint episode",
        ]

    yt_pkg = YouTubePackage(
        title_candidates=yt_titles,
        description=yt_desc,
        chapters=chapters,
        tags=unique_kw or [project_name.lower()],
        thumbnail_concepts=thumb_concepts,
        thumbnail_path=thumbnail_path,
    )

    clean_hashtags = []
    for k in unique_kw[:6]:
        tok = re.sub(r"[^a-zа-яёәіңғүұқөһ0-9]", "", k.lower())
        if len(tok) >= 2:
            clean_hashtags.append(f"#{tok}")

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
                    hashtags=clean_hashtags[:5],
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
    return SocialRepurposingBundle(
        project_id=project_id,
        youtube=yt_pkg,
        shorts_copy=shorts_copy,
        threads_posts=threads_posts,
        short_summary=_trim_to_limit(summary, 320),
        follow_up_ideas=follow_ups,
        telegram_campaign=telegram_campaign,
    )
