"""Modular platform research connectors using official APIs, public feeds, and explicit imports.

Strictly enforces platform compliance: never bypasses auth gates, CAPTCHAs, or
private endpoints, and never fabricates private metrics.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import io
import json
import os
import re
import statistics
import uuid
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from symposium_studio.models.research import (
    ContentCategory,
    PublicEngagementMetrics,
    ResearchPlatform,
    ReusableTechnique,
    ViralContentItem,
)
from symposium_studio.models.project import utc_now_iso


# Verified public creator Atom syndication feeds across diverse non-STEM and STEM niches.
# No auth walls or scraping required; every feed returns up to 15 latest public videos with view counts.
DEFAULT_RESEARCH_CHANNELS: list[tuple[str, ContentCategory, str]] = [
    ("Veritasium", ContentCategory.SCIENCE, "UCHnyfMqiRRG1u-2MsSQLbXA"),
    ("3Blue1Brown", ContentCategory.SCIENCE, "UCYO_jab_esuFRV4b17AJtAw"),
    ("Mark Rober", ContentCategory.ENGINEERING, "UCY1kMZp36IQSyNx_9h4mpCg"),
    ("Fireship", ContentCategory.PROGRAMMING, "UCsBjURrPoezykLs9EqgamOA"),
    ("Computerphile", ContentCategory.TECHNOLOGY, "UC9-y-6csu5WGm29I7JiwpnA"),
    ("MKBHD", ContentCategory.TECHNOLOGY, "UCBJycsmduvYEL83R_U4JriQ"),
    ("Kurzgesagt", ContentCategory.EDUCATION, "UCsXVk37bltHxD1rDPwtNM8Q"),
    ("TED", ContentCategory.EDUCATION, "UCAuUUnT6oDeKwE6v1NGQxug"),
    ("Game Maker's Toolkit", ContentCategory.GAMING, "UCqJ-Xo29CKyLTjn6z2XwYAw"),
    ("Summoning Salt", ContentCategory.GAMING, "UCtUbO6rBht0daVIOGML3c8w"),
    ("Vox", ContentCategory.DOCUMENTARIES, "UCLXo7UDZvByw2ixzpQCufnA"),
    ("Wendover Productions", ContentCategory.DOCUMENTARIES, "UC9RM-iSvTu1uPJb8X5yp3EQ"),
    ("Y Combinator", ContentCategory.STARTUPS, "UCcefcZRL2oaA_uBNeo5UOWg"),
    ("Corridor Crew", ContentCategory.ENTERTAINMENT, "UCSpFnDQr88xCZ80N-X7t0nQ"),
    ("Saturday Night Live", ContentCategory.COMEDY, "UCqFzWxSCi39LnW1JKFR3efg"),
    ("Formula 1", ContentCategory.SPORTS, "UCB_qr75-ydFVKSF9Dmo6izg"),
    ("Matt D'Avella", ContentCategory.LIFESTYLE, "UCJ24N4O0bP7LGLBDvye7oCA"),
    ("Ali Abdaal", ContentCategory.PERSONAL_STORYTELLING, "UCoOae5nYA7VqaXzerajD0lg"),
]


def _days_since_iso(published_iso: str) -> float:
    if not published_iso:
        return 30.0
    try:
        dt = datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return max(1.0, delta)
    except Exception:
        return 30.0


def _analyze_public_packaging(
    title: str,
    description: str,
    category: ContentCategory,
    views: int | None,
    baseline_views: int | None,
    likes_or_ratings: int | None,
    is_short: bool,
) -> tuple[str, str, list[ReusableTechnique]]:
    """Derive observable title/hook/framing techniques strictly from metadata & audience breakout."""
    low = title.lower()
    patterns: list[str] = []
    if "?" in title:
        patterns.append("open-question curiosity gap")
    if re.search(r"\b(why|how|actually|secret|never|wrong|myth|impossible|broken|stop)\b", low):
        patterns.append("counterintuitive mechanism / myth-challenge framing")
    if re.search(r"\b\d+[kmbx%s]?\b", low):
        patterns.append("concrete numerical stakes / scale anchor")
    if re.search(r"\b(i |my |we |tested|built|tried|spent)\b", low):
        patterns.append("first-person experiment / stakes ownership")
    if not patterns:
        patterns.append("direct declarative topic promise")

    hook_summary = (
        f"Title pattern ({category.value}{' Short' if is_short else ''}): "
        + "; ".join(patterns)
        + "."
    )
    ratio = round(views / baseline_views, 2) if views is not None and baseline_views and baseline_views > 0 else None
    like_pct = round((likes_or_ratings / max(1, views)) * 100.0, 2) if (views and likes_or_ratings) else None
    edit_note = (
        f"Public RSS metadata ({'9:16 Short' if is_short else '16:9 YouTube'}; "
        f"breakout={ratio:.2f}x channel median" if ratio is not None else "Public YouTube metadata"
    )
    if like_pct is not None:
        edit_note += f", rating_ratio={like_pct:.2f}%)"
    elif ratio is not None:
        edit_note += ")"
    if description:
        first_line = description.strip().splitlines()[0][:120]
        if first_line:
            edit_note += f" — Lead description: '{first_line}'"

    techniques: list[ReusableTechnique] = []
    if ratio is not None and ratio >= 1.25:
        techniques.append(
            ReusableTechnique(
                technique_id=f"tech_rss_{category.value}_{uuid.uuid4().hex[:6]}",
                name=f"{category.value.replace('_', ' ').title()} Breakout Framing ({ratio:.1f}x Channel Baseline)",
                mechanism=f"Public video outperformed its creator's recent median by {ratio:.2f}x using {' + '.join(patterns)}.",
                applicable_niches=[category.value, "science", "engineering", "programming", "education", "startups"],
                editing_translation=(
                    f"Adapt '{title[:55]}' packaging: open visually with the {patterns[0]} in the first 3 seconds before formal exposition."
                ),
            )
        )
    elif patterns:
        techniques.append(
            ReusableTechnique(
                technique_id=f"tech_meta_{category.value}_{uuid.uuid4().hex[:6]}",
                name=f"Observable {patterns[0].title()} ({category.value})",
                mechanism=f"Uses {patterns[0]} in title metadata (description len={len(description)} chars); frame-level editing unverified without video import.",
                applicable_niches=[category.value, "general"],
                editing_translation=f"Test a clear {patterns[0]} title/opening hook while preserving domain accuracy.",
            )
        )
    return hook_summary, edit_note, techniques


class PlatformConnector(ABC):
    platform: ResearchPlatform
    connector_id: str

    @abstractmethod
    def status(self) -> dict[str, Any]:
        """Describe auth eligibility, limits, and active capabilities."""

    @abstractmethod
    def fetch_public_items(self, query: str, limit: int = 5) -> list[ViralContentItem]:
        """Retrieve permitted public metadata without bypassing access controls."""


class YouTubeOfficialConnector(PlatformConnector):
    platform = ResearchPlatform.YOUTUBE
    connector_id = "youtube_official"

    def status(self) -> dict[str, Any]:
        has_key = bool(os.environ.get("YOUTUBE_API_KEY"))
        return {
            "platform": self.platform.value,
            "connector_id": self.connector_id,
            "official_api_configured": has_key,
            "public_rss_and_oembed_ready": True,
            "notes": (
                "Uses YouTube Data API v3 when YOUTUBE_API_KEY is configured; "
                "falls back to public oEmbed and official channel Atom feeds for supplied IDs/URLs."
            ),
        }

    def fetch_from_url_oembed(self, url: str, category: ContentCategory = ContentCategory.EDUCATION) -> ViralContentItem | None:
        try:
            resp = httpx.get(
                "https://www.youtube.com/oembed",
                params={"url": url, "format": "json"},
                timeout=8.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            title = str(data.get("title") or url)
            is_short = "/shorts/" in url
            plat = ResearchPlatform.YOUTUBE_SHORTS if is_short else ResearchPlatform.YOUTUBE
            hook_s, edit_s, techs = _analyze_public_packaging(
                title=title,
                description="",
                category=category,
                views=None,
                baseline_views=None,
                likes_or_ratings=None,
                is_short=is_short,
            )
            return ViralContentItem(
                item_id=f"yt_oembed_{uuid.uuid4().hex[:8]}",
                platform=plat,
                creator=str(data.get("author_name") or "Unknown Creator"),
                original_url=url,
                title=title,
                published_at=utc_now_iso()[:10],
                metrics=PublicEngagementMetrics(
                    platform=plat,
                    metric_source="official_oembed_public",
                    captured_at=utc_now_iso(),
                ),
                category=category,
                topic=title,
                opening_hook_structure=hook_s,
                editing_style=edit_s,
                narrative_structure="User-submitted YouTube reference (oEmbed metadata only; view counts require Data API v3 or import).",
                reusable_techniques=techs,
                evidence=f"Retrieved via official YouTube oEmbed endpoint ({resp.status_code}); numeric counts omitted because oEmbed does not publish views.",
                confidence=0.82,
                platform_relative_score=0.0,
                score_basis="unscored_structural_benchmark",
                provenance_type="official_api",
            )
        except Exception:
            return None

    def fetch_channel_rss(
        self,
        channel_id: str,
        limit: int = 15,
        category: ContentCategory = ContentCategory.EDUCATION,
        client: httpx.Client | None = None,
    ) -> list[ViralContentItem]:
        """Fetch official YouTube Atom feed with real view/rating counts & channel-median normalization."""
        try:
            http = client or httpx
            resp = http.get(
                "https://www.youtube.com/feeds/videos.xml",
                params={"channel_id": channel_id},
                timeout=6.0,
            )
            if resp.status_code != 200:
                return []
            root = ET.fromstring(resp.text)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "yt": "http://www.youtube.com/xml/schemas/2015",
                "media": "http://search.yahoo.com/mrss/",
            }
            author_el = root.find("atom:title", ns)
            creator = author_el.text.strip() if author_el is not None and author_el.text else channel_id

            raw_entries: list[dict[str, Any]] = []
            for entry in root.findall("atom:entry", ns)[:limit]:
                title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                vid = (entry.findtext("yt:videoId", default="", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                url = link_el.attrib.get("href", "") if link_el is not None else ""
                pub_iso = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
                if not url:
                    continue

                views: int | None = None
                rating_count: int | None = None
                desc = ""
                mg = entry.find("media:group", ns)
                if mg is not None:
                    desc = (mg.findtext("media:description", default="", namespaces=ns) or "").strip()
                    mc = mg.find("media:community", ns)
                    if mc is not None:
                        st_el = mc.find("media:statistics", ns)
                        if st_el is not None and "views" in st_el.attrib:
                            try:
                                views = int(st_el.attrib["views"])
                            except ValueError:
                                views = None
                        sr_el = mc.find("media:starRating", ns)
                        if sr_el is not None and "count" in sr_el.attrib:
                            try:
                                rating_count = int(sr_el.attrib["count"])
                            except ValueError:
                                rating_count = None

                raw_entries.append(
                    {
                        "vid": vid,
                        "title": title or url,
                        "url": url,
                        "pub_iso": pub_iso,
                        "views": views,
                        "rating_count": rating_count,
                        "desc": desc,
                    }
                )

            valid_views = [int(r["views"]) for r in raw_entries if r["views"] is not None and r["views"] > 0]
            baseline_views = int(statistics.median(valid_views)) if valid_views else None

            items: list[ViralContentItem] = []
            now_iso = utc_now_iso()
            for r in raw_entries:
                v = r["views"]
                is_short = "/shorts/" in r["url"]
                plat = ResearchPlatform.YOUTUBE_SHORTS if is_short else ResearchPlatform.YOUTUBE
                ratio = round(v / baseline_views, 2) if (v is not None and baseline_views and baseline_views > 0) else None
                days_old = _days_since_iso(r["pub_iso"])
                vpd = round(v / days_old, 1) if v is not None else None
                hook_s, edit_s, techs = _analyze_public_packaging(
                    title=r["title"],
                    description=r["desc"],
                    category=category,
                    views=v,
                    baseline_views=baseline_views,
                    likes_or_ratings=r["rating_count"],
                    is_short=is_short,
                )
                vid_slug = r["vid"] or uuid.uuid4().hex[:8]
                items.append(
                    ViralContentItem(
                        item_id=f"yt_rss_{vid_slug}",
                        platform=plat,
                        creator=creator,
                        original_url=r["url"],
                        title=r["title"],
                        published_at=r["pub_iso"][:10] or now_iso[:10],
                        metrics=PublicEngagementMetrics(
                            platform=plat,
                            metric_source="official_public_rss",
                            views=v,
                            likes=r["rating_count"],
                            channel_baseline_views=baseline_views,
                            audience_normalized_ratio=ratio,
                            views_per_day=vpd,
                            published_at=r["pub_iso"],
                            captured_at=now_iso,
                        ),
                        category=category,
                        topic=r["title"],
                        opening_hook_structure=hook_s,
                        editing_style=edit_s,
                        narrative_structure=(
                            f"Breakout outlier ({ratio:.2f}x channel median)"
                            if ratio is not None and ratio >= 1.5
                            else f"Channel baseline comparison ({ratio:.2f}x median)" if ratio is not None else "Public RSS entry"
                        ),
                        reusable_techniques=techs,
                        evidence=(
                            f"Official YouTube Atom RSS ({channel_id}): views={v:,}, ratings={r['rating_count']}, "
                            f"channel_15vid_median={baseline_views:,}, breakout_ratio={ratio}x."
                            if v is not None and baseline_views is not None
                            else f"Official YouTube Atom feed ({channel_id})."
                        ),
                        confidence=0.92 if v is not None else 0.78,
                        platform_relative_score=0.0,  # Computed across full platform cohort in ResearchEngine
                        score_basis="audience_normalized_breakout" if ratio is not None else "unscored_structural_benchmark",
                        provenance_type="official_api",
                    )
                )
            return items
        except Exception:
            return []

    def fetch_public_items(
        self,
        query: str,
        limit: int = 50,
        category: ContentCategory = ContentCategory.EDUCATION,
    ) -> list[ViralContentItem]:
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            return []
        try:
            video_ids: list[str] = []
            page_token: str | None = None
            with httpx.Client(timeout=10.0) as client:
                while len(video_ids) < limit:
                    params: dict[str, Any] = {
                        "part": "snippet",
                        "q": query,
                        "type": "video",
                        "maxResults": min(50, limit - len(video_ids)),
                        "order": "viewCount",
                        "key": api_key,
                    }
                    if page_token:
                        params["pageToken"] = page_token
                    search_resp = client.get("https://www.googleapis.com/youtube/v3/search", params=params)
                    if search_resp.status_code != 200:
                        break
                    sjson = search_resp.json()
                    for item in sjson.get("items", []):
                        vid = item.get("id", {}).get("videoId")
                        if vid and vid not in video_ids:
                            video_ids.append(vid)
                    page_token = sjson.get("nextPageToken")
                    if not page_token:
                        break

                if not video_ids:
                    return []

                videos_raw: list[dict[str, Any]] = []
                for start in range(0, len(video_ids), 50):
                    chunk = video_ids[start : start + 50]
                    stats_resp = client.get(
                        "https://www.googleapis.com/youtube/v3/videos",
                        params={"part": "snippet,statistics,contentDetails", "id": ",".join(chunk), "key": api_key},
                    )
                    if stats_resp.status_code == 200:
                        videos_raw.extend(stats_resp.json().get("items", []))

                channel_subs: dict[str, int] = {}
                cids = sorted({str(v.get("snippet", {}).get("channelId") or "") for v in videos_raw} - {""})
                for start in range(0, len(cids), 50):
                    c_chunk = cids[start : start + 50]
                    ch_resp = client.get(
                        "https://www.googleapis.com/youtube/v3/channels",
                        params={"part": "statistics", "id": ",".join(c_chunk), "key": api_key},
                    )
                    if ch_resp.status_code == 200:
                        for ch in ch_resp.json().get("items", []):
                            sc = ch.get("statistics", {}).get("subscriberCount")
                            if sc is not None:
                                channel_subs[ch["id"]] = int(sc)

            out: list[ViralContentItem] = []
            now_iso = utc_now_iso()
            for v in videos_raw:
                snip = v.get("snippet", {})
                stats = v.get("statistics", {})
                vid = v["id"]
                cid = str(snip.get("channelId") or "")
                subs = channel_subs.get(cid)
                views = int(stats["viewCount"]) if "viewCount" in stats else None
                likes = int(stats["likeCount"]) if "likeCount" in stats else None
                comments = int(stats["commentCount"]) if "commentCount" in stats else None
                pub_iso = str(snip.get("publishedAt") or "")
                ratio = round(views / subs, 2) if (views is not None and subs and subs > 0) else None
                vpd = round(views / _days_since_iso(pub_iso), 1) if views is not None else None
                title = str(snip.get("title") or vid)
                hook_s, edit_s, techs = _analyze_public_packaging(
                    title=title,
                    description=str(snip.get("description") or ""),
                    category=category,
                    views=views,
                    baseline_views=subs,
                    likes_or_ratings=likes,
                    is_short=False,
                )
                out.append(
                    ViralContentItem(
                        item_id=f"yt_{vid}",
                        platform=ResearchPlatform.YOUTUBE,
                        creator=str(snip.get("channelTitle") or "Unknown"),
                        original_url=f"https://www.youtube.com/watch?v={vid}",
                        title=title,
                        published_at=pub_iso[:10],
                        metrics=PublicEngagementMetrics(
                            platform=ResearchPlatform.YOUTUBE,
                            metric_source="official_api",
                            views=views,
                            likes=likes,
                            comments=comments,
                            subscribers_or_followers_at_capture=subs,
                            audience_normalized_ratio=ratio,
                            views_per_day=vpd,
                            published_at=pub_iso,
                            captured_at=now_iso,
                        ),
                        category=category,
                        topic=query,
                        opening_hook_structure=hook_s,
                        editing_style=edit_s,
                        narrative_structure=f"YouTube Data API v3 record ({ratio}x subscribers)" if ratio else "YouTube Data API v3 record",
                        reusable_techniques=techs,
                        evidence=(
                            f"Official YouTube Data API v3: query='{query}', views={views}, subscribers={subs}, ratio={ratio}x."
                        ),
                        confidence=0.96,
                        platform_relative_score=0.0,
                        score_basis="audience_normalized_breakout" if ratio is not None else "within_platform_normalized",
                        provenance_type="official_api",
                    )
                )
            return out
        except Exception:
            return []

    def discover_multi_niche_corpus(
        self,
        target_count: int = 100,
        categories: list[ContentCategory] | None = None,
        query: str | None = None,
        channels: list[tuple[str, ContentCategory, str]] | None = None,
    ) -> list[ViralContentItem]:
        """Discover real public videos across multiple niches via official RSS + Data API v3."""
        pool = channels or DEFAULT_RESEARCH_CHANNELS
        if categories:
            cat_set = {c.value if isinstance(c, ContentCategory) else str(c) for c in categories}
            filtered = [c for c in pool if c[1].value in cat_set]
            if filtered:
                pool = filtered

        per_channel_limit = 15 if target_count >= len(pool) * 10 else max(6, min(15, (target_count // max(1, len(pool))) + 3))
        collected: dict[str, ViralContentItem] = {}

        if query:
            for it in self.fetch_public_items(query=query, limit=min(50, target_count)):
                collected[it.item_id] = it

        def _worker(spec: tuple[str, ContentCategory, str]) -> list[ViralContentItem]:
            _name, cat, cid = spec
            with httpx.Client(timeout=6.0) as client:
                return self.fetch_channel_rss(channel_id=cid, limit=per_channel_limit, category=cat, client=client)

        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(_worker, spec) for spec in pool]
            for fut in as_completed(futures):
                try:
                    for it in fut.result():
                        collected[it.item_id] = it
                except Exception:
                    continue

        items = list(collected.values())
        return items[: max(target_count, len(items))] if len(items) < target_count else items[:target_count]


class GatedOfficialConnector(PlatformConnector):
    """Compliant connector stub for platforms requiring reviewed developer apps (TikTok/Instagram/Threads)."""

    def __init__(self, platform: ResearchPlatform, connector_id: str, env_token_name: str, requirements: str) -> None:
        self.platform = platform
        self.connector_id = connector_id
        self.env_token_name = env_token_name
        self.requirements = requirements

    def status(self) -> dict[str, Any]:
        return {
            "platform": self.platform.value,
            "connector_id": self.connector_id,
            "official_api_configured": bool(os.environ.get(self.env_token_name)),
            "supports_manual_url_and_json_import": True,
            "requirements": self.requirements,
        }

    def fetch_public_items(self, query: str, limit: int = 5) -> list[ViralContentItem]:
        # Intentionally does not scrape or bypass auth walls when official token is absent.
        return []


def import_explicit_item(
    platform: ResearchPlatform,
    creator: str,
    original_url: str,
    title: str,
    category: ContentCategory,
    topic: str,
    opening_hook_structure: str,
    editing_style: str,
    narrative_structure: str,
    views: int | None = None,
    likes: int | None = None,
    comments: int | None = None,
    shares_or_reposts: int | None = None,
    subscribers_or_followers: int | None = None,
    duration_sec: float | None = None,
    notes: str = "",
    techniques: list[dict[str, Any]] | None = None,
) -> ViralContentItem:
    """Create a provenance-tracked ViralContentItem from an explicit user link or dataset import."""
    reusable = [
        ReusableTechnique(
            technique_id=f"tech_{uuid.uuid4().hex[:6]}",
            name=str(t.get("name", "Imported Technique")),
            mechanism=str(t.get("mechanism", "")),
            applicable_niches=list(t.get("applicable_niches", ["general"])),
            editing_translation=str(t.get("editing_translation", "")),
        )
        for t in (techniques or [])
    ]
    ratio = (
        round(views / subscribers_or_followers, 2)
        if views is not None and subscribers_or_followers and subscribers_or_followers > 0
        else None
    )
    return ViralContentItem(
        item_id=f"imp_{uuid.uuid4().hex[:10]}",
        platform=platform,
        creator=creator,
        original_url=original_url,
        title=title,
        published_at=utc_now_iso()[:10],
        duration_sec=duration_sec,
        metrics=PublicEngagementMetrics(
            platform=platform,
            metric_source="user_supplied",
            views=views,
            likes=likes,
            comments=comments,
            shares_or_reposts=shares_or_reposts,
            subscribers_or_followers_at_capture=subscribers_or_followers,
            audience_normalized_ratio=ratio,
            captured_at=utc_now_iso(),
        ),
        category=category,
        topic=topic,
        opening_hook_structure=opening_hook_structure,
        editing_style=editing_style,
        narrative_structure=narrative_structure,
        reusable_techniques=reusable,
        evidence=notes or "Explicitly supplied by creator/operator.",
        confidence=0.9 if views is not None else 0.78,
        platform_relative_score=0.0,
        score_basis="audience_normalized_breakout" if ratio is not None else (
            "within_platform_normalized" if views is not None and views > 0 else "unscored_structural_benchmark"
        ),
        provenance_type="explicit_import",
    )


def import_items_from_json(json_path: Path) -> list[ViralContentItem]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else raw.get("items", [])
    out: list[ViralContentItem] = []
    for r in records:
        out.append(
            import_explicit_item(
                platform=ResearchPlatform(r["platform"]),
                creator=r.get("creator", "Unknown"),
                original_url=r["original_url"],
                title=r["title"],
                category=ContentCategory(r.get("category", "education")),
                topic=r.get("topic", r["title"]),
                opening_hook_structure=r.get("opening_hook_structure", ""),
                editing_style=r.get("editing_style", ""),
                narrative_structure=r.get("narrative_structure", ""),
                views=r.get("views"),
                likes=r.get("likes"),
                comments=r.get("comments"),
                shares_or_reposts=r.get("shares_or_reposts"),
                duration_sec=r.get("duration_sec"),
                notes=r.get("notes", f"Imported from {json_path.name}"),
                techniques=r.get("techniques"),
            )
        )
    return out


def infer_platform_from_url(url: str) -> ResearchPlatform:
    u = url.lower()
    if "youtube.com/shorts" in u or "youtu.be" in u and "shorts" in u:
        return ResearchPlatform.YOUTUBE_SHORTS
    if "youtube.com" in u or "youtu.be" in u:
        return ResearchPlatform.YOUTUBE
    if "tiktok.com" in u:
        return ResearchPlatform.TIKTOK
    if "instagram.com/reel" in u:
        return ResearchPlatform.INSTAGRAM_REELS
    if "instagram.com" in u:
        return ResearchPlatform.INSTAGRAM_POST
    if "threads.net" in u or "threads.com" in u:
        return ResearchPlatform.THREADS
    return ResearchPlatform.YOUTUBE


_MEME_TOPIC_HINTS: dict[str, tuple[str, ...]] = {
    "drake": ("vs", "instead", "rather", "choice", "focus", "discipline", "ожидани", "реальн", "таңда", "лучше", "көрсе"),
    "buttons": ("decision", "choice", "hard", "major", "quiz", "internship", "экзамен", "семестр", "выбор", "мамандық"),
    "distracted": ("distract", "focus", "habit", "scroll", "hobby", "отвлек", "привычк", "хобби", "фокус"),
    "change my mind": ("opinion", "expect", "goal", "mind", "дум", "цел", "мақсат", "ой"),
    "harold": ("pain", "stress", "struggle", "fail", "problem", "hard", "тяжел", "сложн", "қиын", "стресс", "quiz"),
    "panik": ("quiz", "exam", "deadline", "semester", "money", "экзамен", "семестр", "деньги", "ақша", "стипенд"),
    "stonks": ("money", "earn", "career", "gpa", "dean", "salary", "зараб", "деньг", "ақша", "стипенд", "бизнес"),
    "disaster": ("burnout", "crash", "bug", "mistake", "problem", "ошибк", "проблем", "қате"),
    "skeleton": ("wait", "year", "long", "later", "time capsule", "жд", "год", "жыл", "капсул"),
    "brain": ("learn", "level", "skill", "code", "algorithm", "data structure", "уч", "код", "программ", "біл"),
    "pigeon": ("confused", "why", "what", "expect", "почему", "непонят", "сұрақ"),
    "batman": ("reality", "truth", "slap", "реальн", "правда", "шындық"),
    "always has been": ("realize", "secret", "truth", "понял", "оказывается"),
    "escobar": ("alone", "wait", "miss", "скуч", "один", "жд"),
}


class MemeTrendConnector(PlatformConnector):
    """Fetch live trending meme templates & viral internet formats from public endpoints."""

    platform = ResearchPlatform.INSTAGRAM_POST
    connector_id = "imgflip_reddit_memes"

    def status(self) -> dict[str, Any]:
        return {
            "platform": self.platform.value,
            "connector_id": self.connector_id,
            "official_api_configured": True,
            "notes": (
                "Queries Imgflip public trending memes endpoint (api.imgflip.com/get_memes) with live "
                "template usage counts and optional public Reddit /r/memes JSON without bypassing auth walls."
            ),
        }

    def fetch_public_items(self, query: str = "", limit: int = 12) -> list[ViralContentItem]:
        items: list[ViralContentItem] = []
        raw_memes: list[dict[str, Any]] = []
        try:
            resp = httpx.get("https://api.imgflip.com/get_memes", timeout=8.0)
            if resp.status_code == 200:
                payload = resp.json()
                raw_memes = list((payload.get("data") or {}).get("memes") or [])
        except Exception:
            raw_memes = []

        if raw_memes:
            caption_counts = [int(m.get("captions") or 0) for m in raw_memes if int(m.get("captions") or 0) > 0]
            median_caps = float(statistics.median(caption_counts)) if caption_counts else 100000.0
            q_low = (query or "").lower()
            q_tokens = [tok for tok in re.split(r"[\s,.;:!?]+", q_low) if len(tok) >= 2]

            scored_memes: list[tuple[float, dict[str, Any]]] = []
            for idx, m in enumerate(raw_memes):
                name = str(m.get("name") or "").strip()
                url = str(m.get("url") or "").strip()
                caps = int(m.get("captions") or 0)
                if not name or not url:
                    continue
                low_name = name.lower()
                rel_bonus = 0.0
                for tok in q_tokens:
                    if tok in low_name:
                        rel_bonus += 0.45
                for stem, hint_words in _MEME_TOPIC_HINTS.items():
                    if stem in low_name and any(hw in q_low for hw in hint_words):
                        rel_bonus += 0.55
                pop_score = (caps / max(1.0, median_caps)) * 0.4 + max(0.0, (40 - idx) / 80.0)
                scored_memes.append((rel_bonus + pop_score, m))

            scored_memes.sort(key=lambda x: x[0], reverse=True)
            for _sc, m in scored_memes[: max(1, limit)]:
                meme_id = str(m.get("id") or uuid.uuid4().hex[:6])
                name = str(m.get("name") or "Trending Meme").strip()
                url = str(m.get("url") or "").strip()
                caps = int(m.get("captions") or 0)
                ratio = round(caps / max(1.0, median_caps), 2) if caps > 0 else 1.0
                tech = ReusableTechnique(
                    technique_id=f"tech_meme_{meme_id}",
                    name=f"Meme Cutaway Pop-up: {name}",
                    mechanism=(
                        f"Uses live trending meme template '{name}' ({caps:,} documented Imgflip caption instances, "
                        f"{ratio:.2f}x cohort median) as a 2.5–3.5s visual reaction punch."
                    ),
                    applicable_niches=["personal_storytelling", "comedy", "internet_trends", "youtube", "shorts"],
                    editing_translation=(
                        f"Pop in '{name}' as an animated PIP card with a 3-word reaction punchline + ui_pop/vine_boom SFX."
                    ),
                )
                items.append(
                    ViralContentItem(
                        item_id=f"meme_imgflip_{meme_id}",
                        platform=ResearchPlatform.INSTAGRAM_POST,
                        creator="Imgflip Live Trends",
                        original_url=url,
                        title=f"{name} (Trending Meme Format)",
                        published_at=utc_now_iso()[:10],
                        metrics=PublicEngagementMetrics(
                            platform=ResearchPlatform.INSTAGRAM_POST,
                            metric_source="imgflip_live_api",
                            views=caps if caps > 0 else None,
                            channel_baseline_views=int(median_caps) if median_caps > 0 else None,
                            audience_normalized_ratio=ratio,
                            captured_at=utc_now_iso(),
                        ),
                        category=ContentCategory.INTERNET_TRENDS,
                        topic=name,
                        opening_hook_structure=f"Visual punchline card anchored by trending format '{name}'.",
                        editing_style="zoomer_vlog reaction cutaway (animated PIP card + SFX punch)",
                        narrative_structure="Relatable setup -> instant meme reaction punchline -> spoken beat continuation.",
                        visual_effects=["spring_pop_in", "meme_reaction_card", "sfx_punch"],
                        reusable_techniques=[tech],
                        evidence=(
                            f"Live response from api.imgflip.com/get_memes: captions_count={caps:,} "
                            f"(template {m.get('width')}x{m.get('height')}, median_cohort={int(median_caps):,})."
                        ),
                        confidence=0.92,
                        platform_relative_score=0.0,
                        score_basis="audience_normalized_breakout",
                        provenance_type="live_public_api",
                    )
                )

        if len(items) < limit:
            try:
                rr = httpx.get(
                    "https://www.reddit.com/r/memes/hot.json?limit=8",
                    headers={"User-Agent": "SymposiumStudio/0.2 (open-source video research)"},
                    timeout=5.0,
                )
                if rr.status_code == 200:
                    children = (rr.json().get("data") or {}).get("children") or []
                    for c in children:
                        if len(items) >= limit:
                            break
                        p = c.get("data") or {}
                        p_url = str(p.get("url") or "")
                        ups = int(p.get("ups") or p.get("score") or 0)
                        if not re.search(r"\.(png|jpe?g|webp)(\?.*)?$", p_url, re.I) or ups <= 0:
                            continue
                        pid = str(p.get("id") or uuid.uuid4().hex[:6])
                        title = str(p.get("title") or "Reddit Trending Meme")[:120]
                        items.append(
                            ViralContentItem(
                                item_id=f"meme_reddit_{pid}",
                                platform=ResearchPlatform.INSTAGRAM_POST,
                                creator=f"r/memes ({p.get('author', 'community')})",
                                original_url=p_url,
                                title=title,
                                published_at=utc_now_iso()[:10],
                                metrics=PublicEngagementMetrics(
                                    platform=ResearchPlatform.INSTAGRAM_POST,
                                    metric_source="reddit_public_json",
                                    views=ups,
                                    likes=ups,
                                    comments=int(p.get("num_comments") or 0),
                                    captured_at=utc_now_iso(),
                                ),
                                category=ContentCategory.INTERNET_TRENDS,
                                topic=title,
                                opening_hook_structure="Current viral post reaction meme from r/memes hot feed.",
                                editing_style="zoomer_vlog reaction cutaway",
                                narrative_structure="Setup -> relatable visual reaction.",
                                reusable_techniques=[
                                    ReusableTechnique(
                                        technique_id=f"tech_rd_{pid}",
                                        name="Current Community Meme Reference",
                                        mechanism=f"Top trending Reddit /r/memes post ({ups:,} upvotes) ready for visual callback.",
                                        applicable_niches=["internet_trends", "comedy", "personal_storytelling"],
                                        editing_translation="Overlay cleanly in upper-right PIP during relatable story beat.",
                                    )
                                ],
                                evidence=f"Public Reddit /r/memes JSON score={ups:,}, comments={int(p.get('num_comments') or 0):,}.",
                                confidence=0.88,
                                platform_relative_score=0.0,
                                score_basis="within_platform_normalized",
                                provenance_type="live_public_api",
                            )
                        )
            except Exception:
                pass
        return items[:limit]


def download_trending_meme_assets(
    items: list[ViralContentItem],
    dest_dir: Path,
    max_count: int = 4,
) -> list[dict[str, Any]]:
    """Download and validate trending meme images locally into workspace so overlays render deterministically."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for it in items:
        if len(downloaded) >= max_count:
            break
        url = (it.original_url or "").strip()
        if not url or url in seen_urls:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if not (
            re.search(r"\.(png|jpe?g|webp)(\?.*)?$", url, re.I)
            or "imgflip.com" in parsed.netloc.lower()
            or "redd.it" in parsed.netloc.lower()
        ):
            continue
        seen_urls.add(url)
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", it.item_id)[:48]
        out_png = dest_dir / f"{safe_id}.png"
        try:
            if not out_png.exists():
                resp = httpx.get(url, timeout=8.0, follow_redirects=True)
                if resp.status_code != 200 or len(resp.content) < 512 or len(resp.content) > 8_000_000:
                    continue
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img.thumbnail((960, 720), Image.Resampling.LANCZOS)
                img.save(out_png, format="PNG")
            if out_png.is_file() and out_png.stat().st_size > 512:
                downloaded.append(
                    {
                        "item_id": it.item_id,
                        "title": it.title,
                        "topic": it.topic,
                        "original_url": url,
                        "image_path": str(out_png.resolve()),
                        "usage_count": it.metrics.views,
                        "metric_source": it.metrics.metric_source,
                    }
                )
        except Exception:
            continue
    return downloaded


def build_default_connectors() -> list[PlatformConnector]:
    return [
        YouTubeOfficialConnector(),
        MemeTrendConnector(),
        GatedOfficialConnector(
            ResearchPlatform.TIKTOK,
            "tiktok_official",
            "TIKTOK_RESEARCH_TOKEN",
            "Requires TikTok Research API or Display API OAuth approval; supports manual links & JSON imports.",
        ),
        GatedOfficialConnector(
            ResearchPlatform.INSTAGRAM_REELS,
            "instagram_graph",
            "INSTAGRAM_ACCESS_TOKEN",
            "Requires Instagram Professional (Business/Creator) + Meta App review; supports manual links & JSON imports.",
        ),
        GatedOfficialConnector(
            ResearchPlatform.THREADS,
            "threads_official",
            "THREADS_ACCESS_TOKEN",
            "Requires Meta Threads Use Case + approved scopes; supports manual links & JSON imports.",
        ),
    ]
