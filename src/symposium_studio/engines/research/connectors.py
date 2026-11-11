"""Modular platform research connectors using official APIs, public feeds, and explicit imports.

Strictly enforces platform compliance: never bypasses auth gates, CAPTCHAs, or
private endpoints, and never fabricates private metrics.
"""

from __future__ import annotations

import json
import os
import re
import uuid
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from symposium_studio.models.research import (
    ContentCategory,
    PublicEngagementMetrics,
    ResearchPlatform,
    ReusableTechnique,
    ViralContentItem,
)
from symposium_studio.models.project import utc_now_iso


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
            is_short = "/shorts/" in url
            plat = ResearchPlatform.YOUTUBE_SHORTS if is_short else ResearchPlatform.YOUTUBE
            return ViralContentItem(
                item_id=f"yt_oembed_{uuid.uuid4().hex[:8]}",
                platform=plat,
                creator=str(data.get("author_name") or "Unknown Creator"),
                original_url=url,
                title=str(data.get("title") or url),
                published_at=utc_now_iso()[:10],
                metrics=PublicEngagementMetrics(
                    platform=plat,
                    metric_source="official_oembed_public",
                    captured_at=utc_now_iso(),
                ),
                category=category,
                topic=str(data.get("title") or "Supplied YouTube reference"),
                opening_hook_structure="Pending local video inspection or transcript review.",
                editing_style="Requires creator analysis or supplied notes.",
                narrative_structure="User-submitted YouTube reference.",
                evidence=f"Retrieved via official YouTube oEmbed endpoint ({resp.status_code}).",
                confidence=0.85,
                platform_relative_score=50.0,
                provenance_type="official_api",
            )
        except Exception:
            return None

    def fetch_channel_rss(self, channel_id: str, limit: int = 5) -> list[ViralContentItem]:
        try:
            resp = httpx.get(
                "https://www.youtube.com/feeds/videos.xml",
                params={"channel_id": channel_id},
                timeout=8.0,
            )
            if resp.status_code != 200:
                return []
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
            author_el = root.find("atom:title", ns)
            creator = author_el.text if author_el is not None and author_el.text else channel_id
            items: list[ViralContentItem] = []
            for entry in root.findall("atom:entry", ns)[:limit]:
                title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                url = link_el.attrib.get("href", "") if link_el is not None else ""
                pub = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
                if not url:
                    continue
                items.append(
                    ViralContentItem(
                        item_id=f"yt_rss_{uuid.uuid4().hex[:8]}",
                        platform=ResearchPlatform.YOUTUBE,
                        creator=creator,
                        original_url=url,
                        title=title or url,
                        published_at=pub or utc_now_iso()[:10],
                        metrics=PublicEngagementMetrics(
                            platform=ResearchPlatform.YOUTUBE,
                            metric_source="official_public_rss",
                            published_at=pub,
                            captured_at=utc_now_iso(),
                        ),
                        category=ContentCategory.EDUCATION,
                        topic=title,
                        opening_hook_structure="Public RSS feed entry; watch/import transcript for hook breakdown.",
                        editing_style="Public RSS metadata.",
                        narrative_structure="Public RSS feed entry.",
                        evidence=f"Fetched from official YouTube public Atom feed for channel {channel_id}.",
                        confidence=0.85,
                        platform_relative_score=50.0,
                        provenance_type="official_api",
                    )
                )
            return items
        except Exception:
            return []

    def fetch_public_items(self, query: str, limit: int = 5) -> list[ViralContentItem]:
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            return []
        try:
            search_resp = httpx.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={"part": "snippet", "q": query, "type": "video", "maxResults": min(limit, 10), "order": "viewCount", "key": api_key},
                timeout=10.0,
            )
            if search_resp.status_code != 200:
                return []
            video_ids = [item["id"]["videoId"] for item in search_resp.json().get("items", []) if item.get("id", {}).get("videoId")]
            if not video_ids:
                return []
            stats_resp = httpx.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "snippet,statistics,contentDetails", "id": ",".join(video_ids), "key": api_key},
                timeout=10.0,
            )
            if stats_resp.status_code != 200:
                return []
            out: list[ViralContentItem] = []
            for v in stats_resp.json().get("items", []):
                snip = v.get("snippet", {})
                stats = v.get("statistics", {})
                vid = v["id"]
                out.append(
                    ViralContentItem(
                        item_id=f"yt_{vid}",
                        platform=ResearchPlatform.YOUTUBE,
                        creator=str(snip.get("channelTitle") or "Unknown"),
                        original_url=f"https://www.youtube.com/watch?v={vid}",
                        title=str(snip.get("title") or vid),
                        published_at=str(snip.get("publishedAt") or "")[:10],
                        metrics=PublicEngagementMetrics(
                            platform=ResearchPlatform.YOUTUBE,
                            metric_source="official_api",
                            views=int(stats["viewCount"]) if "viewCount" in stats else None,
                            likes=int(stats["likeCount"]) if "likeCount" in stats else None,
                            comments=int(stats["commentCount"]) if "commentCount" in stats else None,
                            published_at=str(snip.get("publishedAt") or ""),
                            captured_at=utc_now_iso(),
                        ),
                        category=ContentCategory.EDUCATION,
                        topic=query,
                        opening_hook_structure="Retrieved via YouTube Data API v3; run media analysis if local clip imported.",
                        editing_style="API metadata record.",
                        narrative_structure="API metadata record.",
                        evidence=f"Official YouTube Data API v3 videos.list response for query '{query}'.",
                        confidence=0.95,
                        platform_relative_score=75.0,
                        provenance_type="official_api",
                    )
                )
            return out
        except Exception:
            return []


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
        platform_relative_score=50.0,
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


def build_default_connectors() -> list[PlatformConnector]:
    return [
        YouTubeOfficialConnector(),
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
