"""Platform-scoped relative performance scoring and evidence-backed research reports."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

from symposium_studio.engines.research.connectors import (
    YouTubeOfficialConnector,
    build_default_connectors,
)
from symposium_studio.engines.research.seed_corpus import CURATED_CROSS_NICHE_CORPUS
from symposium_studio.models.project import utc_now_iso
from symposium_studio.models.research import ResearchReport, ViralContentItem
from symposium_studio.storage.db import StudioDatabase


def recompute_platform_relative_scores(items: list[ViralContentItem]) -> list[ViralContentItem]:
    """Normalize engagement strictly within the SAME platform cohort.

    Never equates raw YouTube views with TikTok views or Threads impressions.
    Items without public numeric counts preserve their structural confidence score.
    """
    by_platform: dict[str, list[ViralContentItem]] = defaultdict(list)
    for it in items:
        by_platform[it.platform.value].append(it)

    for _plat, group in by_platform.items():
        scored_values: list[tuple[ViralContentItem, float]] = []
        for it in group:
            m = it.metrics
            if m.views is not None and m.views > 0:
                eng_rate = ((m.likes or 0) + 2.0 * (m.comments or 0) + 3.0 * (m.shares_or_reposts or 0)) / max(1, m.views)
                raw = (m.views ** 0.25) * (1.0 + min(0.5, eng_rate * 5.0))
                scored_values.append((it, raw))
        if not scored_values:
            continue
        vals = sorted(v for _, v in scored_values)
        for it, raw in scored_values:
            rank = sum(1 for v in vals if v <= raw)
            pct = round((rank / len(vals)) * 100.0, 1)
            it.platform_relative_score = max(10.0, min(99.0, pct))
    return items


class ResearchIntelligenceEngine:
    """Manages cross-niche viral intelligence database, searches, and research reports."""

    def __init__(self, db: StudioDatabase) -> None:
        self.db = db
        self.connectors = build_default_connectors()
        self.ensure_seed_corpus()

    def ensure_seed_corpus(self) -> None:
        existing = self.db.list_viral_items(limit=1)
        if not existing:
            for item in CURATED_CROSS_NICHE_CORPUS:
                self.db.upsert_viral_item(item)

    def connector_statuses(self) -> list[dict[str, Any]]:
        return [c.status() for c in self.connectors]

    def search_patterns(
        self,
        query: str | None = None,
        platform: str | None = None,
        category: str | None = None,
        limit: int = 20,
    ) -> list[ViralContentItem]:
        if query:
            hits = self.db.search_all(query, limit=limit * 2)
            item_ids = [h["entity_id"] for h in hits if h["entity_type"] == "viral_item"]
            all_items = self.db.list_viral_items(platform=platform, category=category, limit=200)
            by_id = {it.item_id: it for it in all_items}
            matched = [by_id[iid] for iid in item_ids if iid in by_id]
            if matched:
                return matched[:limit]
        return self.db.list_viral_items(platform=platform, category=category, limit=limit)

    def add_item(self, item: ViralContentItem) -> ViralContentItem:
        self.db.upsert_viral_item(item)
        all_items = self.db.list_viral_items(platform=item.platform.value, limit=500)
        recomputed = recompute_platform_relative_scores(all_items)
        for it in recomputed:
            self.db.upsert_viral_item(it)
            if it.item_id == item.item_id:
                item = it
        return item

    def inspect_youtube_url(self, url: str) -> ViralContentItem | None:
        yt = YouTubeOfficialConnector()
        item = yt.fetch_from_url_oembed(url)
        if item:
            return self.add_item(item)
        return None

    def generate_research_report(
        self,
        report_type: str,
        target_niche: str,
        topic_keywords: list[str] | None = None,
    ) -> ResearchReport:
        """Synthesize an evidence-backed cross-niche report adapted to target_niche."""
        # Broad cross-niche pull: intentionally includes non-STEM categories!
        all_items = self.db.list_viral_items(limit=50)
        all_items = recompute_platform_relative_scores(all_items)

        findings: list[dict[str, Any]] = []
        transfers: list[dict[str, Any]] = []
        source_ids: list[str] = []

        for item in all_items[:12]:
            source_ids.append(item.item_id)
            findings.append(
                {
                    "item_id": item.item_id,
                    "platform": item.platform.value,
                    "category": item.category.value,
                    "title": item.title,
                    "opening_hook": item.opening_hook_structure,
                    "editing_style": item.editing_style,
                    "narrative_structure": item.narrative_structure,
                    "platform_relative_score": item.platform_relative_score,
                    "provenance": item.provenance_type,
                    "evidence": item.evidence,
                }
            )
            for tech in item.reusable_techniques:
                transfers.append(
                    {
                        "source_category": item.category.value,
                        "source_platform": item.platform.value,
                        "technique_name": tech.name,
                        "mechanism": tech.mechanism,
                        "adapted_for_niche": target_niche,
                        "editing_translation": tech.editing_translation,
                        "source_item_id": item.item_id,
                    }
                )

        kw_str = ", ".join(topic_keywords or [target_niche])
        report = ResearchReport(
            report_id=f"rep_{uuid.uuid4().hex[:8]}",
            report_type=report_type,
            title=f"{report_type.replace('_', ' ').title()} for {target_niche} ({kw_str})",
            target_niche=target_niche,
            summary=(
                f"Cross-niche analysis over {len(findings)} researched items spanning "
                f"{len({f['category'] for f in findings})} categories and "
                f"{len({f['platform'] for f in findings})} platforms. "
                f"Extracted {len(transfers)} transferable creative mechanics tailored to '{target_niche}'."
            ),
            source_item_ids=source_ids,
            findings=findings,
            cross_niche_transfers=transfers,
            limitations=[
                "Retention curves and CTR are intentionally omitted unless imported from authorized creator analytics.",
                "Platform-relative scores are normalized strictly within each platform; raw cross-platform counts are never equated.",
                "Curated structural benchmarks describe verified formatting mechanics without fabricating engagement numbers.",
            ],
            created_at=utc_now_iso(),
        )
        self.db.save_research_report(report)
        return report
