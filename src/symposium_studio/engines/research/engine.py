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
from symposium_studio.models.research import ContentCategory, ResearchReport, ViralContentItem
from symposium_studio.storage.db import StudioDatabase


def recompute_platform_relative_scores(items: list[ViralContentItem]) -> list[ViralContentItem]:
    """Normalize engagement strictly within the SAME platform cohort & audience baseline.

    Never equates raw YouTube views with TikTok views or Threads impressions.
    Unscored structural benchmarks without verified public view counts stay at 0.0
    so unverified archetypes never impersonate breakout viral hits.
    """
    by_platform: dict[str, list[ViralContentItem]] = defaultdict(list)
    for it in items:
        by_platform[it.platform.value].append(it)

    for _plat, group in by_platform.items():
        scored_values: list[tuple[ViralContentItem, float]] = []
        for it in group:
            m = it.metrics
            if m.views is None or m.views <= 0:
                it.platform_relative_score = 0.0
                it.score_basis = "unscored_structural_benchmark"
                continue
            baseline = m.subscribers_or_followers_at_capture or m.channel_baseline_views
            if m.audience_normalized_ratio is None and baseline and baseline > 0:
                m.audience_normalized_ratio = round(m.views / baseline, 2)
            ratio = m.audience_normalized_ratio if (m.audience_normalized_ratio and m.audience_normalized_ratio > 0) else 1.0
            eng_rate = ((m.likes or 0) + 2.0 * (m.comments or 0) + 3.0 * (m.shares_or_reposts or 0)) / max(1, m.views)
            vel = m.views_per_day if (m.views_per_day and m.views_per_day > 0) else float(m.views)
            raw = (max(0.05, ratio) ** 0.55) * (max(1.0, vel) ** 0.15) * (1.0 + min(0.8, eng_rate * 8.0))
            scored_values.append((it, raw))
        if not scored_values:
            continue
        vals = sorted(v for _, v in scored_values)
        for it, raw in scored_values:
            rank = sum(1 for v in vals if v <= raw)
            pct = round((rank / len(vals)) * 100.0, 1)
            it.platform_relative_score = max(10.0, min(99.0, pct))
            it.score_basis = (
                "audience_normalized_breakout"
                if it.metrics.audience_normalized_ratio is not None
                else "within_platform_normalized"
            )
    return items


class ResearchIntelligenceEngine:
    """Manages cross-niche viral intelligence database, searches, live discovery, and research reports."""

    def __init__(self, db: StudioDatabase) -> None:
        self.db = db
        self.connectors = build_default_connectors()
        self.ensure_seed_corpus()

    def ensure_seed_corpus(self) -> None:
        existing = self.db.list_viral_items(limit=50)
        if not existing:
            for item in CURATED_CROSS_NICHE_CORPUS:
                self.db.upsert_viral_item(item)
            return
        # Ensure legacy DB rows from older seeds without views never keep fake 90+ scores
        for row in existing:
            if row.metrics.views is None and row.platform_relative_score > 0.0:
                row.platform_relative_score = 0.0
                row.score_basis = "unscored_structural_benchmark"
                self.db.upsert_viral_item(row)

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

    def discover_cohort(
        self,
        target_count: int = 100,
        categories: list[str] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Fetch real public items across multiple niches/creators and score relative to channel baseline."""
        yt = YouTubeOfficialConnector()
        cat_enums = [ContentCategory(c) for c in categories] if categories else None
        discovered = yt.discover_multi_niche_corpus(
            target_count=target_count,
            categories=cat_enums,
            query=query,
        )
        for it in discovered:
            self.db.upsert_viral_item(it)

        all_items = self.db.list_viral_items(limit=600)
        recomputed = recompute_platform_relative_scores(all_items)
        for it in recomputed:
            self.db.upsert_viral_item(it)

        by_id = {it.item_id: it for it in recomputed}
        updated_discovered = [by_id[it.item_id] for it in discovered if it.item_id in by_id]
        verified_outliers = sorted(
            [it for it in updated_discovered if it.metrics.views is not None],
            key=lambda x: ((x.metrics.audience_normalized_ratio or 1.0), x.platform_relative_score),
            reverse=True,
        )
        return {
            "discovered_count": len(updated_discovered),
            "verified_metric_items_count": len(verified_outliers),
            "categories_covered": sorted({it.category.value for it in updated_discovered}),
            "creators_sampled": len({it.creator for it in updated_discovered}),
            "top_breakout_outliers": [
                {
                    "item_id": it.item_id,
                    "creator": it.creator,
                    "category": it.category.value,
                    "title": it.title,
                    "url": it.original_url,
                    "views": it.metrics.views,
                    "channel_baseline_views": it.metrics.channel_baseline_views,
                    "audience_normalized_ratio": it.metrics.audience_normalized_ratio,
                    "platform_relative_score": it.platform_relative_score,
                }
                for it in verified_outliers[:12]
            ],
        }

    def generate_research_report(
        self,
        report_type: str,
        target_niche: str,
        topic_keywords: list[str] | None = None,
    ) -> ResearchReport:
        """Synthesize an evidence-backed cross-niche report adapted to target_niche."""
        all_items = recompute_platform_relative_scores(self.db.list_viral_items(limit=300))

        # Strictly prioritize REAL public/API/imported metriced videos before structural archetypes
        verified = sorted(
            [it for it in all_items if it.metrics.views is not None and it.metrics.views > 0],
            key=lambda x: (
                (x.metrics.audience_normalized_ratio or 1.0) * 1.5 + x.platform_relative_score / 25.0
            ),
            reverse=True,
        )
        archetypes = [it for it in all_items if it.metrics.views is None]

        # Ensure diverse category representation across top selected real items
        selected_real: list[ViralContentItem] = []
        seen_creators_per_cat: dict[str, int] = defaultdict(int)
        for it in verified:
            if seen_creators_per_cat[it.category.value] < 2:
                selected_real.append(it)
                seen_creators_per_cat[it.category.value] += 1
            if len(selected_real) >= 16:
                break
        if len(selected_real) < 16:
            seen_ids = {x.item_id for x in selected_real}
            for it in verified:
                if it.item_id not in seen_ids:
                    selected_real.append(it)
                if len(selected_real) >= 16:
                    break

        combined = selected_real + archetypes[:4]
        findings: list[dict[str, Any]] = []
        transfers: list[dict[str, Any]] = []
        source_ids: list[str] = []

        for item in combined:
            source_ids.append(item.item_id)
            row: dict[str, Any] = {
                "item_id": item.item_id,
                "platform": item.platform.value,
                "category": item.category.value,
                "creator": item.creator,
                "url": item.original_url,
                "title": item.title,
                "views": item.metrics.views,
                "channel_baseline_views": item.metrics.channel_baseline_views,
                "audience_normalized_ratio": item.metrics.audience_normalized_ratio,
                "views_per_day": item.metrics.views_per_day,
                "opening_hook": item.opening_hook_structure,
                "editing_style": item.editing_style,
                "narrative_structure": item.narrative_structure,
                "platform_relative_score": item.platform_relative_score,
                "score_basis": item.score_basis,
                "provenance": item.provenance_type,
                "evidence": item.evidence,
            }
            if report_type == "hook_library":
                row["focus"] = "opening_hook"
            elif report_type in {"editing_techniques", "viral_formats"}:
                row["focus"] = "visual_effects_and_pacing"
                row["visual_effects"] = item.visual_effects
            elif report_type == "audience_interests":
                row["focus"] = "topic_breakout_demand"
            findings.append(row)

            for tech in item.reusable_techniques:
                transfers.append(
                    {
                        "source_category": item.category.value,
                        "source_platform": item.platform.value,
                        "source_creator": item.creator,
                        "source_url": item.original_url,
                        "audience_normalized_ratio": item.metrics.audience_normalized_ratio,
                        "technique_name": tech.name,
                        "mechanism": tech.mechanism,
                        "adapted_for_niche": target_niche,
                        "editing_translation": tech.editing_translation,
                        "source_item_id": item.item_id,
                    }
                )

        kw_str = ", ".join(topic_keywords or [target_niche])
        cat_count = len({f["category"] for f in findings})
        plat_count = len({f["platform"] for f in findings})
        summary = (
            f"Analyzed {len(verified)} metriced public videos and {len(archetypes)} unscored structural references "
            f"across {cat_count} categories and {plat_count} platforms; highlighted top {len(selected_real)} "
            f"audience-normalized items into {len(transfers)} transferable mechanics for '{target_niche}'."
            if verified
            else f"No live metriced public videos ingested yet (using {len(archetypes)} unscored structural benchmarks). "
            f"Run Live Discovery (`discover_cohort`) or import creator links to ground scores in real views and baseline ratios."
        )
        report = ResearchReport(
            report_id=f"rep_{uuid.uuid4().hex[:8]}",
            report_type=report_type,
            title=f"{report_type.replace('_', ' ').title()} for {target_niche} ({kw_str})",
            target_niche=target_niche,
            summary=summary,
            source_item_ids=source_ids,
            findings=findings,
            cross_niche_transfers=transfers,
            limitations=[
                "Retention curves and CTR are intentionally omitted unless imported from authorized creator analytics.",
                "Platform-relative scores are computed strictly within each platform; unscored structural archetypes are pinned to 0.0.",
                "Public Atom RSS metadata gives views, ratings, publish date, titles, and descriptions; cut timings require video analysis.",
            ],
            created_at=utc_now_iso(),
        )
        self.db.save_research_report(report)
        return report
