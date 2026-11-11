"""Growth Intelligence engine: evaluates authorized metrics and funnel conversions."""

from __future__ import annotations

import uuid
from typing import Any

from symposium_studio.models.project import utc_now_iso
from symposium_studio.models.social import GrowthMetricSnapshot
from symposium_studio.storage.db import StudioDatabase


class GrowthIntelligenceEngine:
    """Computes platform-scoped conversion performance and strategic feedback."""

    def __init__(self, db: StudioDatabase) -> None:
        self.db = db

    def record_snapshot(
        self,
        project_id: str,
        platform: str,
        views_or_impressions: int,
        likes_or_reactions: int = 0,
        comments: int = 0,
        shares: int = 0,
        telegram_clicks: int = 0,
        telegram_joins: int = 0,
        data_provenance: str = "authorized_analytics_import",
        notes: str = "",
    ) -> GrowthMetricSnapshot:
        snap = GrowthMetricSnapshot(
            snapshot_id=f"gm_{uuid.uuid4().hex[:8]}",
            project_id=project_id,
            platform=platform,
            views_or_impressions=views_or_impressions,
            likes_or_reactions=likes_or_reactions,
            comments=comments,
            shares=shares,
            telegram_clicks=telegram_clicks,
            telegram_joins=telegram_joins,
            data_provenance=data_provenance,
            recorded_at=utc_now_iso(),
            notes=notes,
        )
        self.db.save_growth_metric(snap)
        return snap

    def summarize(self, project_id: str | None = None) -> dict[str, Any]:
        metrics = self.db.list_growth_metrics(project_id=project_id)
        links = self.db.list_funnel_links(project_id=project_id)

        by_platform: dict[str, dict[str, Any]] = {}
        for m in metrics:
            bucket = by_platform.setdefault(
                m.platform,
                {
                    "platform": m.platform,
                    "snapshots": 0,
                    "total_views_or_impressions": 0,
                    "total_engagements": 0,
                    "telegram_clicks": 0,
                    "telegram_joins": 0,
                },
            )
            bucket["snapshots"] += 1
            bucket["total_views_or_impressions"] += m.views_or_impressions
            bucket["total_engagements"] += m.likes_or_reactions + m.comments + m.shares
            bucket["telegram_clicks"] += m.telegram_clicks
            bucket["telegram_joins"] += m.telegram_joins

        for b in by_platform.values():
            views = max(1, b["total_views_or_impressions"])
            clicks = b["telegram_clicks"]
            b["engagement_rate_pct"] = round(b["total_engagements"] / views * 100.0, 2)
            b["click_to_join_pct"] = round(b["telegram_joins"] / max(1, clicks) * 100.0, 2) if clicks else None

        total_clicks = sum(int(l.get("clicks", 0)) for l in links)
        total_conv = sum(int(l.get("conversions", 0)) for l in links)
        recommendations: list[str] = []
        if not metrics:
            recommendations.append(
                "No authorized performance snapshots imported yet. Export official YouTube/Telegram/Threads analytics after publishing to feed Growth Intelligence."
            )
        else:
            best = max(by_platform.values(), key=lambda x: x["engagement_rate_pct"])
            recommendations.append(
                f"Highest within-platform engagement observed on {best['platform']} ({best['engagement_rate_pct']}%). Prioritize repeating its hook & caption cadence."
            )
        return {
            "project_id": project_id,
            "platform_summaries": list(by_platform.values()),
            "funnel_links": links,
            "funnel_totals": {"clicks": total_clicks, "conversions": total_conv},
            "recommendations": recommendations,
            "note": "Metrics are never compared raw across platforms or invented when absent.",
        }
