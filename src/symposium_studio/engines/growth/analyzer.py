"""Growth Intelligence engine: evaluates authorized metrics, prevents cumulative double-counting, tracks 1x/1.5x/2x targets, and closes the experiment learning loop."""

from __future__ import annotations

import os
import statistics
import uuid
from collections import defaultdict
from typing import Any

import httpx

from symposium_studio.models.growth import ContentExperimentRecord
from symposium_studio.models.project import utc_now_iso
from symposium_studio.models.research import ResearchReport
from symposium_studio.models.social import GrowthMetricSnapshot
from symposium_studio.models.symposium import SymposiumSession
from symposium_studio.storage.db import StudioDatabase


DEFAULT_EXPERIMENT_PLATFORMS: tuple[str, ...] = (
    "youtube",
    "youtube_shorts",
    "tiktok",
    "instagram_reels",
    "threads",
)


def _classify_growth_tier(ratio: float | None) -> tuple[str, str]:
    if ratio is None:
        return "awaiting_post_publication_data", "pending"
    if ratio >= 2.0:
        return "stretch_target_2.0x", "outperformed"
    if ratio >= 1.5:
        return "intermediate_target_1.5x", "outperformed"
    if ratio >= 1.0:
        return "baseline_met_1.0x", "matched_baseline"
    return "below_baseline", "underperformed"


class GrowthIntelligenceEngine:
    """Computes platform-scoped conversion performance & closed-loop experiment learning."""

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
        measurement_mode: str = "cumulative",
        external_post_id: str | None = None,
        subscriber_or_member_count: int | None = None,
        first_7d_views: int | None = None,
        new_subscribers_or_followers: int = 0,
        avg_view_duration_sec: float | None = None,
        watch_time_minutes: float | None = None,
        authorized_ctr_pct: float | None = None,
        authorized_retention_pct: float | None = None,
        data_provenance: str = "authorized_analytics_import",
        notes: str = "",
        recorded_at: str | None = None,
        auto_evaluate_experiments: bool = True,
    ) -> GrowthMetricSnapshot:
        mode = measurement_mode if measurement_mode in {"cumulative", "delta"} else "cumulative"
        snap = GrowthMetricSnapshot(
            snapshot_id=f"gm_{uuid.uuid4().hex[:8]}",
            project_id=project_id,
            platform=platform,
            measurement_mode=mode,
            external_post_id=external_post_id,
            views_or_impressions=views_or_impressions,
            first_7d_views=first_7d_views,
            likes_or_reactions=likes_or_reactions,
            comments=comments,
            shares=shares,
            new_subscribers_or_followers=new_subscribers_or_followers,
            avg_view_duration_sec=avg_view_duration_sec,
            watch_time_minutes=watch_time_minutes,
            authorized_ctr_pct=authorized_ctr_pct,
            authorized_retention_pct=authorized_retention_pct,
            telegram_clicks=telegram_clicks,
            telegram_joins=telegram_joins,
            subscriber_or_member_count=subscriber_or_member_count,
            data_provenance=data_provenance,
            recorded_at=recorded_at or utc_now_iso(),
            notes=notes,
        )
        self.db.save_growth_metric(snap)
        if auto_evaluate_experiments and self.db.list_experiments(project_id=project_id, limit=5):
            self.evaluate_project_experiments(project_id)
        return snap

    def sync_telegram_channel_stats(self, project_id: str, chat_id: str) -> GrowthMetricSnapshot:
        """Pull live authorized subscriber/member count from Telegram Bot API when TELEGRAM_BOT_TOKEN is set."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN not configured. Supply token for authorized Bot API stats or import Telegram CSV/JSON exports."
            )
        target = chat_id if chat_id.startswith("@") or chat_id.lstrip("-").isdigit() else f"@{chat_id}"
        resp = httpx.get(
            f"https://api.telegram.org/bot{token}/getChatMemberCount",
            params={"chat_id": target},
            timeout=10.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API error: {payload}")
        members = int(payload["result"])
        return self.record_snapshot(
            project_id=project_id,
            platform="telegram",
            views_or_impressions=0,
            subscriber_or_member_count=members,
            measurement_mode="cumulative",
            external_post_id=target,
            data_provenance="official_telegram_bot_api",
            notes=f"Official Telegram Bot API getChatMemberCount for {target}: {members} members.",
        )

    def import_analytics_records(self, project_id: str, records: list[dict[str, Any]]) -> list[GrowthMetricSnapshot]:
        """Bulk-import authorized platform analytics rows cleanly."""
        saved: list[GrowthMetricSnapshot] = []
        for r in records:
            saved.append(
                self.record_snapshot(
                    project_id=project_id,
                    platform=str(r["platform"]),
                    views_or_impressions=int(r.get("views_or_impressions", r.get("views", 0))),
                    first_7d_views=r.get("first_7d_views"),
                    likes_or_reactions=int(r.get("likes_or_reactions", r.get("likes", 0))),
                    comments=int(r.get("comments", 0)),
                    shares=int(r.get("shares", 0)),
                    new_subscribers_or_followers=int(r.get("new_subscribers_or_followers", r.get("subscribers_gained", 0))),
                    avg_view_duration_sec=r.get("avg_view_duration_sec"),
                    watch_time_minutes=r.get("watch_time_minutes"),
                    authorized_ctr_pct=r.get("authorized_ctr_pct"),
                    authorized_retention_pct=r.get("authorized_retention_pct"),
                    telegram_clicks=int(r.get("telegram_clicks", 0)),
                    telegram_joins=int(r.get("telegram_joins", 0)),
                    measurement_mode=str(r.get("measurement_mode", "cumulative")),
                    external_post_id=r.get("external_post_id"),
                    subscriber_or_member_count=r.get("subscriber_or_member_count"),
                    data_provenance=str(r.get("data_provenance", "authorized_analytics_import")),
                    recorded_at=r.get("recorded_at"),
                    notes=str(r.get("notes", "")),
                    auto_evaluate_experiments=False,
                )
            )
        if self.db.list_experiments(project_id=project_id, limit=5):
            self.evaluate_project_experiments(project_id)
        return saved

    def record_experiment_from_production(
        self,
        project_id: str,
        title: str,
        creator_niche: str,
        session: SymposiumSession,
        research_report: ResearchReport | None = None,
        platforms: tuple[str, ...] = DEFAULT_EXPERIMENT_PLATFORMS,
    ) -> list[ContentExperimentRecord]:
        """Register traceable pre-publication experiments per platform from Symposium predictions & strategy."""
        selected = next((c for c in session.candidate_evaluations if c.selected), None) or (
            session.candidate_evaluations[0] if session.candidate_evaluations else None
        )
        strategy_name = selected.strategy_name if selected else session.storyboard.selected_strategy
        editing_style = selected.editing_style if selected else "educational_explainer"
        pred_score = selected.composite_score if selected else 0.72
        pred_unc = float(selected.confidence_interval.get("uncertainty", 0.22)) if selected else 0.22
        pred_dims = selected.scorecard.model_dump() if selected else {}

        obj = session.growth_objective
        hook_strategy = obj.selected_hook_strategy if obj else session.storyboard.selected_strategy
        content_hyp = obj.content_hypothesis if obj else "Deliver hook-first mechanism proof with cross-platform funnel."
        audiences = list(obj.target_audience) if obj else ["physics_enthusiasts", "electrical_engineers", "students"]
        res_ids = list(research_report.source_item_ids[:6]) if research_report else []
        dec_summary = "; ".join(f"{d.topic}: {d.chosen_option}" for d in session.decisions[:3])
        now = utc_now_iso()

        existing = {e.platform: e for e in self.db.list_experiments(project_id=project_id, limit=50)}
        saved: list[ContentExperimentRecord] = []
        for plat in platforms:
            exp_id = existing[plat].experiment_id if plat in existing else f"exp_{project_id}_{plat}"
            exp = ContentExperimentRecord(
                experiment_id=exp_id,
                project_id=project_id,
                title=title,
                creator_niche=creator_niche,
                platform=plat,
                editing_style=editing_style,
                strategy_name=strategy_name,
                hook_strategy=hook_strategy,
                target_audience=audiences,
                content_hypothesis=content_hyp,
                predicted_composite_score=pred_score,
                predicted_uncertainty=pred_unc,
                predicted_dimensions=pred_dims,
                research_item_ids=res_ids,
                decision_summary=dec_summary,
                published_at=existing[plat].published_at if plat in existing else now,
                updated_at=now,
            )
            self.db.upsert_experiment(exp)
            saved.append(exp)
        return saved

    def _aggregate_project_metrics_by_platform(self, project_id: str) -> dict[str, dict[str, Any]]:
        """Deduplicate cumulative snapshots per external_post_id & extract observed analytics per platform."""
        metrics = self.db.list_growth_metrics(project_id=project_id)
        cum_streams: dict[tuple[str, str], list[GrowthMetricSnapshot]] = defaultdict(list)
        delta_rows: list[GrowthMetricSnapshot] = []
        for m in metrics:
            if m.measurement_mode == "delta":
                delta_rows.append(m)
            else:
                cum_streams[(m.platform, m.external_post_id or "__channel_default__")].append(m)

        agg: dict[str, dict[str, Any]] = {}

        def _b(plat: str) -> dict[str, Any]:
            return agg.setdefault(
                plat,
                {
                    "views": 0,
                    "first_7d_views": None,
                    "engagements": 0,
                    "shares": 0,
                    "new_subs": 0,
                    "tg_clicks": 0,
                    "tg_joins": 0,
                    "avd_sec": None,
                    "watch_min": None,
                    "ctr_pct": None,
                    "ret_pct": None,
                    "has_data": False,
                },
            )

        for (plat, _ext), series in cum_streams.items():
            latest = sorted(series, key=lambda x: x.recorded_at)[-1]
            d = _b(plat)
            d["has_data"] = True
            d["views"] += latest.views_or_impressions
            f7 = latest.first_7d_views if latest.first_7d_views is not None else (latest.views_or_impressions if latest.views_or_impressions > 0 else None)
            if f7 is not None:
                d["first_7d_views"] = (d["first_7d_views"] or 0) + f7
            d["engagements"] += latest.likes_or_reactions + latest.comments + latest.shares
            d["shares"] += latest.shares
            d["new_subs"] += latest.new_subscribers_or_followers
            d["tg_clicks"] += latest.telegram_clicks
            d["tg_joins"] += latest.telegram_joins
            if latest.avg_view_duration_sec is not None:
                d["avd_sec"] = latest.avg_view_duration_sec
            if latest.watch_time_minutes is not None:
                d["watch_min"] = (d["watch_min"] or 0.0) + latest.watch_time_minutes
            if latest.authorized_ctr_pct is not None:
                d["ctr_pct"] = latest.authorized_ctr_pct
            if latest.authorized_retention_pct is not None:
                d["ret_pct"] = latest.authorized_retention_pct

        for m in delta_rows:
            d = _b(m.platform)
            d["has_data"] = True
            d["views"] += m.views_or_impressions
            f7 = m.first_7d_views if m.first_7d_views is not None else (m.views_or_impressions if m.views_or_impressions > 0 else None)
            if f7 is not None:
                d["first_7d_views"] = (d["first_7d_views"] or 0) + f7
            d["engagements"] += m.likes_or_reactions + m.comments + m.shares
            d["shares"] += m.shares
            d["new_subs"] += m.new_subscribers_or_followers
            d["tg_clicks"] += m.telegram_clicks
            d["tg_joins"] += m.telegram_joins
            if m.avg_view_duration_sec is not None:
                d["avd_sec"] = m.avg_view_duration_sec
            if m.watch_time_minutes is not None:
                d["watch_min"] = (d["watch_min"] or 0.0) + m.watch_time_minutes
            if m.authorized_ctr_pct is not None:
                d["ctr_pct"] = m.authorized_ctr_pct
            if m.authorized_retention_pct is not None:
                d["ret_pct"] = m.authorized_retention_pct
        return agg

    def evaluate_project_experiments(self, project_id: str) -> list[ContentExperimentRecord]:
        """Compare observed post-publication metrics vs creator's same-platform historical median & generate hypotheses."""
        proj_exps = self.db.list_experiments(project_id=project_id, limit=50)
        if not proj_exps:
            return []

        all_exps = self.db.list_experiments(limit=500)
        by_plat_data = self._aggregate_project_metrics_by_platform(project_id)
        now = utc_now_iso()
        updated: list[ContentExperimentRecord] = []

        for exp in proj_exps:
            obs = by_plat_data.get(exp.platform)
            if not obs or not obs["has_data"]:
                updated.append(exp)
                continue

            # Historical median STRICTLY on the same platform from OTHER projects with observed views
            comparable_views = [
                float(other.first_7d_views)
                for other in all_exps
                if other.platform == exp.platform
                and other.project_id != project_id
                and other.first_7d_views is not None
                and other.first_7d_views > 0
            ]
            baseline_med = round(statistics.median(comparable_views), 1) if comparable_views else None
            views_7d = obs["first_7d_views"] if obs["first_7d_views"] is not None else obs["views"]
            eng_rate = round(obs["engagements"] / views_7d * 100.0, 2) if views_7d > 0 else 0.0
            ratio = round(views_7d / baseline_med, 2) if (baseline_med and baseline_med > 0 and views_7d > 0) else None
            tier, verdict = _classify_growth_tier(ratio)
            if ratio is None and views_7d > 0:
                tier = "initial_baseline_recorded"
                verdict = "matched_baseline"

            hyps: list[str] = []
            if ratio is not None:
                if ratio >= 1.5:
                    hyps.append(
                        f"WIN ({ratio}x {exp.platform} median): '{exp.strategy_name}' + '{exp.editing_style}' beat prior baseline ({views_7d:,} vs {int(baseline_med):,}); elevate priority in next Symposium."
                    )
                elif ratio >= 1.0:
                    hyps.append(
                        f"BASELINE+ ({ratio}x {exp.platform} median): Strategy matched/exceeded recent baseline; sharpen hook tension or packaging to push toward 1.5x intermediate target."
                    )
                else:
                    hyps.append(
                        f"UNDERPERFORMED ({ratio}x {exp.platform} median, {views_7d:,} vs {int(baseline_med):,}): Inspect opening 3s drop-off and packaging gap vs winning research outliers."
                    )
            else:
                hyps.append(
                    f"Initial {exp.platform} observation logged ({views_7d:,} views, {eng_rate}% engagement). Needs prior comparable {exp.platform} videos before claiming 1.5x/2.0x lift."
                )

            if ratio is not None:
                if verdict == "outperformed":
                    hyps.append(
                        f"Prediction confirmed: pre-render composite {exp.predicted_composite_score:.2f} (±{exp.predicted_uncertainty:.2f}) aligned with {ratio}x observed performance; reinforce similar hook & pacing priors."
                    )
                elif verdict == "underperformed" and exp.predicted_composite_score >= 0.70:
                    hyps.append(
                        f"Prediction calibration gap: high pre-render composite ({exp.predicted_composite_score:.2f} ± {exp.predicted_uncertainty:.2f}) fell short of baseline ({ratio}x); audit thumbnail/title packaging vs actual retention."
                    )

            if len(comparable_views) < 3 and comparable_views:
                hyps.append(
                    f"Sample-size caution: only {len(comparable_views)} comparable prior {exp.platform} experiment(s); avoid treating a single outlier as proof."
                )

            if obs["ctr_pct"] is not None and obs["ctr_pct"] < 4.5 and (obs["ret_pct"] or 50.0) >= 48.0:
                hyps.append(
                    f"Diagnosis: retention healthy ({obs['ret_pct']}%) but CTR modest ({obs['ctr_pct']}%) — iterate thumbnail contrast & title curiosity gap before changing edit pacing."
                )
            if obs["tg_clicks"] > 0 and obs["tg_joins"] == 0:
                hyps.append(
                    f"Funnel friction: {obs['tg_clicks']} Telegram clicks yielded 0 recorded joins — strengthen the concrete resource payoff advertised in the pin."
                )
            elif obs["tg_joins"] > 0:
                hyps.append(
                    f"Funnel conversion active: {obs['tg_joins']} Telegram join(s) recorded from {exp.platform}."
                )

            exp.first_7d_views = views_7d
            exp.engagement_rate_pct = eng_rate
            exp.avg_view_duration_sec = obs["avd_sec"]
            exp.watch_time_minutes = obs["watch_min"]
            exp.authorized_ctr_pct = obs["ctr_pct"]
            exp.authorized_retention_pct = obs["ret_pct"]
            exp.shares = obs["shares"]
            exp.new_subscribers_or_followers = obs["new_subs"]
            exp.telegram_clicks = obs["tg_clicks"]
            exp.telegram_joins = obs["tg_joins"]
            exp.creator_baseline_median_views = baseline_med
            exp.ratio_vs_creator_median = ratio
            exp.growth_tier_reached = tier
            exp.outcome_verdict = verdict
            exp.learned_hypotheses = hyps
            exp.updated_at = now
            self.db.upsert_experiment(exp)
            updated.append(exp)
        return updated

    def summarize(self, project_id: str | None = None) -> dict[str, Any]:
        metrics = self.db.list_growth_metrics(project_id=project_id)
        links = self.db.list_funnel_links(project_id=project_id)

        cum_streams: dict[tuple[str, str, str], list[GrowthMetricSnapshot]] = defaultdict(list)
        delta_rows: list[GrowthMetricSnapshot] = []
        for m in metrics:
            if m.measurement_mode == "delta":
                delta_rows.append(m)
            else:
                key = (m.platform, m.project_id, m.external_post_id or "__channel_default__")
                cum_streams[key].append(m)

        by_platform: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []

        def _ensure_bucket(plat: str) -> dict[str, Any]:
            return by_platform.setdefault(
                plat,
                {
                    "platform": plat,
                    "snapshots": 0,
                    "cumulative_series_count": 0,
                    "delta_snapshots_count": 0,
                    "total_views_or_impressions": 0,
                    "total_engagements": 0,
                    "telegram_clicks": 0,
                    "telegram_joins": 0,
                    "latest_views_delta_since_prev": 0,
                    "latest_member_count": None,
                    "new_subscribers_or_followers": 0,
                },
            )

        for (plat, _pid, _ext), series in cum_streams.items():
            series_sorted = sorted(series, key=lambda x: x.recorded_at)
            latest = series_sorted[-1]
            prev = series_sorted[-2] if len(series_sorted) >= 2 else None
            b = _ensure_bucket(plat)
            b["snapshots"] += len(series_sorted)
            b["cumulative_series_count"] += 1
            b["total_views_or_impressions"] += latest.views_or_impressions
            b["total_engagements"] += latest.likes_or_reactions + latest.comments + latest.shares
            b["telegram_clicks"] += latest.telegram_clicks
            b["telegram_joins"] += latest.telegram_joins
            b["new_subscribers_or_followers"] += latest.new_subscribers_or_followers
            if prev is not None:
                delta_v = latest.views_or_impressions - prev.views_or_impressions
                b["latest_views_delta_since_prev"] += delta_v
                if delta_v < 0:
                    warnings.append(
                        f"Cumulative counter regression on {plat} ({_ext}): dropped from {prev.views_or_impressions} to {latest.views_or_impressions}."
                    )
            if latest.subscriber_or_member_count is not None:
                b["latest_member_count"] = latest.subscriber_or_member_count

        for m in delta_rows:
            b = _ensure_bucket(m.platform)
            b["snapshots"] += 1
            b["delta_snapshots_count"] += 1
            b["total_views_or_impressions"] += m.views_or_impressions
            b["total_engagements"] += m.likes_or_reactions + m.comments + m.shares
            b["telegram_clicks"] += m.telegram_clicks
            b["telegram_joins"] += m.telegram_joins
            b["new_subscribers_or_followers"] += m.new_subscribers_or_followers
            if m.subscriber_or_member_count is not None:
                b["latest_member_count"] = m.subscriber_or_member_count

        for b in by_platform.values():
            views = b["total_views_or_impressions"]
            clicks = b["telegram_clicks"]
            b["engagement_rate_pct"] = round(b["total_engagements"] / views * 100.0, 2) if views > 0 else 0.0
            b["click_to_join_pct"] = round(b["telegram_joins"] / max(1, clicks) * 100.0, 2) if clicks else None

        # Separate platform evaluation for 1.0x -> 1.5x -> 2.0x measurable growth targets
        all_exps = self.db.list_experiments(project_id=project_id, limit=200)
        global_exps = self.db.list_experiments(limit=400)
        exps_by_plat: dict[str, list[ContentExperimentRecord]] = defaultdict(list)
        for ex in global_exps:
            if ex.first_7d_views is not None and ex.first_7d_views > 0:
                exps_by_plat[ex.platform].append(ex)

        target_progress: list[dict[str, Any]] = []
        for plat, ex_list in sorted(exps_by_plat.items()):
            ratios = [e.ratio_vs_creator_median for e in ex_list if e.ratio_vs_creator_median is not None]
            views_list = [e.first_7d_views for e in ex_list if e.first_7d_views is not None]
            med_views = round(statistics.median(views_list), 1) if views_list else None
            med_ratio = round(statistics.median(ratios), 2) if ratios else None
            target_progress.append(
                {
                    "platform": plat,
                    "published_samples": len(ex_list),
                    "median_first_7d_views": med_views,
                    "median_ratio_vs_baseline": med_ratio,
                    "reached_1_5x_count": sum(1 for r in ratios if r >= 1.5),
                    "reached_2_0x_count": sum(1 for r in ratios if r >= 2.0),
                    "sample_confidence": "meaningful_multi_video_sample" if len(ex_list) >= 3 else "early_sample_avoid_outlier_bias",
                }
            )

        prediction_vs_actual = [
            {
                "experiment_id": e.experiment_id,
                "platform": e.platform,
                "strategy_name": e.strategy_name,
                "predicted_score": e.predicted_composite_score,
                "predicted_uncertainty": e.predicted_uncertainty,
                "first_7d_views": e.first_7d_views,
                "ratio_vs_creator_median": e.ratio_vs_creator_median,
                "growth_tier_reached": e.growth_tier_reached,
                "outcome_verdict": e.outcome_verdict,
                "learned_hypotheses": e.learned_hypotheses,
            }
            for e in all_exps[:20]
        ]

        total_clicks = sum(int(l.get("clicks", 0)) for l in links)
        total_conv = sum(int(l.get("conversions", 0)) for l in links)
        recommendations: list[str] = []
        if not metrics:
            recommendations.append(
                "No authorized performance snapshots imported yet. Export official YouTube/Telegram/Threads analytics after publishing or sync Telegram Bot API to close the learning loop."
            )
        else:
            with_views = [v for v in by_platform.values() if v["total_views_or_impressions"] > 0]
            if with_views:
                best = max(with_views, key=lambda x: x["engagement_rate_pct"])
                recommendations.append(
                    f"Highest within-platform engagement observed on {best['platform']} ({best['engagement_rate_pct']}%). Prioritize repeating its hook & caption cadence."
                )
            else:
                recommendations.append(
                    "Snapshots recorded (member/click counts); import view/impression metrics for engagement-rate comparison."
                )
        for ex in all_exps:
            if ex.learned_hypotheses:
                recommendations.append(ex.learned_hypotheses[0])
                break

        return {
            "project_id": project_id,
            "platform_summaries": list(by_platform.values()),
            "target_progress": target_progress,
            "experiments": [e.model_dump(mode="json") for e in all_exps[:30]],
            "prediction_vs_actual": prediction_vs_actual,
            "funnel_links": links,
            "funnel_totals": {"clicks": total_clicks, "conversions": total_conv},
            "warnings": warnings,
            "recommendations": recommendations,
            "note": (
                "Cumulative snapshots take only the latest state per (platform, project, external_post_id) so repeated captures never double-count; "
                "long-form, Shorts, TikTok, Instagram, and Threads are evaluated strictly within their own cohorts against creator medians (1.0x / 1.5x / 2.0x targets)."
            ),
        }
