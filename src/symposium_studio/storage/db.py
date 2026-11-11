"""SQLite + FTS5 searchable database for projects, transcripts, viral research, and assets.

Designed behind a structured repository API so a PostgreSQL adapter can be
substituted without changing engine logic.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol

from symposium_studio.models.growth import ContentExperimentRecord
from symposium_studio.models.project import AssetRecord, ProjectRecord, RunRecord
from symposium_studio.models.research import ResearchReport, ViralContentItem
from symposium_studio.models.social import FunnelLink, GrowthMetricSnapshot


class StorageBackendProtocol(Protocol):
    """Adapter contract allowing SQLite initially and PostgreSQL in scaled deployments."""

    def upsert_project(self, project: ProjectRecord) -> None: ...
    def get_project(self, project_id: str) -> ProjectRecord | None: ...
    def list_projects(self) -> list[ProjectRecord]: ...
    def search_all(self, query: str, limit: int = 25) -> dict[str, list[dict[str, Any]]]: ...


class StudioDatabase:
    """Thread-safe SQLite + FTS5 persistence engine for Symposium Studio."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    slug TEXT NOT NULL,
                    creator_niche TEXT NOT NULL,
                    editing_style TEXT NOT NULL,
                    workspace_dir TEXT NOT NULL,
                    source_video_path TEXT,
                    source_sha256 TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    command TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    absolute_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS transcript_segments (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    language TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    low_confidence INTEGER NOT NULL,
                    speaker_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    UNIQUE(project_id, segment_id)
                );

                CREATE TABLE IF NOT EXISTS viral_items (
                    item_id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    creator TEXT NOT NULL,
                    original_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    platform_relative_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    provenance_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_reports (
                    report_id TEXT PRIMARY KEY,
                    report_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    target_niche TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS funnel_links (
                    link_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    source_platform TEXT NOT NULL,
                    campaign_slug TEXT NOT NULL,
                    target_telegram_channel TEXT NOT NULL,
                    tagged_url TEXT NOT NULL,
                    clicks INTEGER NOT NULL DEFAULT 0,
                    conversions INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS growth_metrics (
                    snapshot_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    views_or_impressions INTEGER NOT NULL,
                    telegram_clicks INTEGER NOT NULL,
                    telegram_joins INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS growth_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    outcome_verdict TEXT NOT NULL,
                    growth_tier_reached TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
                    entity_type,
                    entity_id,
                    project_id,
                    title,
                    body,
                    tokenize = 'porter unicode61'
                );
                """
            )

    def _upsert_fts(
        self,
        conn: sqlite3.Connection,
        entity_type: str,
        entity_id: str,
        project_id: str,
        title: str,
        body: str,
    ) -> None:
        conn.execute(
            "DELETE FROM fts_index WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        )
        conn.execute(
            "INSERT INTO fts_index (entity_type, entity_id, project_id, title, body) VALUES (?, ?, ?, ?, ?)",
            (entity_type, entity_id, project_id, title, body),
        )

    def upsert_project(self, project: ProjectRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    project_id, name, slug, creator_niche, editing_style,
                    workspace_dir, source_video_path, source_sha256, status,
                    created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name = excluded.name,
                    creator_niche = excluded.creator_niche,
                    editing_style = excluded.editing_style,
                    source_video_path = excluded.source_video_path,
                    source_sha256 = excluded.source_sha256,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    project.project_id,
                    project.name,
                    project.slug,
                    project.creator_niche,
                    project.editing_style,
                    project.workspace_dir,
                    project.source_video_path,
                    project.source_sha256,
                    project.status,
                    project.created_at,
                    project.updated_at,
                    project.model_dump_json(),
                ),
            )
            self._upsert_fts(
                conn,
                "project",
                project.project_id,
                project.project_id,
                project.name,
                f"{project.slug} {project.creator_niche} {project.editing_style} {' '.join(project.tags)}",
            )

    def get_project(self, project_id: str) -> ProjectRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM projects WHERE project_id = ? OR slug = ?",
                (project_id, project_id),
            ).fetchone()
            if not row:
                return None
            return ProjectRecord.model_validate_json(row["payload_json"])

    def list_projects(self) -> list[ProjectRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM projects ORDER BY updated_at DESC"
            ).fetchall()
            return [ProjectRecord.model_validate_json(r["payload_json"]) for r in rows]

    def upsert_run(self, run: RunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, project_id, command, status, started_at, finished_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    finished_at = excluded.finished_at,
                    payload_json = excluded.payload_json
                """,
                (
                    run.run_id,
                    run.project_id,
                    run.command,
                    run.status,
                    run.started_at,
                    run.finished_at,
                    run.model_dump_json(),
                ),
            )

    def upsert_asset(self, asset: AssetRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assets (
                    asset_id, project_id, run_id, role, relative_path,
                    absolute_path, sha256, size_bytes, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    payload_json = excluded.payload_json
                """,
                (
                    asset.asset_id,
                    asset.project_id,
                    asset.run_id,
                    asset.role.value,
                    asset.relative_path,
                    asset.absolute_path,
                    asset.sha256,
                    asset.size_bytes,
                    asset.created_at,
                    asset.model_dump_json(),
                ),
            )
            self._upsert_fts(
                conn,
                "asset",
                asset.asset_id,
                asset.project_id,
                f"{asset.role.value}: {asset.relative_path}",
                f"{asset.source_attribution} {asset.license_info} {json.dumps(asset.metadata)}",
            )

    def list_assets(self, project_id: str) -> list[AssetRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM assets WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
            return [AssetRecord.model_validate_json(r["payload_json"]) for r in rows]

    def index_transcript_segments(self, project_id: str, segments: list[dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM transcript_segments WHERE project_id = ?", (project_id,))
            conn.execute(
                "DELETE FROM fts_index WHERE entity_type = 'transcript' AND project_id = ?",
                (project_id,),
            )
            for seg in segments:
                conn.execute(
                    """
                    INSERT INTO transcript_segments (
                        project_id, segment_id, start_sec, end_sec, language,
                        confidence, low_confidence, speaker_id, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        seg["segment_id"],
                        seg["start_sec"],
                        seg["end_sec"],
                        seg.get("language", "en"),
                        seg.get("confidence", 0.9),
                        1 if seg.get("low_confidence") else 0,
                        seg.get("speaker_id", "speaker_0"),
                        seg["text"],
                    ),
                )
                self._upsert_fts(
                    conn,
                    "transcript",
                    f"{project_id}:{seg['segment_id']}",
                    project_id,
                    f"{seg.get('speaker_id', 'speaker_0')} [{seg['start_sec']:.1f}s-{seg['end_sec']:.1f}s]",
                    seg["text"],
                )

    def upsert_viral_item(self, item: ViralContentItem) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO viral_items (
                    item_id, platform, creator, original_url, title, category,
                    topic, platform_relative_score, confidence, provenance_type, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    platform_relative_score = excluded.platform_relative_score,
                    confidence = excluded.confidence,
                    payload_json = excluded.payload_json
                """,
                (
                    item.item_id,
                    item.platform.value,
                    item.creator,
                    item.original_url,
                    item.title,
                    item.category.value,
                    item.topic,
                    item.platform_relative_score,
                    item.confidence,
                    item.provenance_type,
                    item.model_dump_json(),
                ),
            )
            techniques_txt = " ".join(
                f"{t.name} {t.mechanism} {t.editing_translation}" for t in item.reusable_techniques
            )
            self._upsert_fts(
                conn,
                "viral_item",
                item.item_id,
                "global",
                f"[{item.platform.value}] {item.title}",
                f"{item.creator} {item.category.value} {item.topic} {item.opening_hook_structure} "
                f"{item.editing_style} {item.narrative_structure} {techniques_txt} {item.evidence}",
            )

    def list_viral_items(
        self,
        platform: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[ViralContentItem]:
        clauses: list[str] = []
        params: list[Any] = []
        if platform:
            clauses.append("platform = ?")
            params.append(platform)
        if category:
            clauses.append("category = ?")
            params.append(category)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload_json FROM viral_items {where_sql} ORDER BY platform_relative_score DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [ViralContentItem.model_validate_json(r["payload_json"]) for r in rows]

    def save_research_report(self, report: ResearchReport) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_reports (report_id, report_type, title, target_niche, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (
                    report.report_id,
                    report.report_type,
                    report.title,
                    report.target_niche,
                    report.created_at,
                    report.model_dump_json(),
                ),
            )
            self._upsert_fts(
                conn,
                "research_report",
                report.report_id,
                "global",
                report.title,
                f"{report.report_type} {report.target_niche} {report.summary}",
            )

    def list_research_reports(self, limit: int = 50) -> list[ResearchReport]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM research_reports ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [ResearchReport.model_validate_json(r["payload_json"]) for r in rows]

    def upsert_funnel_link(self, project_id: str, link: FunnelLink) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO funnel_links (
                    link_id, project_id, source_platform, campaign_slug,
                    target_telegram_channel, tagged_url, clicks, conversions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(link_id) DO UPDATE SET
                    clicks = excluded.clicks,
                    conversions = excluded.conversions
                """,
                (
                    link.link_id,
                    project_id,
                    link.source_platform,
                    link.campaign_slug,
                    link.target_telegram_channel,
                    link.tagged_url,
                    link.clicks,
                    link.conversions,
                ),
            )

    def record_funnel_event(self, link_id: str, clicks_delta: int = 0, conversions_delta: int = 0) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE funnel_links
                SET clicks = clicks + ?, conversions = conversions + ?
                WHERE link_id = ?
                """,
                (clicks_delta, conversions_delta, link_id),
            )

    def list_funnel_links(self, project_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if project_id:
                rows = conn.execute(
                    "SELECT * FROM funnel_links WHERE project_id = ?",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM funnel_links").fetchall()
            return [dict(r) for r in rows]

    def save_growth_metric(self, snapshot: GrowthMetricSnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO growth_metrics (
                    snapshot_id, project_id, platform, views_or_impressions,
                    telegram_clicks, telegram_joins, recorded_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.project_id,
                    snapshot.platform,
                    snapshot.views_or_impressions,
                    snapshot.telegram_clicks,
                    snapshot.telegram_joins,
                    snapshot.recorded_at,
                    snapshot.model_dump_json(),
                ),
            )

    def list_growth_metrics(self, project_id: str | None = None) -> list[GrowthMetricSnapshot]:
        with self._connect() as conn:
            if project_id:
                rows = conn.execute(
                    "SELECT payload_json FROM growth_metrics WHERE project_id = ? ORDER BY recorded_at DESC",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload_json FROM growth_metrics ORDER BY recorded_at DESC"
                ).fetchall()
            return [GrowthMetricSnapshot.model_validate_json(r["payload_json"]) for r in rows]

    def upsert_experiment(self, exp: ContentExperimentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO growth_experiments (
                    experiment_id, project_id, platform, strategy_name,
                    outcome_verdict, growth_tier_reached, published_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    strategy_name = excluded.strategy_name,
                    outcome_verdict = excluded.outcome_verdict,
                    growth_tier_reached = excluded.growth_tier_reached,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    exp.experiment_id,
                    exp.project_id,
                    exp.platform,
                    exp.strategy_name,
                    exp.outcome_verdict,
                    exp.growth_tier_reached,
                    exp.published_at,
                    exp.updated_at,
                    exp.model_dump_json(),
                ),
            )
            body = " | ".join(
                [
                    exp.platform,
                    exp.editing_style,
                    exp.strategy_name,
                    exp.hook_strategy,
                    exp.content_hypothesis,
                    exp.outcome_verdict,
                    exp.growth_tier_reached,
                    " ".join(exp.learned_hypotheses),
                ]
            )
            self._upsert_fts(conn, "experiment", exp.experiment_id, exp.project_id, f"[{exp.platform}] {exp.title}", body)

    def list_experiments(
        self,
        project_id: str | None = None,
        platform: str | None = None,
        limit: int = 200,
    ) -> list[ContentExperimentRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if platform:
            clauses.append("platform = ?")
            params.append(platform)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT payload_json FROM growth_experiments {where} ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [ContentExperimentRecord.model_validate_json(r["payload_json"]) for r in rows]

    def search_all(self, query: str, limit: int = 25, project_id: str | None = None) -> list[dict[str, Any]]:
        cleaned = " ".join(tok for tok in query.strip().split() if tok)
        if not cleaned:
            return []
        fts_query = " OR ".join(f'"{tok.replace(chr(34), "")}"*' for tok in cleaned.split())
        with self._connect() as conn:
            if project_id:
                rows = conn.execute(
                    """
                    SELECT entity_type, entity_id, project_id, title,
                           snippet(fts_index, 4, '[', ']', '...', 24) AS snippet,
                           bm25(fts_index) AS rank
                    FROM fts_index
                    WHERE fts_index MATCH ? AND (project_id = ? OR project_id = 'global')
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, project_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT entity_type, entity_id, project_id, title,
                           snippet(fts_index, 4, '[', ']', '...', 24) AS snippet,
                           bm25(fts_index) AS rank
                    FROM fts_index
                    WHERE fts_index MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
            return [dict(r) for r in rows]
