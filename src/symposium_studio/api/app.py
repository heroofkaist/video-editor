"""FastAPI backend exposing all Symposium Studio tools, timelines, research, and UI assets."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from symposium_studio.config import StudioConfig
from symposium_studio.engines.distribution import execute_publish
from symposium_studio.engines.effects import search_effects
from symposium_studio.engines.research import import_explicit_item, infer_platform_from_url
from symposium_studio.engines.symposium import AGENT_ROSTER
from symposium_studio.models.research import ContentCategory, ResearchPlatform
from symposium_studio.models.social import PublishingMode, PublishRequest
from symposium_studio.models.symposium import ResearchMode
from symposium_studio.studio import SymposiumStudio


class CreateProjectRequest(BaseModel):
    name: str
    creator_niche: str = "science_engineering"
    editing_style: str = "educational_explainer"
    tags: list[str] = Field(default_factory=list)


class IngestPathRequest(BaseModel):
    source_path: str
    copy_to_workspace: bool = True


class AnalyzeRequest(BaseModel):
    language: str | None = None
    whisper_model: str | None = None


class SymposiumRequest(BaseModel):
    mode: ResearchMode = ResearchMode.STANDARD
    style_name: str | None = None
    target_min_sec: float = 60.0
    target_max_sec: float = 90.0


class AnimateRequest(BaseModel):
    template_id: str = "em_induction_faraday"
    title: str | None = None
    duration_sec: float = 7.0
    width: int = 1920
    height: int = 1080
    transparent_background: bool = True
    preferred_adapter: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class IterativeInstructionRequest(BaseModel):
    instruction: str


class SocialRequest(BaseModel):
    telegram_channel: str = "@symposium_lab"
    threads_max_chars: int = 280


class ViralImportRequest(BaseModel):
    original_url: str
    creator: str
    title: str
    category: ContentCategory = ContentCategory.EDUCATION
    topic: str
    opening_hook_structure: str
    editing_style: str
    narrative_structure: str
    platform: ResearchPlatform | None = None
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares_or_reposts: int | None = None
    notes: str = ""


class GrowthMetricRequest(BaseModel):
    platform: str
    views_or_impressions: int
    first_7d_views: int | None = None
    likes_or_reactions: int = 0
    comments: int = 0
    shares: int = 0
    new_subscribers_or_followers: int = 0
    avg_view_duration_sec: float | None = None
    watch_time_minutes: float | None = None
    authorized_ctr_pct: float | None = None
    authorized_retention_pct: float | None = None
    telegram_clicks: int = 0
    telegram_joins: int = 0
    measurement_mode: str = "cumulative"
    external_post_id: str | None = None
    subscriber_or_member_count: int | None = None
    notes: str = ""


class DiscoverRequest(BaseModel):
    target_count: int = 100
    categories: list[str] | None = None
    query: str | None = None


class InspectUrlRequest(BaseModel):
    url: str


class StyleCloneRequest(BaseModel):
    reference_video_path: str


class TelegramSyncRequest(BaseModel):
    chat_id: str


class GrowthImportRequest(BaseModel):
    records: list[dict[str, Any]]


def create_app(config: StudioConfig | None = None) -> FastAPI:
    cfg = config or StudioConfig.from_env()
    studio = SymposiumStudio(cfg)

    app = FastAPI(
        title="Symposium Studio API",
        version="0.2.0",
        description="Agent-operable AI Content Production, Creative Symposium, Animation, Multi-Platform Video & Growth Studio.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.studio = studio

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.2.0",
            "workspaces_root": str(cfg.workspaces_root),
            "llm_backend": studio.symposium.llm.detect_backend(),
            "adapters": [c.model_dump(mode="json") for c in studio.animations.list_capabilities()],
            "connectors": studio.research.connector_statuses(),
            "agents_count": len(AGENT_ROSTER),
        }

    # --- Projects & Media ---
    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        return [p.model_dump(mode="json") for p in studio.db.list_projects()]

    @app.post("/api/projects")
    def create_project(req: CreateProjectRequest) -> dict[str, Any]:
        manifest = studio.create_project(req.name, req.creator_niche, req.editing_style, req.tags)
        return manifest.model_dump(mode="json")

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        try:
            return studio.workspaces.get_manifest(project_id).model_dump(mode="json")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/projects/{project_id}/ingest")
    def ingest_from_path(project_id: str, req: IngestPathRequest) -> dict[str, Any]:
        try:
            return studio.ingest_video(project_id, Path(req.source_path), copy_to_workspace=req.copy_to_workspace)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/projects/{project_id}/upload")
    async def upload_video(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / (file.filename or "upload.mp4")
            with dest.open("wb") as fh:
                shutil.copyfileobj(file.file, fh)
            return studio.ingest_video(project_id, dest, copy_to_workspace=True)

    @app.post("/api/projects/{project_id}/analyze")
    def analyze(project_id: str, req: AnalyzeRequest = AnalyzeRequest()) -> dict[str, Any]:
        try:
            return studio.analyze_project(project_id, language=req.language, whisper_model=req.whisper_model).model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/projects/{project_id}/analysis")
    def get_analysis(project_id: str) -> dict[str, Any]:
        try:
            return studio.load_analysis(project_id).model_dump(mode="json")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # --- Research Intelligence ---
    @app.get("/api/research/items")
    def search_viral(
        q: str | None = None,
        platform: str | None = None,
        category: str | None = None,
        limit: int = Query(default=25, le=100),
    ) -> list[dict[str, Any]]:
        return [i.model_dump(mode="json") for i in studio.research.search_patterns(q, platform, category, limit)]

    @app.post("/api/research/items")
    def add_viral_item(req: ViralImportRequest) -> dict[str, Any]:
        plat = req.platform or infer_platform_from_url(req.original_url)
        item = import_explicit_item(
            platform=plat,
            creator=req.creator,
            original_url=req.original_url,
            title=req.title,
            category=req.category,
            topic=req.topic,
            opening_hook_structure=req.opening_hook_structure,
            editing_style=req.editing_style,
            narrative_structure=req.narrative_structure,
            views=req.views,
            likes=req.likes,
            comments=req.comments,
            shares_or_reposts=req.shares_or_reposts,
            notes=req.notes,
        )
        saved = studio.research.add_item(item)
        return saved.model_dump(mode="json")

    @app.post("/api/research/discover")
    def discover_research(req: DiscoverRequest = DiscoverRequest()) -> dict[str, Any]:
        return studio.discover_live_research(
            target_count=req.target_count,
            categories=req.categories,
            query=req.query,
        )

    @app.post("/api/research/inspect-url")
    def inspect_research_url(req: InspectUrlRequest) -> dict[str, Any]:
        item = studio.inspect_research_url(req.url)
        if not item:
            raise HTTPException(status_code=404, detail="Could not retrieve public YouTube oEmbed metadata for URL.")
        return item.model_dump(mode="json")

    @app.post("/api/projects/{project_id}/style-from-reference")
    def clone_style(project_id: str, req: StyleCloneRequest) -> dict[str, Any]:
        ref = Path(req.reference_video_path)
        if not ref.exists():
            raise HTTPException(status_code=404, detail=f"Reference video not found: {ref}")
        try:
            return studio.apply_reference_style(project_id, ref)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/research/reports")
    def list_reports() -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in studio.db.list_research_reports()]

    @app.post("/api/projects/{project_id}/research")
    def create_research_report(project_id: str, report_type: str = "content_proposal") -> dict[str, Any]:
        return studio.run_research_report(project_id, report_type=report_type).model_dump(mode="json")

    # --- AI Symposium ---
    @app.get("/api/symposium/agents")
    def get_agents() -> list[dict[str, Any]]:
        return [{"agent_id": a.agent_id, "title": a.title, "team": a.team.value, "min_mode": a.min_mode.value} for a in AGENT_ROSTER]

    @app.post("/api/projects/{project_id}/symposium")
    def run_symposium(project_id: str, req: SymposiumRequest = SymposiumRequest()) -> dict[str, Any]:
        return studio.run_symposium_conference(
            project_id,
            mode=req.mode,
            style_name=req.style_name,
            target_min_sec=req.target_min_sec,
            target_max_sec=req.target_max_sec,
        ).model_dump(mode="json")

    @app.get("/api/projects/{project_id}/symposium")
    def get_latest_symposium(project_id: str) -> dict[str, Any]:
        manifest = studio.workspaces.get_manifest(project_id)
        sym_files = sorted((Path(manifest.project.workspace_dir) / "symposium").glob("sym_*.json"))
        if not sym_files:
            raise HTTPException(status_code=404, detail="No symposium session yet.")
        return studio.workspaces.get_manifest(project_id).model_dump(mode="json") | {
            "latest_session": __import__("json").loads(sym_files[-1].read_text(encoding="utf-8"))
        }

    # --- Animation & Effects ---
    @app.get("/api/animations/adapters")
    def list_adapters() -> list[dict[str, Any]]:
        return [c.model_dump(mode="json") for c in studio.animations.list_capabilities()]

    @app.post("/api/projects/{project_id}/animate")
    def animate(project_id: str, req: AnimateRequest) -> dict[str, Any]:
        return studio.generate_animation(
            project_id=project_id,
            template_id=req.template_id,
            title=req.title,
            duration_sec=req.duration_sec,
            width=req.width,
            height=req.height,
            transparent_background=req.transparent_background,
            preferred_adapter=req.preferred_adapter,
            parameters=req.parameters,
        ).model_dump(mode="json")

    @app.get("/api/effects")
    def get_effects(q: str | None = None, category: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "preset_id": e.preset_id,
                "name": e.name,
                "category": e.category.value,
                "description": e.description,
                "editorial_purpose": e.editorial_purpose,
                "ffmpeg_filter_hint": e.ffmpeg_filter_hint,
                "parameters": e.parameters,
            }
            for e in search_effects(q, category)
        ]

    # --- Render, Iterative Edit, Social, Telegram, Growth, Search ---
    @app.post("/api/projects/{project_id}/render")
    def render_all(project_id: str, short_variants: int = 1) -> dict[str, Any]:
        return studio.build_and_render_deliverables(project_id, short_variants_per_platform=short_variants)

    @app.post("/api/projects/{project_id}/instruct")
    def instruct(project_id: str, req: IterativeInstructionRequest) -> dict[str, Any]:
        return studio.apply_agent_instruction(project_id, req.instruction)

    @app.post("/api/projects/{project_id}/social")
    def social(project_id: str, req: SocialRequest = SocialRequest()) -> dict[str, Any]:
        return studio.generate_social_and_telegram(
            project_id, telegram_channel=req.telegram_channel, threads_max_chars=req.threads_max_chars
        ).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/publish")
    def publish(project_id: str, req: PublishRequest) -> dict[str, Any]:
        manifest = studio.workspaces.get_manifest(project_id)
        try:
            return execute_publish(req, Path(manifest.project.workspace_dir))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/projects/{project_id}/report")
    def production_report(project_id: str) -> dict[str, Any]:
        return studio.write_production_report(project_id)

    @app.post("/api/projects/{project_id}/growth")
    def add_growth_metric(project_id: str, req: GrowthMetricRequest) -> dict[str, Any]:
        return studio.growth.record_snapshot(
            project_id=project_id,
            platform=req.platform,
            views_or_impressions=req.views_or_impressions,
            first_7d_views=req.first_7d_views,
            likes_or_reactions=req.likes_or_reactions,
            comments=req.comments,
            shares=req.shares,
            new_subscribers_or_followers=req.new_subscribers_or_followers,
            avg_view_duration_sec=req.avg_view_duration_sec,
            watch_time_minutes=req.watch_time_minutes,
            authorized_ctr_pct=req.authorized_ctr_pct,
            authorized_retention_pct=req.authorized_retention_pct,
            telegram_clicks=req.telegram_clicks,
            telegram_joins=req.telegram_joins,
            measurement_mode=req.measurement_mode,
            external_post_id=req.external_post_id,
            subscriber_or_member_count=req.subscriber_or_member_count,
            notes=req.notes,
        ).model_dump(mode="json")

    @app.post("/api/growth/{project_id}/evaluate")
    def evaluate_experiments(project_id: str) -> dict[str, Any]:
        evaluated = studio.evaluate_growth_experiments(project_id)
        return {"project_id": project_id, "evaluated_count": len(evaluated), "experiments": evaluated}

    @app.get("/api/growth/experiments")
    def list_growth_experiments(
        q: str | None = None,
        platform: str | None = None,
        project_id: str | None = None,
        limit: int = Query(default=50, le=200),
    ) -> list[dict[str, Any]]:
        return studio.search_experiments(query=q, platform=platform, project_id=project_id, limit=limit)

    @app.post("/api/growth/{project_id}/telegram-sync")
    def sync_telegram(project_id: str, req: TelegramSyncRequest) -> dict[str, Any]:
        try:
            return studio.growth.sync_telegram_channel_stats(project_id, req.chat_id).model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/growth/{project_id}/import")
    def import_growth(project_id: str, req: GrowthImportRequest) -> dict[str, Any]:
        try:
            snaps = studio.growth.import_analytics_records(project_id, req.records)
            return {"imported_count": len(snaps), "snapshots": [s.model_dump(mode="json") for s in snaps]}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/funnel/{link_id}/event")
    def funnel_event(link_id: str, clicks_delta: int = 1, conversions_delta: int = 0) -> dict[str, Any]:
        studio.db.record_funnel_event(link_id, clicks_delta=clicks_delta, conversions_delta=conversions_delta)
        links = [l for l in studio.db.list_funnel_links() if l["link_id"] == link_id]
        if not links:
            raise HTTPException(status_code=404, detail="Funnel link not found.")
        return links[0]

    @app.get("/r/{link_id}")
    def redirect_funnel_link(link_id: str) -> RedirectResponse:
        links = [l for l in studio.db.list_funnel_links() if l["link_id"] == link_id]
        if not links:
            raise HTTPException(status_code=404, detail="Funnel link not found.")
        target = str(links[0].get("tagged_url", ""))
        if not target.startswith("https://t.me/"):
            raise HTTPException(status_code=400, detail="Unsafe redirect target outside https://t.me/ rejected.")
        studio.db.record_funnel_event(link_id, clicks_delta=1)
        return RedirectResponse(url=target, status_code=302)

    @app.get("/api/growth")
    def get_growth(project_id: str | None = None) -> dict[str, Any]:
        return studio.growth.summarize(project_id=project_id)

    @app.get("/api/search")
    def search(q: str, project_id: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        return studio.db.search_all(q, limit=limit, project_id=project_id)

    @app.get("/api/projects/{project_id}/files/{rel_path:path}")
    def serve_workspace_file(project_id: str, rel_path: str) -> FileResponse:
        manifest = studio.workspaces.get_manifest(project_id)
        ws_root = Path(manifest.project.workspace_dir).resolve()
        target = (ws_root / rel_path).resolve()
        if not str(target).startswith(str(ws_root)) or not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found inside project workspace.")
        return FileResponse(path=target)

    # Serve built frontend dist if present
    web_dist = cfg.repo_root / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web_ui")

    return app


app = create_app()
