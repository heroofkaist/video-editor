"""Machine-readable CLI (`symposium`) for operators and external AI agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import uvicorn

from symposium_studio.models.research import ContentCategory
from symposium_studio.models.symposium import ResearchMode
from symposium_studio.studio import SymposiumStudio

cli = typer.Typer(help="Symposium Studio — Open-source AI content production & growth CLI.")


def _emit(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


@cli.command("health")
def cmd_health() -> None:
    """Check studio configuration, animation adapters, and platform connectors."""
    studio = SymposiumStudio()
    _emit(
        {
            "workspaces_root": str(studio.config.workspaces_root),
            "db_path": str(studio.config.db_path),
            "adapters": [c.model_dump(mode="json") for c in studio.animations.list_capabilities()],
            "connectors": studio.research.connector_statuses(),
        }
    )


@cli.command("projects")
def cmd_list_projects() -> None:
    """List all workspace projects."""
    studio = SymposiumStudio()
    _emit([p.model_dump(mode="json") for p in studio.db.list_projects()])


@cli.command("create")
def cmd_create(
    name: str,
    niche: str = typer.Option("science_engineering", help="Creator niche"),
    style: str = typer.Option("educational_explainer", help="Default editing style profile"),
) -> None:
    """Create a structured project workspace."""
    studio = SymposiumStudio()
    manifest = studio.create_project(name=name, creator_niche=niche, editing_style=style)
    _emit(manifest.model_dump(mode="json"))


@cli.command("ingest")
def cmd_ingest(project_id: str, video_path: Path) -> None:
    """Ingest source media into project workspace with SHA256 verification."""
    studio = SymposiumStudio()
    _emit(studio.ingest_video(project_id, video_path))


@cli.command("analyze")
def cmd_analyze(
    project_id: str,
    language: str | None = typer.Option(None, help="Optional ISO language hint (en, ru, kk, ko)"),
    whisper_model: str | None = typer.Option(None, help="Local Whisper model override"),
) -> None:
    """Run local transcription, diarization, PySceneDetect, face tracking, and highlight scoring."""
    studio = SymposiumStudio()
    bundle = studio.analyze_project(project_id, language=language, whisper_model=whisper_model)
    _emit(bundle.model_dump(mode="json"))


@cli.command("edit-transcript")
def cmd_edit_transcript(
    project_id: str,
    corrections_json: Path | None = typer.Option(None, help="JSON file mapping segment_id -> corrected text"),
    segment_id: str | None = typer.Option(None, help="Single segment_id to update"),
    text: str | None = typer.Option(None, help="Replacement text when --segment-id is provided"),
    language: str | None = typer.Option(None, help="Optional language override for --segment-id"),
    confirm: bool = typer.Option(True, help="Mark edited segments as human_confirmed"),
) -> None:
    """Update transcript segment texts cleanly without re-running ASR."""
    corrections: dict[str, str] = {}
    lang_overrides: dict[str, str] = {}
    if corrections_json is not None:
        raw = json.loads(corrections_json.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k, v in raw.items():
                corrections[str(k)] = str(v)
    if segment_id is not None and text is not None:
        corrections[segment_id] = text
        if language:
            lang_overrides[segment_id] = language
    if not corrections and not lang_overrides:
        raise typer.BadParameter("Provide --corrections-json or both --segment-id and --text.")
    studio = SymposiumStudio()
    bundle = studio.update_transcript_segments(
        project_id,
        corrections=corrections,
        language_overrides=lang_overrides or None,
        mark_confirmed=confirm,
    )
    _emit(bundle.model_dump(mode="json"))


@cli.command("research-search")
def cmd_research_search(
    query: str | None = typer.Option(None),
    platform: str | None = typer.Option(None),
    category: str | None = typer.Option(None),
    limit: int = typer.Option(15),
) -> None:
    """Search the cross-niche Viral Intelligence Database."""
    studio = SymposiumStudio()
    items = studio.research.search_patterns(query=query, platform=platform, category=category, limit=limit)
    _emit([i.model_dump(mode="json") for i in items])


@cli.command("research-discover")
def cmd_research_discover(
    target_count: int = typer.Option(100, help="Target number of real public videos to fetch across niches"),
    category: list[str] | None = typer.Option(None, help="Optional category filters"),
    query: str | None = typer.Option(None, help="Optional YouTube Data API query"),
) -> None:
    """Discover real public multi-niche creator videos via official Atom feeds / Data API v3."""
    studio = SymposiumStudio()
    _emit(studio.discover_live_research(target_count=target_count, categories=category or None, query=query))


@cli.command("research-inspect")
def cmd_research_inspect(url: str) -> None:
    """Inspect a public YouTube URL via official oEmbed metadata."""
    studio = SymposiumStudio()
    item = studio.inspect_research_url(url)
    _emit(item.model_dump(mode="json") if item else {"error": "Could not inspect URL"})


@cli.command("style-clone")
def cmd_style_clone(project_id: str, reference_video: Path) -> None:
    """Extract editing pace & color signature from a reference video and apply it to the project."""
    studio = SymposiumStudio()
    _emit(studio.apply_reference_style(project_id, reference_video))


@cli.command("telegram-sync")
def cmd_telegram_sync(project_id: str, chat_id: str) -> None:
    """Fetch official subscriber count via TELEGRAM_BOT_TOKEN (requires bot admin/channel access)."""
    studio = SymposiumStudio()
    _emit(studio.growth.sync_telegram_channel_stats(project_id, chat_id).model_dump(mode="json"))


@cli.command("growth-experiments")
def cmd_growth_experiments(
    q: str | None = typer.Option(None, help="Full-text query across experiments and learned hypotheses"),
    platform: str | None = typer.Option(None, help="Filter by platform (youtube, youtube_shorts, tiktok, instagram_reels, threads)"),
    project_id: str | None = typer.Option(None, help="Filter by project ID"),
) -> None:
    """Search recorded content growth experiments & learned improvement hypotheses."""
    studio = SymposiumStudio()
    _emit(studio.search_experiments(query=q, platform=platform, project_id=project_id))


@cli.command("growth-evaluate")
def cmd_growth_evaluate(project_id: str) -> None:
    """Compare published post metrics against creator same-platform baselines & extract learned hypotheses."""
    studio = SymposiumStudio()
    _emit({"project_id": project_id, "experiments": studio.evaluate_growth_experiments(project_id)})


@cli.command("symposium")
def cmd_symposium(
    project_id: str,
    mode: ResearchMode = typer.Option(ResearchMode.STANDARD),
    style: str | None = typer.Option(None),
) -> None:
    """Run the multi-agent AI Symposium deliberation and generate storyboard."""
    studio = SymposiumStudio()
    session = studio.run_symposium_conference(project_id, mode=mode, style_name=style)
    _emit(session.model_dump(mode="json"))


@cli.command("animate")
def cmd_animate(
    project_id: str,
    template: str = typer.Option("em_induction_faraday"),
    duration: float = typer.Option(6.5),
    adapter: str | None = typer.Option(None),
) -> None:
    """Generate an editable SceneSpec and render an original animation."""
    studio = SymposiumStudio()
    res = studio.generate_animation(
        project_id=project_id,
        template_id=template,
        duration_sec=duration,
        preferred_adapter=adapter,
    )
    _emit(res.model_dump(mode="json"))


@cli.command("render")
def cmd_render(project_id: str, short_variants: int = typer.Option(1)) -> None:
    """Build OTIO timelines, render YouTube 16:9 + Shorts/TikTok/Reels 9:16, thumbnail, and QA reports."""
    studio = SymposiumStudio()
    _emit(studio.build_and_render_deliverables(project_id, short_variants_per_platform=short_variants))


@cli.command("instruct")
def cmd_instruct(project_id: str, instruction: str) -> None:
    """Apply an iterative natural-language editing or campaign instruction."""
    studio = SymposiumStudio()
    _emit(studio.apply_agent_instruction(project_id, instruction))


@cli.command("repurpose")
def cmd_repurpose(
    project_id: str,
    telegram_channel: str = typer.Option("@symposium_lab"),
    threads_max_chars: int = typer.Option(280),
) -> None:
    """Generate YouTube metadata, Threads posts, Shorts/Reels copy, and Telegram funnel campaign."""
    studio = SymposiumStudio()
    bundle = studio.generate_social_and_telegram(
        project_id, telegram_channel=telegram_channel, threads_max_chars=threads_max_chars
    )
    _emit(bundle.model_dump(mode="json"))


@cli.command("campaign")
def cmd_campaign(
    name: str,
    video_path: Path,
    niche: str = typer.Option("science_engineering"),
    style: str = typer.Option("educational_explainer"),
    mode: ResearchMode = typer.Option(ResearchMode.STANDARD),
    telegram_channel: str = typer.Option("@symposium_lab"),
) -> None:
    """Run the full end-to-end production pipeline from raw footage to multi-platform deliverables."""
    studio = SymposiumStudio()
    res = studio.run_campaign_pipeline(
        name=name,
        video_path=video_path,
        creator_niche=niche,
        editing_style=style,
        symposium_mode=mode,
        telegram_channel=telegram_channel,
    )
    _emit(res)


@cli.command("search")
def cmd_search(query: str, project_id: str | None = typer.Option(None)) -> None:
    """Full-text search across projects, transcripts, research reports, viral patterns, and assets."""
    studio = SymposiumStudio()
    _emit(studio.db.search_all(query, project_id=project_id))


@cli.command("serve")
def cmd_serve(host: str = "127.0.0.1", port: int = 8420) -> None:
    """Launch the Symposium Studio FastAPI backend & Web UI server."""
    uvicorn.run("symposium_studio.api.app:app", host=host, port=port, reload=False)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
