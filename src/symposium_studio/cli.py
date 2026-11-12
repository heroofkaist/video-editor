"""Machine-readable CLI (`symposium`) for operators and external AI agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import uvicorn

from symposium_studio.models.reference import SelectionMode
from symposium_studio.models.research import ContentCategory
from symposium_studio.models.symposium import ResearchMode
from symposium_studio.studio import SymposiumStudio

cli = typer.Typer(help="Symposium Studio — Open-source AI content production & growth CLI.")
ref_cli = typer.Typer(help="Multi-reference video intelligence & style transfer.")
cli.add_typer(ref_cli, name="reference")


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


@ref_cli.command("library")
def cmd_ref_library(limit: int = typer.Option(100, help="Max references to list")) -> None:
    """List reusable references across the personal library."""
    studio = SymposiumStudio()
    _emit([r.model_dump(mode="json") for r in studio.list_library_references(limit=limit)])


@ref_cli.command("add")
def cmd_ref_add(
    project_id: str,
    files: list[Path] = typer.Option([], "--file", "-f", help="Local reference video path(s)"),
    workspace_paths: list[str] = typer.Option([], "--workspace-path", help="Relative file path(s) inside workspace"),
    urls: list[str] = typer.Option([], "--url", "-u", help="Reference URL(s) (YouTube/Shorts/TikTok/Reels)"),
    library_ids: list[str] = typer.Option([], "--library-id", "-l", help="Existing reference_id(s) from personal library"),
    platform: str = typer.Option("local", help="Reference platform tag"),
    creator: str = typer.Option("", help="Creator attribution"),
    allow_network: bool = typer.Option(False, help="Allow metadata/media fetch for URLs"),
) -> None:
    """Ingest local files, workspace videos, URLs, or personal library references into a project."""
    studio = SymposiumStudio()
    results: list[dict[str, Any]] = []
    cache_hits = 0
    if files:
        res_f = studio.ingest_reference_files(project_id, files, platform=platform, creator=creator)
        results.extend(res_f["references"])
        cache_hits += int(res_f["cache_hits"])
    for rel in workspace_paths:
        res_w = studio.ingest_reference_workspace_file(project_id, rel)
        cache_hits += int(bool(res_w["cache_hit"]))
        results.append({**res_w["reference"], "cache_hit": res_w["cache_hit"]})
    for u in urls:
        res_u = studio.ingest_reference_url(project_id, u, creator=creator, allow_network=allow_network)
        cache_hits += int(bool(res_u["cache_hit"]))
        results.append({**res_u["reference"], "cache_hit": res_u["cache_hit"]})
    for rid in library_ids:
        res_l = studio.import_library_reference(project_id, rid)
        cache_hits += 1
        results.append({**res_l["reference"], "cache_hit": True})
    _emit({"project_id": project_id, "ingested_count": len(results), "cache_hits": cache_hits, "references": results})


@ref_cli.command("analyze")
def cmd_ref_analyze(
    project_id: str,
    reference_ids: list[str] = typer.Option([], "--ref-id", help="Optional specific reference_id(s)"),
    run_asr: bool = typer.Option(True, "--asr/--no-asr", help="Run local Whisper ASR cadence check on references"),
    force: bool = typer.Option(False, "--force", help="Re-analyze even if StyleDNA is cached"),
) -> None:
    """Run deep CV, SceneDetect, audio SFX, and ASR analysis across project references."""
    studio = SymposiumStudio()
    _emit(
        studio.analyze_project_references(
            project_id,
            reference_ids=reference_ids or None,
            run_asr=run_asr,
            force=force,
        )
    )


@ref_cli.command("compare")
def cmd_ref_compare(
    project_id: str,
    platform: str = typer.Option("youtube_long", "--platform", help="youtube_long | youtube_shorts | tiktok | instagram_reels"),
    mode: SelectionMode = typer.Option(SelectionMode.BEST_SINGLE, "--mode", help="best_single | intelligent_hybrid | manual_override"),
    manual_id: str | None = typer.Option(None, "--manual-id", help="Manual reference_id override"),
) -> None:
    """Rank analyzed references via 13 compatibility criteria and run the AI Symposium Reference Conference."""
    studio = SymposiumStudio()
    _emit(
        studio.compare_and_select_references(
            project_id,
            target_platform=platform,
            selection_mode=mode,
            manual_reference_id=manual_id,
        )
    )


@ref_cli.command("select")
def cmd_ref_select(
    project_id: str,
    reference_id: str,
    platform: str = typer.Option("youtube_long", "--platform", help="Target deliverable platform"),
) -> None:
    """Manually select a reference and regenerate Symposium decision + StyleTransferPlan."""
    studio = SymposiumStudio()
    _emit(
        studio.compare_and_select_references(
            project_id,
            target_platform=platform,
            selection_mode=SelectionMode.MANUAL_OVERRIDE,
            manual_reference_id=reference_id,
        )
    )


@ref_cli.command("adjust")
def cmd_ref_adjust(
    project_id: str,
    pacing_multiplier: float | None = typer.Option(None, help="Explicit pacing multiplier (0.5..1.8)"),
    increase_pacing: bool = typer.Option(False, "--increase-pacing", help="Boost pacing +0.25x"),
    decrease_pacing: bool = typer.Option(False, "--decrease-pacing", help="Relax pacing -0.20x"),
    subtitle_minimal: bool = typer.Option(False, "--subtitle-minimal", help="Use minimal captions"),
    subtitle_bold: bool = typer.Option(False, "--subtitle-bold", help="Use bold/heavy captions"),
    disable_memes: bool = typer.Option(False, "--disable-memes", help="Disable meme cards"),
    enable_memes: bool = typer.Option(False, "--enable-memes", help="Enable meme cards"),
    boost_edu: bool = typer.Option(False, "--boost-edu", help="Boost educational graphics"),
    keep_structure: bool = typer.Option(False, "--keep-structure", help="Keep storyboard order, apply rhythm only"),
    platform: str | None = typer.Option(None, "--platform"),
) -> None:
    """Interactively refine StyleTransferParameters without re-analyzing references."""
    adj: dict[str, Any] = {}
    if pacing_multiplier is not None:
        adj["pacing_multiplier"] = pacing_multiplier
    if increase_pacing:
        adj["increase_pacing"] = True
    if decrease_pacing:
        adj["decrease_pacing"] = True
    if subtitle_minimal:
        adj["subtitle_minimal"] = True
    if subtitle_bold:
        adj["subtitle_bold"] = True
    if disable_memes:
        adj["disable_memes"] = True
    if enable_memes:
        adj["enable_memes"] = True
    if boost_edu:
        adj["boost_educational_graphics"] = True
    if keep_structure:
        adj["keep_structure_apply_rhythm_only"] = True
    studio = SymposiumStudio()
    _emit(studio.adjust_style_transfer_plan(project_id, adjustments=adj, target_platform=platform))


@ref_cli.command("render")
def cmd_ref_render(
    project_id: str,
    platform: str = typer.Option("youtube_long", "--platform"),
    include_shorts: bool = typer.Option(True, "--shorts/--no-shorts"),
    include_tiktok: bool = typer.Option(False, "--tiktok"),
    include_reels: bool = typer.Option(False, "--reels"),
    variants: int = typer.Option(1, "--variants"),
) -> None:
    """Render deliverables using the locked StyleTransferPlan and generate style_similarity.json."""
    studio = SymposiumStudio()
    _emit(
        studio.render_with_reference_style(
            project_id,
            target_platform=platform,
            include_shorts=include_shorts,
            include_tiktok=include_tiktok,
            include_reels=include_reels,
            short_variants_per_platform=variants,
        )
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
