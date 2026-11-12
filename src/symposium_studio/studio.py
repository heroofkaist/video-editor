"""Unified Symposium Studio application facade connecting all five core engines."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from symposium_studio.config import StudioConfig
from symposium_studio.engines.analysis import (
    align_transcript_to_audio,
    apply_transcript_corrections,
    extract_raw_mono_wav,
    run_full_media_analysis,
)
from symposium_studio.engines.analysis.semantics import detect_repetitions, score_highlights, segment_topics
from symposium_studio.engines.analysis.styles import analyze_reference_style
from symposium_studio.engines.animation import AnimationEngineRegistry, build_scene_spec
from symposium_studio.engines.distribution import (
    build_social_repurposing_bundle,
    build_telegram_campaign,
    execute_publish,
    generate_threads_posts,
)
from symposium_studio.engines.effects import download_accent_sounds, download_click_sounds, search_effects
from symposium_studio.engines.growth import GrowthIntelligenceEngine
from symposium_studio.engines.production import (
    add_click_sounds_to_timeline,
    apply_iterative_edit,
    build_timeline_from_storyboard,
    build_vertical_short_timelines,
    ensure_project_sfx,
    export_otio,
    render_thumbnail,
    render_timeline,
    run_quality_assurance,
)
from symposium_studio.engines.research import (
    MemeTrendConnector,
    ResearchIntelligenceEngine,
    download_trending_meme_assets,
    import_explicit_item,
    import_items_from_json,
    infer_platform_from_url,
    score_meme_for_beat,
)
from symposium_studio.engines.symposium import AGENT_ROSTER, SymposiumOrchestrator
from symposium_studio.models.animation import AnimationDomain, AnimationRenderResult, FactualityMode, SceneSpec
from symposium_studio.models.media import MediaAnalysisBundle
from symposium_studio.models.project import AssetRole, ProjectManifest
from symposium_studio.models.research import ContentCategory, ResearchPlatform, ResearchReport, ViralContentItem
from symposium_studio.models.social import (
    PublishingMode,
    PublishRequest,
    QualityAssuranceReport,
    SocialRepurposingBundle,
    ThreadsStyle,
)
from symposium_studio.models.symposium import ResearchMode, SymposiumSession
from symposium_studio.models.timeline import RenderTimeline, TargetPlatformFormat
from symposium_studio.storage import StudioDatabase, WorkspaceManager


class SymposiumStudio:
    """Primary programmatic entrypoint for CLI, FastAPI, and external AI agents."""

    def __init__(self, config: StudioConfig | None = None) -> None:
        self.config = config or StudioConfig.from_env()
        self.db = StudioDatabase(self.config.db_path)
        self.workspaces = WorkspaceManager(self.config, self.db)
        self.research = ResearchIntelligenceEngine(self.db)
        self.symposium = SymposiumOrchestrator(self.config)
        self.animations = AnimationEngineRegistry(self.config)
        self.growth = GrowthIntelligenceEngine(self.db)

    # ------------------------------------------------------------------
    # Projects & Workspaces
    # ------------------------------------------------------------------
    def create_project(
        self,
        name: str,
        creator_niche: str = "science_engineering",
        editing_style: str = "educational_explainer",
        tags: list[str] | None = None,
    ) -> ProjectManifest:
        return self.workspaces.create_project(
            name=name,
            creator_niche=creator_niche,
            editing_style=editing_style,
            tags=tags,
        )

    def ingest_video(self, project_id: str, video_path: Path, copy_to_workspace: bool = True) -> dict[str, Any]:
        manifest, asset = self.workspaces.ingest_source_media(project_id, video_path, copy_to_workspace=copy_to_workspace)
        return {"project": manifest.project.model_dump(mode="json"), "source_asset": asset.model_dump(mode="json")}

    def analyze_project(
        self,
        project_id: str,
        language: str | None = None,
        whisper_model: str | None = None,
    ) -> MediaAnalysisBundle:
        with self.workspaces.track_run(project_id, "analyze_project", {"language": language}) as (manifest, run):
            src = manifest.project.source_video_path
            if not src:
                raise ValueError(f"Project {project_id} has no ingested source video.")
            bundle = run_full_media_analysis(
                project_id=project_id,
                media_path=Path(src),
                whisper_model=whisper_model or self.config.whisper_model,
                language=language,
            )
            ws_dir = Path(manifest.project.workspace_dir)
            transcript_path = ws_dir / "transcripts" / "analysis_bundle.json"
            transcript_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

            # Index transcript segments into FTS5
            self.db.index_transcript_segments(
                project_id,
                [s.model_dump(mode="json") for s in bundle.transcript.segments],
            )
            asset = self.workspaces.register_asset(
                manifest,
                run_id=run.run_id,
                role=AssetRole.TRANSCRIPT,
                path=transcript_path,
                metadata={
                    "detected_language": bundle.transcript.detected_language,
                    "segments": len(bundle.transcript.segments),
                    "uncertain_segments": len(bundle.transcript.uncertain_segment_ids),
                    "highlights": len(bundle.highlights),
                },
            )
            run.output_asset_ids.append(asset.asset_id)
            manifest.project.status = "analyzed"
            self.workspaces.add_decision(
                run,
                step="media_analysis",
                actor="analysis_engine",
                decision=f"Detected language={bundle.transcript.detected_language}, {len(bundle.highlights)} highlights, crop_cx={bundle.recommended_crop_center_x}",
                rationale=f"Transcribed with {bundle.transcript.engine} ({bundle.transcript.model}); verbatim preserved.",
            )
        return bundle

    def load_analysis(self, project_id: str) -> MediaAnalysisBundle:
        manifest = self.workspaces.get_manifest(project_id)
        p = Path(manifest.project.workspace_dir) / "transcripts" / "analysis_bundle.json"
        if not p.exists():
            raise FileNotFoundError(f"No media analysis yet for {project_id}. Run analyze first.")
        return MediaAnalysisBundle.model_validate_json(p.read_text(encoding="utf-8"))

    def update_transcript_segments(
        self,
        project_id: str,
        corrections: dict[str, Any],
        language_overrides: dict[str, str] | None = None,
        mark_confirmed: bool = True,
        align_audio: bool = True,
    ) -> MediaAnalysisBundle:
        """Apply verified/manual transcript segment edits and re-index semantics without re-running ASR."""
        bundle = self.load_analysis(project_id)
        with self.workspaces.track_run(
            project_id,
            "update_transcript_segments",
            {"edited_segments": sorted(corrections.keys()), "mark_confirmed": mark_confirmed},
        ) as (manifest, run):
            bundle.transcript = apply_transcript_corrections(
                bundle.transcript,
                corrections=corrections,
                language_overrides=language_overrides,
                mark_confirmed=mark_confirmed,
            )
            src_media = Path(bundle.probe.path)
            if align_audio and src_media.exists():
                raw_wav = Path(manifest.project.workspace_dir) / "transcripts" / "source_raw_16k.wav"
                if not raw_wav.exists() or raw_wav.stat().st_mtime < src_media.stat().st_mtime:
                    extract_raw_mono_wav(src_media, raw_wav)
                target_ids = set(corrections.keys()) if corrections else None
                bundle.transcript = align_transcript_to_audio(
                    bundle.transcript,
                    raw_wav,
                    segment_ids=target_ids,
                    force_all=True,
                )
            bundle.topics = segment_topics(bundle.transcript.segments)
            bundle.repetitions_flagged = detect_repetitions(bundle.transcript.segments)
            bundle.highlights = score_highlights(
                bundle.transcript.segments,
                bundle.topics,
                bundle.visual_samples,
                bundle.probe.duration_sec,
            )
            ws_dir = Path(manifest.project.workspace_dir)
            transcript_path = ws_dir / "transcripts" / "analysis_bundle.json"
            transcript_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

            self.db.index_transcript_segments(
                project_id,
                [s.model_dump(mode="json") for s in bundle.transcript.segments],
            )
            asset = self.workspaces.register_asset(
                manifest,
                run_id=run.run_id,
                role=AssetRole.TRANSCRIPT,
                path=transcript_path,
                metadata={
                    "detected_language": bundle.transcript.detected_language,
                    "segments": len(bundle.transcript.segments),
                    "uncertain_segments": len(bundle.transcript.uncertain_segment_ids),
                    "highlights": len(bundle.highlights),
                    "human_confirmed_segments": sum(
                        1 for s in bundle.transcript.segments if s.verification_status == "human_confirmed"
                    ),
                },
            )
            run.output_asset_ids.append(asset.asset_id)
            self.workspaces.add_decision(
                run,
                step="transcript_correction",
                actor="human_editor" if mark_confirmed else "transcript_verifier",
                decision=(
                    f"Updated {len(corrections)} transcript segment(s) "
                    f"and recomputed {len(bundle.highlights)} highlights"
                ),
                rationale=(
                    "Propagated edits to word timestamps, FTS5 search index, topics, "
                    "and highlight windows without re-running ASR."
                ),
            )
        return bundle

    # ------------------------------------------------------------------
    # Research & Symposium
    # ------------------------------------------------------------------
    def discover_live_research(
        self,
        target_count: int = 100,
        categories: list[str] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Discover real public multi-niche creator videos via official RSS / Data API v3."""
        return self.research.discover_cohort(target_count=target_count, categories=categories, query=query)

    def inspect_research_url(self, url: str) -> ViralContentItem | None:
        return self.research.inspect_youtube_url(url)

    def apply_reference_style(self, project_id: str, reference_video_path: Path) -> dict[str, Any]:
        """Analyze a reference edit's tempo/luminance/saturation signature and apply its matched EditingStyleProfile."""
        profile = analyze_reference_style(reference_video_path)
        with self.workspaces.track_run(
            project_id, "apply_reference_style", {"reference_video": str(reference_video_path)}
        ) as (manifest, run):
            manifest.project.editing_style = profile.name.value
            self.workspaces.add_decision(
                run,
                step="style_clone",
                actor="creative_director",
                decision=f"Applied cloned editing style `{profile.name.value}` ({profile.display_name})",
                rationale=f"Matched cut interval / visual signature from {reference_video_path.name}.",
            )
        return {"project_id": project_id, "editing_style": profile.name.value, "profile": profile.model_dump(mode="json")}

    def run_research_report(
        self,
        project_id: str,
        report_type: str = "content_proposal",
        topic_keywords: list[str] | None = None,
    ) -> ResearchReport:
        with self.workspaces.track_run(project_id, "run_research_report", {"report_type": report_type}) as (manifest, run):
            report = self.research.generate_research_report(
                report_type=report_type,
                target_niche=manifest.project.creator_niche,
                topic_keywords=topic_keywords,
            )
            ws_dir = Path(manifest.project.workspace_dir)
            out_path = ws_dir / "research" / f"{report.report_id}.json"
            out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            asset = self.workspaces.register_asset(
                manifest,
                run_id=run.run_id,
                role=AssetRole.RESEARCH_REPORT,
                path=out_path,
            )
            run.output_asset_ids.append(asset.asset_id)
        return report

    def run_symposium_conference(
        self,
        project_id: str,
        mode: ResearchMode = ResearchMode.STANDARD,
        style_name: str | None = None,
        target_min_sec: float = 60.0,
        target_max_sec: float = 90.0,
    ) -> SymposiumSession:
        analysis = self.load_analysis(project_id)
        manifest = self.workspaces.get_manifest(project_id)
        keywords = [k for t in analysis.topics for k in t.keywords][:6]
        rep_files = sorted((Path(manifest.project.workspace_dir) / "research").glob("*.json"))
        if rep_files:
            report = ResearchReport.model_validate_json(rep_files[-1].read_text(encoding="utf-8"))
        else:
            report = self.run_research_report(project_id, "content_proposal", keywords)

        with self.workspaces.track_run(project_id, "run_symposium", {"mode": mode.value}) as (manifest, run):
            prior_exps = self.db.list_experiments(limit=300)
            research_pool = self.db.list_viral_items(limit=120)
            session = self.symposium.run_symposium(
                project_id=project_id,
                analysis=analysis,
                research_report=report,
                style_name=style_name or manifest.project.editing_style,
                mode=mode,
                target_min_sec=target_min_sec,
                target_max_sec=target_max_sec,
                prior_experiments=prior_exps,
                research_items=research_pool,
            )
            ws_dir = Path(manifest.project.workspace_dir)
            sym_path = ws_dir / "symposium" / f"{session.session_id}.json"
            sym_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
            sb_path = ws_dir / "storyboards" / f"{session.storyboard.storyboard_id}.json"
            sb_path.write_text(session.storyboard.model_dump_json(indent=2), encoding="utf-8")

            self.growth.record_experiment_from_production(
                project_id=project_id,
                title=manifest.project.name,
                creator_niche=manifest.project.creator_niche,
                session=session,
                research_report=report,
            )

            a1 = self.workspaces.register_asset(manifest, run.run_id, AssetRole.SYMPOSIUM_LOG, sym_path)
            a2 = self.workspaces.register_asset(manifest, run.run_id, AssetRole.STORYBOARD, sb_path)
            run.output_asset_ids.extend([a1.asset_id, a2.asset_id])
            for dec in session.decisions:
                self.workspaces.add_decision(
                    run,
                    step="symposium_decision",
                    actor="creative_director",
                    decision=f"{dec.topic}: {dec.chosen_option}",
                    rationale=dec.rationale,
                    evidence_ids=dec.evidence_refs,
                )
        return session

    # ------------------------------------------------------------------
    # Original Animations
    # ------------------------------------------------------------------
    def generate_animation(
        self,
        project_id: str,
        template_id: str = "em_induction_faraday",
        title: str | None = None,
        duration_sec: float = 7.0,
        width: int = 1920,
        height: int = 1080,
        transparent_background: bool = True,
        preferred_adapter: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> AnimationRenderResult:
        with self.workspaces.track_run(project_id, "generate_animation", {"template_id": template_id}) as (manifest, run):
            spec = build_scene_spec(
                template_id=template_id,
                title=title,
                duration_sec=duration_sec,
                width=width,
                height=height,
                transparent_background=transparent_background,
                preferred_adapter=preferred_adapter,
                parameters=parameters,
            )
            anim_dir = Path(manifest.project.workspace_dir) / "animations"
            res = self.animations.render_scene(spec, anim_dir)
            a_spec = self.workspaces.register_asset(
                manifest, run.run_id, AssetRole.ANIMATION_SPEC, Path(res.scene_spec_path)
            )
            a_vid = self.workspaces.register_asset(
                manifest,
                run.run_id,
                AssetRole.ANIMATION_RENDER,
                Path(res.output_path),
                metadata={"adapter": res.adapter_used, "factuality_mode": res.factuality_mode.value},
            )
            run.output_asset_ids.extend([a_spec.asset_id, a_vid.asset_id])
        return res

    # ------------------------------------------------------------------
    # Timelines, Rendering & QA
    # ------------------------------------------------------------------
    def _latest_symposium_and_anim(self, manifest: ProjectManifest) -> tuple[SymposiumSession, AnimationRenderResult | None]:
        ws_dir = Path(manifest.project.workspace_dir)
        sym_files = sorted(
            (ws_dir / "symposium").glob("sym_*.json"),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )
        if not sym_files:
            raise FileNotFoundError("No symposium session found; run symposium first.")
        session = SymposiumSession.model_validate_json(sym_files[-1].read_text(encoding="utf-8"))

        anim_specs = sorted(
            (ws_dir / "animations").glob("anim_*.spec.json"),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )
        anim_res: AnimationRenderResult | None = None
        if anim_specs:
            spec = SceneSpec.model_validate_json(anim_specs[-1].read_text(encoding="utf-8"))
            mov = ws_dir / "animations" / f"{spec.scene_id}.mov"
            mp4 = ws_dir / "animations" / f"{spec.scene_id}.mp4"
            out_file = mov if mov.exists() else mp4
            if out_file.exists():
                anim_res = AnimationRenderResult(
                    scene_id=spec.scene_id,
                    adapter_used=spec.preferred_adapter or "chromium_canvas",
                    output_path=str(out_file),
                    has_alpha=mov.exists(),
                    width=spec.width,
                    height=spec.height,
                    fps=float(spec.fps),
                    duration_sec=spec.duration_sec,
                    scene_spec_path=str(anim_specs[-1]),
                    render_time_sec=0.0,
                    factuality_mode=spec.factuality_mode,
                )
        return session, anim_res

    def _meme_batch_animations(self, ws_dir: Path, max_count: int = 10) -> list[AnimationRenderResult]:
        anim_dir = ws_dir / "animations"
        if not anim_dir.exists():
            return []
        found: list[AnimationRenderResult] = []
        for sp in sorted(anim_dir.glob("anim_*.spec.json")):
            try:
                spec = SceneSpec.model_validate_json(sp.read_text(encoding="utf-8"))
                is_beat_callout = spec.template_id == "kinetic_callout" and "beat_index" in spec.parameters
                if spec.template_id not in {"meme_reaction_card", "meme_cutaway"} and not is_beat_callout:
                    continue
                out_file = next(
                    (
                        anim_dir / f"{spec.scene_id}{ext}"
                        for ext in (".mp4", ".mov", ".png", ".gif")
                        if (anim_dir / f"{spec.scene_id}{ext}").exists()
                    ),
                    None,
                )
                if out_file is not None:
                    found.append(
                        AnimationRenderResult(
                            scene_id=spec.scene_id,
                            adapter_used=spec.preferred_adapter or "chromium_canvas",
                            output_path=str(out_file),
                            has_alpha=out_file.suffix.lower() in {".mov", ".png"},
                            width=spec.width,
                            height=spec.height,
                            fps=float(spec.fps),
                            duration_sec=spec.duration_sec,
                            scene_spec_path=str(sp),
                            render_time_sec=0.0,
                            factuality_mode=spec.factuality_mode,
                        )
                    )
            except Exception:
                continue
        return found[-max_count:]

    @staticmethod
    def _compact_excerpt(text: str, max_words: int = 8, max_chars: int = 52) -> str:
        cleaned = " ".join((text or "").replace("\n", " ").split()).strip(" ,;:-")
        if not cleaned:
            return ""
        for sep in (". ", "! ", "? ", " — "):
            if sep in cleaned and cleaned.index(sep) >= 10:
                cleaned = cleaned.split(sep, 1)[0].strip()
                break
        words = cleaned.split()
        out_words: list[str] = []
        for w in words[:max_words]:
            cand = " ".join(out_words + [w])
            if out_words and len(cand) > max_chars:
                break
            out_words.append(w)
        out = " ".join(out_words).strip()
        if len(words) > len(out_words) and out and not out.endswith((".", "!", "?", "...")):
            out = f"{out}..."
        return out

    def _build_meme_card_text(
        self,
        slot_idx: int,
        beat_excerpt: str,
        meme_topic: str,
        beat_purpose: str = "",
    ) -> tuple[str, str, str]:
        low = (beat_excerpt or "").lower()
        purpose_low = (beat_purpose or "").lower()
        accents = ["#FACC15", "#38BDF8", "#F43F5E", "#A855F7"]
        accent = accents[slot_idx % len(accents)]
        short = self._compact_excerpt(beat_excerpt, max_words=8, max_chars=52)
        if not short:
            short = self._compact_excerpt(meme_topic, max_words=6, max_chars=44) or "KEY MOMENT"

        if any(k in low for k in ("ожида", "реальн", "салыстыр", "expect")):
            top_label = "EXPECTATIONS vs REALITY"
        elif "hook" in purpose_low or slot_idx == 0:
            top_label = "POV: HOOK"
        elif any(k in low for k in ("қиын", "сложн", "тяжел", "problem", "quiz", "deadline", "страш")):
            top_label = "REALITY CHECK"
        elif any(
            k in low
            for k in ("зараб", "стипенд", "gpa", "dean", "дисциплин", "үміт", "надеюсь", "ойлаймын", "goal", "мамандық", "резюме")
        ):
            top_label = "LOCK IN MODE"
        elif "payoff" in purpose_low or "ending" in purpose_low or any(k in low for k in ("туған күн", "екі жыл", "рахмет", "хорошо")):
            top_label = "PAYOFF"
        else:
            top_label = "KEY POINT"
        return top_label, short, accent

    def _generate_zoomer_meme_cards(
        self,
        manifest: ProjectManifest,
        analysis: MediaAnalysisBundle,
        session: SymposiumSession,
        max_cards: int = 10,
    ) -> tuple[list[AnimationRenderResult], list[dict[str, Any]]]:
        ws_dir = Path(manifest.project.workspace_dir)
        pid = manifest.project.project_id
        beats = session.storyboard.beats
        hl_map = {h.highlight_id: h.transcript_excerpt for h in analysis.highlights}

        def _beat_excerpt(beat_start: float, beat_end: float, evidence_ids: list[str]) -> str:
            for eid in evidence_ids:
                if eid in hl_map and hl_map[eid]:
                    return hl_map[eid]
            if analysis.transcript.segments:
                overlapping = [
                    s.text
                    for s in analysis.transcript.segments
                    if s.end_sec >= beat_start and s.start_sec <= beat_end
                ]
                if overlapping:
                    return " ".join(overlapping[:2])
                return analysis.transcript.segments[0].text
            return ""

        slots: list[tuple[int, float, float, str, str, str]] = []
        if not beats:
            seg0 = analysis.transcript.segments[0] if analysis.transcript.segments else None
            b_s = seg0.start_sec if seg0 else 0.0
            b_e = min(analysis.probe.duration_sec, b_s + 8.0)
            slots.append((0, b_s, b_e, "hook", "", seg0.text if seg0 else manifest.project.name))
        else:
            # Cover all narrative beats first, then subdivide longer beats into ~7-9s reaction windows up to max_cards
            for bi, b in enumerate(beats):
                if len(slots) >= max_cards:
                    break
                b_dur = max(1.0, b.source_end_sec - b.source_start_sec)
                sub_parts = 1
                if max_cards > len(beats) and b_dur >= 12.0:
                    sub_parts = min(3, max(2, int(round(b_dur / 8.5))))
                step = b_dur / sub_parts
                for sp_i in range(sub_parts):
                    if len(slots) >= max_cards:
                        break
                    s0 = round(b.source_start_sec + sp_i * step, 3)
                    s1 = round(b.source_start_sec + (sp_i + 1) * step, 3)
                    ex = _beat_excerpt(s0, s1, b.evidence_ids if sp_i == 0 else [])
                    slots.append((bi, s0, s1, b.role, b.visual_treatment, ex))

        beat_text_blob = " ".join(s[5] for s in slots)
        full_text = " ".join(s.text for s in analysis.transcript.segments)
        query_seed = (
            f"{manifest.project.name} "
            f"{' '.join(k for t in analysis.topics for k in t.keywords[:6])} "
            f"{beat_text_blob[:600]} {full_text[:500]}"
        )
        live_memes = MemeTrendConnector().fetch_public_items(query=query_seed, limit=max(28, max_cards * 3))
        for it in live_memes:
            self.db.upsert_viral_item(it)
        downloaded = download_trending_meme_assets(live_memes, ws_dir / "memes", max_count=max(max_cards * 2, 16))

        anim_dir = ws_dir / "animations"
        anim_dir.mkdir(parents=True, exist_ok=True)
        rendered_anims: list[AnimationRenderResult] = []
        used_meme_indices: set[int] = set()

        for slot_i, (b_idx, b_start, b_end, b_purpose, b_vis, excerpt) in enumerate(slots):
            best_idx: int | None = None
            best_score = 0.0
            match_context = f"{excerpt} {b_vis}".strip()
            for cand_i, cand in enumerate(downloaded):
                if cand_i in used_meme_indices:
                    continue
                cw = int(cand.get("width") or 0)
                ch = int(cand.get("height") or 0)
                if cw > 0 and ch > 0 and ch > cw * 1.45 and cw < 500:
                    continue
                sc = score_meme_for_beat(cand, match_context)
                if sc > best_score:
                    best_score = sc
                    best_idx = cand_i

            best_cand = downloaded[best_idx] if best_idx is not None else {}
            accept_thresh = 0.22 if best_cand.get("media_kind") == "video" else 0.34
            if best_idx is not None and best_score >= accept_thresh:
                meme_meta = best_cand
            else:
                # Prefer any crisp non-mismatched animated reaction video before falling back to static text cards
                vid_fallback = next(
                    (
                        (ci, cd)
                        for ci, cd in enumerate(downloaded)
                        if ci not in used_meme_indices
                        and cd.get("media_kind") == "video"
                        and score_meme_for_beat(cd, match_context) >= 0.16
                    ),
                    None,
                )
                if vid_fallback is not None:
                    best_idx, meme_meta = vid_fallback
                else:
                    meme_meta = {}
            asset_raw = str(meme_meta.get("asset_path") or meme_meta.get("image_path") or "").strip()
            meme_topic = str(meme_meta.get("topic") or meme_meta.get("title") or "Visual Beat")
            src_asset = Path(asset_raw) if asset_raw else None

            if src_asset is not None and src_asset.is_file() and best_idx is not None:
                used_meme_indices.add(best_idx)
                scene_id = f"anim_m_{slot_i}_{uuid.uuid4().hex[:6]}"
                ext = src_asset.suffix.lower() or ".mp4"
                dest_asset = anim_dir / f"{scene_id}{ext}"
                shutil.copy2(src_asset, dest_asset)
                dur_val = float(meme_meta.get("duration_sec") or 2.2)
                dur_val = round(max(1.5, min(3.2, dur_val)), 3)
                spec = SceneSpec(
                    scene_id=scene_id,
                    template_id="meme_cutaway",
                    domain=AnimationDomain.STORYTELLING,
                    title=f"Meme Cutaway #{slot_i + 1}: {meme_topic[:44]}",
                    duration_sec=dur_val,
                    width=int(meme_meta.get("width") or 640),
                    height=int(meme_meta.get("height") or 480),
                    fps=30,
                    transparent_background=False,
                    factuality_mode=FactualityMode.ARTISTIC_ILLUSTRATION,
                    preferred_adapter="meme_cutaway",
                    parameters={
                        "meme_name": meme_topic[:48],
                        "media_kind": str(meme_meta.get("media_kind") or "video"),
                        "source_url": str(meme_meta.get("original_url") or ""),
                        "match_score": best_score,
                        "beat_index": b_idx,
                        "beat_source_start": b_start,
                        "beat_source_end": b_end,
                    },
                )
                spec_path = anim_dir / f"{scene_id}.spec.json"
                spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
                anim = AnimationRenderResult(
                    scene_id=scene_id,
                    adapter_used="meme_cutaway",
                    output_path=str(dest_asset.resolve()),
                    has_alpha=ext in {".png", ".mov"},
                    width=spec.width,
                    height=spec.height,
                    fps=30.0,
                    duration_sec=dur_val,
                    scene_spec_path=str(spec_path.resolve()),
                    render_time_sec=0.0,
                    factuality_mode=FactualityMode.ARTISTIC_ILLUSTRATION,
                )
                rendered_anims.append(anim)
            else:
                top_label, reaction_text, _accent = self._build_meme_card_text(
                    slot_i, excerpt, meme_topic, beat_purpose=b_purpose
                )
                anim = self.generate_animation(
                    pid,
                    template_id="kinetic_callout",
                    title=f"Beat Callout #{slot_i + 1}: {top_label}",
                    duration_sec=2.6,
                    width=960,
                    height=540,
                    transparent_background=True,
                    preferred_adapter="chromium_canvas",
                    parameters={
                        "headline": top_label,
                        "subtitle": reaction_text,
                        "beat_index": b_idx,
                        "beat_source_start": b_start,
                        "beat_source_end": b_end,
                    },
                )
                rendered_anims.append(anim)
        return rendered_anims, downloaded

    def build_and_render_deliverables(
        self,
        project_id: str,
        include_shorts: bool = True,
        include_tiktok: bool = True,
        include_reels: bool = True,
        short_variants_per_platform: int = 1,
        extra_animations: list[AnimationRenderResult] | None = None,
        download_click_sounds: bool | None = None,
        animation_result: AnimationRenderResult | None = None,
    ) -> dict[str, Any]:
        analysis = self.load_analysis(project_id)
        manifest = self.workspaces.get_manifest(project_id)
        session, latest_anim = self._latest_symposium_and_anim(manifest)
        anim_res = animation_result or latest_anim
        ws_dir = Path(manifest.project.workspace_dir)

        resolved_clicks = (
            manifest.project.editing_style in {"zoomer_vlog", "short_form_viral", "comedy"}
            if download_click_sounds is None
            else bool(download_click_sounds)
        )
        resolved_extra = list(extra_animations or [])
        if manifest.project.editing_style == "zoomer_vlog" and (
            extra_animations is None
            or (anim_res is not None and any(a.scene_id == anim_res.scene_id for a in resolved_extra))
        ):
            meme_batch = self._meme_batch_animations(ws_dir, max_count=10)
            if meme_batch:
                anim_res = meme_batch[0]
                resolved_extra = meme_batch[1:]
        elif anim_res is not None:
            resolved_extra = [a for a in resolved_extra if a.scene_id != anim_res.scene_id]

        with self.workspaces.track_run(project_id, "build_and_render_deliverables") as (manifest, run):
            sfx_map = ensure_project_sfx(ws_dir / "effects", download_clicks=resolved_clicks)
            # 1. Main YouTube 16:9 timeline & render
            yt_tl = build_timeline_from_storyboard(
                project_id=project_id,
                title=manifest.project.name,
                analysis=analysis,
                storyboard=session.storyboard,
                style_name=manifest.project.editing_style,
                animation_result=anim_res,
                platform_format=TargetPlatformFormat.YOUTUBE_LONG,
                sfx_paths=sfx_map,
                extra_overlays=resolved_extra,
            )
            yt_tl_json = ws_dir / "timelines" / f"{yt_tl.timeline_id}_youtube.json"
            yt_tl_json.write_text(yt_tl.model_dump_json(indent=2), encoding="utf-8")
            yt_otio = ws_dir / "timelines" / f"{yt_tl.timeline_id}_youtube.otio"
            export_otio(yt_tl, yt_otio)

            yt_mp4 = ws_dir / "renders" / f"{manifest.project.slug}_youtube_1080p.mp4"
            render_files = render_timeline(yt_tl, yt_mp4, ws_dir)
            if isinstance(render_files.get("rendered_timeline"), RenderTimeline):
                yt_tl = render_files["rendered_timeline"]
                yt_tl_json.write_text(yt_tl.model_dump_json(indent=2), encoding="utf-8")
            thumb_png = ws_dir / "thumbnails" / f"{manifest.project.slug}_thumbnail.png"
            thumb_capture = yt_tl.clips[0].source_start_sec + min(2.0, max(0.5, (yt_tl.clips[0].source_end_sec - yt_tl.clips[0].source_start_sec) * 0.25))
            render_thumbnail(
                Path(analysis.probe.path),
                headline=manifest.project.name,
                badge=manifest.project.creator_niche.replace("_", " "),
                out_png=thumb_png,
                capture_sec=thumb_capture,
            )

            yt_qa = run_quality_assurance(
                project_id=project_id,
                video_path=yt_mp4,
                timeline=yt_tl,
                frames_out_dir=ws_dir / "reports" / "qa_frames",
                analysis=analysis,
            )
            qa_path = ws_dir / "reports" / f"qa_{yt_tl.timeline_id}.json"
            qa_path.write_text(yt_qa.model_dump_json(indent=2), encoding="utf-8")

            self.workspaces.register_asset(manifest, run.run_id, AssetRole.TIMELINE_JSON, yt_tl_json)
            self.workspaces.register_asset(manifest, run.run_id, AssetRole.TIMELINE_OTIO, yt_otio)
            self.workspaces.register_asset(manifest, run.run_id, AssetRole.RENDER_YOUTUBE, yt_mp4)
            self.workspaces.register_asset(manifest, run.run_id, AssetRole.THUMBNAIL, thumb_png)
            self.workspaces.register_asset(manifest, run.run_id, AssetRole.CAPTION_FILE, Path(render_files["srt_path"]))
            self.workspaces.register_asset(manifest, run.run_id, AssetRole.QA_REPORT, qa_path)

            # 2. Platform-specific 9:16 vertical deliverables
            short_outputs: dict[str, list[dict[str, Any]]] = {}
            platform_specs: list[tuple[bool, TargetPlatformFormat, AssetRole]] = [
                (include_shorts, TargetPlatformFormat.YOUTUBE_SHORTS, AssetRole.RENDER_SHORT),
                (include_tiktok, TargetPlatformFormat.TIKTOK, AssetRole.RENDER_TIKTOK),
                (include_reels, TargetPlatformFormat.INSTAGRAM_REELS, AssetRole.RENDER_REEL),
            ]
            short_style = "zoomer_vlog" if manifest.project.editing_style == "zoomer_vlog" else "short_form_viral"
            for enabled, fmt, role in platform_specs:
                if not enabled:
                    continue
                v_tls = build_vertical_short_timelines(
                    project_id=project_id,
                    base_title=manifest.project.name,
                    analysis=analysis,
                    animation_result=anim_res,
                    platform_format=fmt,
                    max_variants=short_variants_per_platform,
                    sfx_paths=sfx_map,
                    style_name=short_style,
                    extra_overlays=resolved_extra,
                )
                short_outputs[fmt.value] = []
                for idx, vtl in enumerate(v_tls):
                    v_json = ws_dir / "timelines" / f"{vtl.timeline_id}_{fmt.value}.json"
                    v_json.write_text(vtl.model_dump_json(indent=2), encoding="utf-8")
                    v_otio = ws_dir / "timelines" / f"{vtl.timeline_id}_{fmt.value}.otio"
                    export_otio(vtl, v_otio)
                    v_mp4 = ws_dir / "shorts" / f"{manifest.project.slug}_{fmt.value}_{idx + 1}.mp4"
                    v_render_files = render_timeline(vtl, v_mp4, ws_dir)
                    if isinstance(v_render_files.get("rendered_timeline"), RenderTimeline):
                        vtl = v_render_files["rendered_timeline"]
                        v_json.write_text(vtl.model_dump_json(indent=2), encoding="utf-8")
                    v_qa = run_quality_assurance(
                        project_id,
                        v_mp4,
                        vtl,
                        ws_dir / "reports" / "qa_frames",
                        analysis=analysis,
                    )
                    self.workspaces.register_asset(manifest, run.run_id, role, v_mp4)
                    self.workspaces.register_asset(manifest, run.run_id, AssetRole.TIMELINE_OTIO, v_otio)
                    short_outputs[fmt.value].append(
                        {
                            "timeline_id": vtl.timeline_id,
                            "video_path": str(v_mp4),
                            "otio_path": str(v_otio),
                            "qa": v_qa.model_dump(mode="json"),
                        }
                    )
            manifest.project.status = "rendered"

        return {
            "youtube": {
                "timeline": yt_tl.model_dump(mode="json"),
                "video_path": str(yt_mp4),
                "otio_path": str(yt_otio),
                "thumbnail_path": str(thumb_png),
                "srt_path": render_files["srt_path"],
                "ass_path": render_files["ass_path"],
                "qa": yt_qa.model_dump(mode="json"),
            },
            "vertical_variants": short_outputs,
        }

    def add_click_sounds(
        self,
        project_id: str,
        custom_urls: dict[str, str] | None = None,
        search_query: str | None = None,
        allow_network: bool = True,
        render_video: bool = True,
        include_shorts: bool = True,
        min_spacing_sec: float | None = None,
    ) -> dict[str, Any]:
        """Download real internet click SFX and insert them at punch cuts, overlays, and emphasis beats."""
        manifest = self.workspaces.get_manifest(project_id)
        ws_dir = Path(manifest.project.workspace_dir)
        dl = download_click_sounds(
            ws_dir / "effects",
            custom_urls=custom_urls,
            search_query=search_query,
            allow_network=allow_network,
        )
        if allow_network:
            download_accent_sounds(ws_dir / "effects", allow_network=True)
        click_paths = dl["click_paths"]

        tl_files = sorted((ws_dir / "timelines").glob("*_youtube.json"))
        if not tl_files:
            raise FileNotFoundError("No YouTube timeline found yet to add click sounds.")
        yt_json_path = tl_files[-1]
        yt_tl = RenderTimeline.model_validate_json(yt_json_path.read_text(encoding="utf-8"))
        updated_yt, yt_placements = add_click_sounds_to_timeline(
            yt_tl,
            click_paths,
            style_name=manifest.project.editing_style,
            min_spacing_sec=min_spacing_sec,
            replace_existing=True,
        )
        yt_json_path.write_text(updated_yt.model_dump_json(indent=2), encoding="utf-8")
        yt_otio_path = ws_dir / "timelines" / f"{updated_yt.timeline_id}_youtube.otio"
        export_otio(updated_yt, yt_otio_path)

        yt_mp4 = ws_dir / "renders" / f"{manifest.project.slug}_youtube_1080p.mp4"
        yt_qa_payload: dict[str, Any] | None = None
        if render_video:
            yt_rf = render_timeline(updated_yt, yt_mp4, ws_dir)
            if isinstance(yt_rf.get("rendered_timeline"), RenderTimeline):
                updated_yt = yt_rf["rendered_timeline"]
            yt_qa = run_quality_assurance(project_id, yt_mp4, updated_yt, ws_dir / "reports" / "qa_frames")
            (ws_dir / "reports" / f"qa_{updated_yt.timeline_id}.json").write_text(
                yt_qa.model_dump_json(indent=2), encoding="utf-8"
            )
            yt_qa_payload = yt_qa.model_dump(mode="json")

        short_updates: list[dict[str, Any]] = []
        if include_shorts:
            otio_assets = [a for a in manifest.assets if a.role == AssetRole.TIMELINE_OTIO]
            for fmt in ("youtube_shorts", "tiktok", "instagram_reels"):
                matched_jsons: list[Path] = []
                seen_stems: set[str] = set()
                for asset in otio_assets:
                    ap = Path(asset.absolute_path or asset.relative_path)
                    if ap.name.endswith(f"_{fmt}.otio"):
                        jp = ws_dir / "timelines" / f"{ap.stem}.json"
                        if jp.is_file() and jp.stem not in seen_stems:
                            matched_jsons.append(jp)
                            seen_stems.add(jp.stem)
                if not matched_jsons:
                    matched_jsons = sorted((ws_dir / "timelines").glob(f"*_{fmt}.json"), key=lambda p: p.stat().st_mtime)
                short_style = "zoomer_vlog" if manifest.project.editing_style == "zoomer_vlog" else "short_form_viral"
                for idx, s_json in enumerate(matched_jsons):
                    stl = RenderTimeline.model_validate_json(s_json.read_text(encoding="utf-8"))
                    ustl, s_placements = add_click_sounds_to_timeline(
                        stl,
                        click_paths,
                        style_name=short_style,
                        min_spacing_sec=min_spacing_sec,
                        replace_existing=True,
                    )
                    s_json.write_text(ustl.model_dump_json(indent=2), encoding="utf-8")
                    export_otio(ustl, ws_dir / "timelines" / f"{ustl.timeline_id}_{fmt}.otio")
                    s_mp4 = ws_dir / "shorts" / f"{manifest.project.slug}_{fmt}_{idx + 1}.mp4"
                    if render_video:
                        s_rf = render_timeline(ustl, s_mp4, ws_dir)
                        if isinstance(s_rf.get("rendered_timeline"), RenderTimeline):
                            ustl = s_rf["rendered_timeline"]
                    short_updates.append(
                        {
                            "platform": fmt,
                            "variant": idx + 1,
                            "timeline_id": ustl.timeline_id,
                            "click_count": len(s_placements),
                            "video_path": str(s_mp4),
                        }
                    )

        with self.workspaces.track_run(project_id, "add_click_sounds") as (m2, run):
            src_manifest = Path(dl["manifest_path"])
            if src_manifest.is_file():
                a_src = self.workspaces.register_asset(m2, run.run_id, AssetRole.SFX_ASSET, src_manifest)
                run.output_asset_ids.append(a_src.asset_id)

        return {
            "command": "add_click_sounds",
            "downloaded_count": dl["downloaded_count"],
            "cached_count": dl["cached_count"],
            "fallback_count": dl["fallback_count"],
            "sources": dl["sources"],
            "sfx_manifest_path": dl["manifest_path"],
            "youtube_click_count": len(yt_placements),
            "placements": yt_placements,
            "revision": updated_yt.revision,
            "video_path": str(yt_mp4) if render_video else None,
            "qa": yt_qa_payload,
            "shorts_updated": short_updates,
        }

    # ------------------------------------------------------------------
    # Iterative Agent Instructions
    # ------------------------------------------------------------------
    def apply_agent_instruction(self, project_id: str, instruction: str) -> dict[str, Any]:
        """Support natural-language iterative commands (edit timeline, click SFX, animation, shorts, Threads/Telegram)."""
        low = instruction.lower()
        manifest = self.workspaces.get_manifest(project_id)
        ws_dir = Path(manifest.project.workspace_dir)

        if "click" in low or "sfx" in low or ("sound" in low and ("add" in low or "download" in low)):
            return self.add_click_sounds(project_id, allow_network=True, render_video=True)

        if "animation" in low and ("create" in low or "generate" in low or "show" in low):
            tmpl = "em_induction_faraday"
            if "meme" in low or "zoomer" in low:
                tmpl = "meme_reaction_card"
            elif "circuit" in low or "voltage" in low or "signal" in low:
                tmpl = "rlc_signal_flow"
            elif "tree" in low or "algorithm" in low or "data structure" in low:
                tmpl = "binary_tree_trace"
            elif "architecture" in low or "pipeline" in low or "microservice" in low:
                tmpl = "system_architecture_flow"
            elif "wave" in low or "interference" in low or "superposition" in low:
                tmpl = "wave_interference_sim"
            elif "timeline" in low or "story" in low:
                tmpl = "narrative_timeline"
            anim = self.generate_animation(project_id, template_id=tmpl)
            return {"command": "generate_animation", "animation": anim.model_dump(mode="json")}

        if "three shorts" in low or "3 shorts" in low or ("turn" in low and "short" in low):
            analysis = self.load_analysis(project_id)
            _, anim_res = self._latest_symposium_and_anim(manifest)
            sfx_map = ensure_project_sfx(ws_dir / "effects")
            extra_memes = (
                self._meme_batch_animations(ws_dir, max_count=3)
                if manifest.project.editing_style == "zoomer_vlog"
                else []
            )
            v_tls = build_vertical_short_timelines(
                project_id=project_id,
                base_title=manifest.project.name,
                analysis=analysis,
                animation_result=extra_memes[0] if extra_memes else anim_res,
                platform_format=TargetPlatformFormat.YOUTUBE_SHORTS,
                max_variants=3,
                sfx_paths=sfx_map,
                style_name="zoomer_vlog" if manifest.project.editing_style == "zoomer_vlog" else "short_form_viral",
                extra_overlays=extra_memes[1:] if len(extra_memes) > 1 else None,
            )
            rendered = []
            for idx, vtl in enumerate(v_tls):
                out_mp4 = ws_dir / "shorts" / f"{manifest.project.slug}_iter_short_{idx + 1}.mp4"
                render_timeline(vtl, out_mp4, ws_dir)
                rendered.append({"timeline_id": vtl.timeline_id, "video_path": str(out_mp4)})
            return {"command": "create_three_shorts", "shorts": rendered}

        if "threads" in low or "telegram" in low:
            bundle = self.generate_social_and_telegram(project_id)
            return {"command": "social_and_telegram", "bundle": bundle.model_dump(mode="json")}

        # Timeline edit + re-render latest YouTube timeline
        tl_files = sorted((ws_dir / "timelines").glob("*_youtube.json"))
        if not tl_files:
            raise FileNotFoundError("No YouTube timeline found yet to modify.")
        tl = RenderTimeline.model_validate_json(tl_files[-1].read_text(encoding="utf-8"))
        updated_tl, changes = apply_iterative_edit(tl, instruction)
        tl_files[-1].write_text(updated_tl.model_dump_json(indent=2), encoding="utf-8")
        export_otio(updated_tl, ws_dir / "timelines" / f"{updated_tl.timeline_id}_youtube.otio")
        yt_mp4 = ws_dir / "renders" / f"{manifest.project.slug}_youtube_1080p.mp4"
        it_rf = render_timeline(updated_tl, yt_mp4, ws_dir)
        if isinstance(it_rf.get("rendered_timeline"), RenderTimeline):
            updated_tl = it_rf["rendered_timeline"]
            tl_files[-1].write_text(updated_tl.model_dump_json(indent=2), encoding="utf-8")
        qa = run_quality_assurance(project_id, yt_mp4, updated_tl, ws_dir / "reports" / "qa_frames")
        return {
            "command": "edit_timeline",
            "changes": changes,
            "revision": updated_tl.revision,
            "video_path": str(yt_mp4),
            "qa": qa.model_dump(mode="json"),
        }

    # ------------------------------------------------------------------
    # Social & Telegram Funnel + Report
    # ------------------------------------------------------------------
    def generate_social_and_telegram(
        self,
        project_id: str,
        telegram_channel: str = "@symposium_lab",
        threads_max_chars: int = 280,
    ) -> SocialRepurposingBundle:
        analysis = self.load_analysis(project_id)
        manifest = self.workspaces.get_manifest(project_id)
        ws_dir = Path(manifest.project.workspace_dir)

        tl_files = sorted((ws_dir / "timelines").glob("*_youtube.json"))
        if not tl_files:
            session, anim_res = self._latest_symposium_and_anim(manifest)
            yt_tl = build_timeline_from_storyboard(
                project_id, manifest.project.name, analysis, session.storyboard, animation_result=anim_res
            )
        else:
            yt_tl = RenderTimeline.model_validate_json(tl_files[-1].read_text(encoding="utf-8"))

        with self.workspaces.track_run(project_id, "generate_social_and_telegram") as (manifest, run):
            tg = build_telegram_campaign(
                project_id=project_id,
                project_name=manifest.project.name,
                analysis=analysis,
                target_channel=telegram_channel,
                db=self.db,
            )
            thumb_files = sorted((ws_dir / "thumbnails").glob("*.png"))
            short_map = {
                "youtube_shorts": [str(p) for p in sorted((ws_dir / "shorts").glob("*youtube_shorts*.mp4"))],
                "tiktok": [str(p) for p in sorted((ws_dir / "shorts").glob("*tiktok*.mp4"))],
                "instagram_reels": [str(p) for p in sorted((ws_dir / "shorts").glob("*instagram_reels*.mp4"))],
            }
            bundle = build_social_repurposing_bundle(
                project_id=project_id,
                project_name=manifest.project.name,
                analysis=analysis,
                youtube_timeline=yt_tl,
                telegram_campaign=tg,
                thumbnail_path=str(thumb_files[-1]) if thumb_files else None,
                short_video_paths=short_map,
                threads_max_chars=threads_max_chars,
            )
            social_path = ws_dir / "social" / "social_repurposing_bundle.json"
            social_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
            tg_path = ws_dir / "telegram" / "telegram_campaign.json"
            tg_path.write_text(tg.model_dump_json(indent=2), encoding="utf-8")

            a1 = self.workspaces.register_asset(manifest, run.run_id, AssetRole.SOCIAL_PACKAGE, social_path)
            a2 = self.workspaces.register_asset(manifest, run.run_id, AssetRole.TELEGRAM_CAMPAIGN, tg_path)
            run.output_asset_ids.extend([a1.asset_id, a2.asset_id])
        return bundle

    def evaluate_growth_experiments(self, project_id: str) -> list[dict[str, Any]]:
        """Evaluate post-publication metrics vs historical platform medians and extract learned hypotheses."""
        exps = self.growth.evaluate_project_experiments(project_id)
        return [e.model_dump(mode="json") for e in exps]

    def search_experiments(
        self,
        query: str | None = None,
        platform: str | None = None,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search recorded content growth experiments & learned hypotheses via FTS5 or platform filters."""
        if query:
            hits = self.db.search_all(query, limit=limit * 2, project_id=project_id)
            exp_ids = [h["entity_id"] for h in hits if h["entity_type"] == "experiment"]
            pool = {e.experiment_id: e for e in self.db.list_experiments(project_id=project_id, platform=platform, limit=300)}
            matched = [pool[eid].model_dump(mode="json") for eid in exp_ids if eid in pool]
            if matched:
                return matched[:limit]
        return [
            e.model_dump(mode="json")
            for e in self.db.list_experiments(project_id=project_id, platform=platform, limit=limit)
        ]

    def write_production_report(self, project_id: str) -> dict[str, Any]:
        manifest = self.workspaces.get_manifest(project_id)
        analysis = self.load_analysis(project_id)
        ws_dir = Path(manifest.project.workspace_dir)
        session, _ = self._latest_symposium_and_anim(manifest)

        qa_files = sorted((ws_dir / "reports").glob("qa_*.json"))
        qa_reports = [QualityAssuranceReport.model_validate_json(f.read_text(encoding="utf-8")).model_dump(mode="json") for f in qa_files]

        rep_files = sorted((ws_dir / "research").glob("*.json"))
        research_reports = [ResearchReport.model_validate_json(f.read_text(encoding="utf-8")).model_dump(mode="json") for f in rep_files]

        report_payload = {
            "project": manifest.project.model_dump(mode="json"),
            "media_probe": analysis.probe.model_dump(mode="json"),
            "transcript_summary": {
                "engine": analysis.transcript.engine,
                "model": analysis.transcript.model,
                "detected_language": analysis.transcript.detected_language,
                "segment_count": len(analysis.transcript.segments),
                "uncertain_segment_ids": analysis.transcript.uncertain_segment_ids,
            },
            "research_sources": [
                {
                    "report_id": r["report_id"],
                    "source_item_ids": r["source_item_ids"],
                    "cross_niche_transfers_count": len(r["cross_niche_transfers"]),
                    "limitations": r["limitations"],
                }
                for r in research_reports
            ],
            "symposium_decisions": [d.model_dump(mode="json") for d in session.decisions],
            "moderator_challenges": [c.model_dump(mode="json") for c in session.challenges],
            "relevance_assessment": session.relevance_assessment.model_dump(mode="json") if session.relevance_assessment else None,
            "candidate_evaluations": [c.model_dump(mode="json") for c in session.candidate_evaluations],
            "growth_objective": session.growth_objective.model_dump(mode="json") if session.growth_objective else None,
            "creative_director_review": session.creative_director_review.model_dump(mode="json") if session.creative_director_review else None,
            "qa_reports": qa_reports,
            "assets": [a.model_dump(mode="json") for a in manifest.assets],
            "limitations": [
                "No private retention curves, CTR, or platform watch-time data were fabricated.",
                "Local acoustic diarization uses spectral clustering unless external pyannote/sherpa weights are configured.",
                "External publishing defaults to export_only / draft_prep and requires explicit approval + official OAuth credentials.",
            ],
        }

        json_path = ws_dir / "reports" / "production_report.json"
        json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        md_lines = [
            f"# Production Report — {manifest.project.name} (`{project_id}`)",
            "",
            f"- **Detected Language**: `{analysis.transcript.detected_language}` via `{analysis.transcript.engine}` (`{analysis.transcript.model}`)",
            f"- **Source Duration**: `{analysis.probe.duration_sec}s` (`{analysis.probe.width}x{analysis.probe.height}` @ `{analysis.probe.fps}` fps)",
            f"- **Symposium Mode**: `{session.mode.value}` (`{len(session.participating_agents)}` agents, backend `{session.backend_used}`)",
            "",
            "## Key Creative Decisions & Evidence",
        ]
        for d in session.decisions:
            md_lines.append(f"- **{d.topic}**: {d.chosen_option} _(Evidence: {', '.join(d.evidence_refs) or 'internal analysis'})_ — {d.rationale}")
        md_lines.append("\n## Quality Verification")
        for q in qa_reports:
            md_lines.append(f"- `{Path(q['artifact_path']).name}`: passed=`{q['passed']}`, {q['width']}x{q['height']}, dur=`{q['duration_sec']}s`, mean_audio=`{q['audio_mean_db']} dB`, captions=`{q['caption_count']}`, overlays=`{q['overlay_count']}`")
        md_lines.append("\n## Limitations & Compliance Notes")
        for lim in report_payload["limitations"]:
            md_lines.append(f"- {lim}")

        md_path = ws_dir / "reports" / "production_report.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        with self.workspaces.track_run(project_id, "write_production_report") as (m2, run):
            a_json = self.workspaces.register_asset(m2, run.run_id, AssetRole.PRODUCTION_REPORT, json_path)
            a_md = self.workspaces.register_asset(m2, run.run_id, AssetRole.PRODUCTION_REPORT, md_path)
            run.output_asset_ids.extend([a_json.asset_id, a_md.asset_id])

        return {"json_path": str(json_path), "markdown_path": str(md_path), "report": report_payload}

    def run_campaign_pipeline(
        self,
        name: str,
        video_path: Path,
        creator_niche: str = "science_engineering",
        editing_style: str = "educational_explainer",
        symposium_mode: ResearchMode = ResearchMode.STANDARD,
        animation_template: str | None = None,
        telegram_channel: str = "@symposium_lab",
        discover_live_research: bool = False,
        short_variants_per_platform: int = 1,
        download_click_sounds: bool | None = None,
    ) -> dict[str, Any]:
        """Execute complete end-to-end pipeline from raw footage to multi-platform campaign."""
        manifest = self.create_project(name=name, creator_niche=creator_niche, editing_style=editing_style)
        pid = manifest.project.project_id
        self.ingest_video(pid, video_path)
        analysis = self.analyze_project(pid)
        keywords = [k for t in analysis.topics for k in t.keywords][:6]
        discovery_summary = None
        if discover_live_research:
            discovery_summary = self.discover_live_research(target_count=100, query=" ".join(keywords[:3]) or None)
        research_rep = self.run_research_report(pid, "content_proposal", keywords)
        is_long_zoomer = editing_style == "zoomer_vlog" and analysis.probe.duration_sec > 120.0
        target_min = 90.0 if is_long_zoomer else 60.0
        target_max = min(145.0, max(95.0, analysis.probe.duration_sec * 0.78)) if is_long_zoomer else 90.0
        session = self.run_symposium_conference(
            pid,
            mode=symposium_mode,
            style_name=editing_style,
            target_min_sec=target_min,
            target_max_sec=target_max,
        )

        # Pick template decided by Symposium storyboard if not explicitly specified
        chosen_tmpl = animation_template or next(
            (b.animation_template for b in session.storyboard.beats if b.animation_template),
            "meme_reaction_card" if editing_style == "zoomer_vlog" else "em_induction_faraday",
        )
        downloaded_memes: list[dict[str, Any]] = []
        extra_anims: list[AnimationRenderResult] = []
        if chosen_tmpl == "meme_reaction_card" or editing_style == "zoomer_vlog":
            meme_anims, downloaded_memes = self._generate_zoomer_meme_cards(manifest, analysis, session, max_cards=10)
            anim = meme_anims[0]
            extra_anims = meme_anims[1:]
        else:
            anim = self.generate_animation(pid, template_id=chosen_tmpl, duration_sec=6.5)

        resolved_download_clicks = (
            bool(discover_live_research or editing_style in {"zoomer_vlog", "short_form_viral", "comedy"})
            if download_click_sounds is None
            else bool(download_click_sounds)
        )
        renders = self.build_and_render_deliverables(
            pid,
            short_variants_per_platform=short_variants_per_platform,
            animation_result=anim,
            extra_animations=extra_anims,
            download_click_sounds=resolved_download_clicks,
        )
        social = self.generate_social_and_telegram(pid, telegram_channel=telegram_channel)
        report = self.write_production_report(pid)

        return {
            "project_id": pid,
            "workspace_dir": manifest.project.workspace_dir,
            "detected_language": analysis.transcript.detected_language,
            "highlights_count": len(analysis.highlights),
            "discovery": discovery_summary,
            "trending_memes": downloaded_memes,
            "research_report_id": research_rep.report_id,
            "symposium_session_id": session.session_id,
            "animation": anim.model_dump(mode="json"),
            "renders": renders,
            "social": social.model_dump(mode="json"),
            "report_paths": {"json": report["json_path"], "markdown": report["markdown_path"]},
        }
