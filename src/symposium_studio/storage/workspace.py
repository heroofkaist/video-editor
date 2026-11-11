"""Structured workspace manager, manifest tracker, and SHA256 provenance."""

from __future__ import annotations

import hashlib
import mimetypes
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from symposium_studio.config import StudioConfig
from symposium_studio.models.project import (
    AssetRecord,
    AssetRole,
    DecisionTraceEntry,
    ProjectManifest,
    ProjectRecord,
    RunRecord,
    ToolVersionRecord,
    WORKSPACE_SUBDIRS,
    utc_now_iso,
)
from symposium_studio.storage.db import StudioDatabase


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s or "project"


def detect_tool_versions() -> list[ToolVersionRecord]:
    tools: list[ToolVersionRecord] = []
    # FFmpeg
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        out = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True)
        ver = out.stdout.splitlines()[0] if out.stdout else "unknown"
        tools.append(ToolVersionRecord(name="ffmpeg", version=ver, available=True, license="LGPL/GPL CLI"))
    else:
        tools.append(ToolVersionRecord(name="ffmpeg", version="missing", available=False))

    # Python libs
    for pkg, lic in [
        ("opentimelineio", "Apache-2.0"),
        ("av", "BSD-3-Clause"),
        ("scenedetect", "BSD-3-Clause"),
        ("cv2", "Apache-2.0"),
        ("mlx_whisper", "MIT"),
        ("playwright", "Apache-2.0"),
        ("manim", "MIT"),
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            tools.append(ToolVersionRecord(name=pkg, version=str(ver), available=True, license=lic))
        except Exception:
            tools.append(ToolVersionRecord(name=pkg, version="not_installed", available=False, license=lic))

    # External optional binaries & hybrid runners
    for bin_name, flag, lic in [
        ("whisper-cli", "--version", "MIT"),
        ("blender", "--version", "GPL-3.0"),
        ("node", "--version", "MIT"),
    ]:
        b = shutil.which(bin_name)
        if b:
            out = subprocess.run([b, flag], capture_output=True, text=True)
            txt = (out.stdout or out.stderr or "").splitlines()
            ver = txt[0].strip() if txt else "installed"
            tools.append(ToolVersionRecord(name=bin_name, version=ver, available=True, license=lic))
        else:
            tools.append(ToolVersionRecord(name=bin_name, version="not_installed", available=False, license=lic))
    return tools


class WorkspaceManager:
    """Manages project directory hierarchy, manifest.json, runs, and asset indexing."""

    def __init__(self, config: StudioConfig, db: StudioDatabase) -> None:
        self.config = config
        self.db = db
        self.config.workspaces_root.mkdir(parents=True, exist_ok=True)

    def create_project(
        self,
        name: str,
        creator_niche: str = "science_engineering",
        editing_style: str = "educational_explainer",
        tags: list[str] | None = None,
    ) -> ProjectManifest:
        project_id = f"proj_{uuid.uuid4().hex[:10]}"
        slug = f"{slugify(name)}-{project_id[-6:]}"
        ws_dir = self.config.workspaces_root / slug
        ws_dir.mkdir(parents=True, exist_ok=True)
        for sub in WORKSPACE_SUBDIRS:
            (ws_dir / sub).mkdir(parents=True, exist_ok=True)

        record = ProjectRecord(
            project_id=project_id,
            name=name,
            slug=slug,
            creator_niche=creator_niche,
            editing_style=editing_style,
            workspace_dir=str(ws_dir),
            tags=tags or [],
        )
        manifest = ProjectManifest(project=record)
        self.save_manifest(manifest)
        self.db.upsert_project(record)
        return manifest

    def get_manifest(self, project_id: str) -> ProjectManifest:
        record = self.db.get_project(project_id)
        if not record:
            raise FileNotFoundError(f"Project not found: {project_id}")
        manifest_path = Path(record.workspace_dir) / "manifest.json"
        if not manifest_path.exists():
            return ProjectManifest(project=record)
        return ProjectManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: ProjectManifest) -> Path:
        manifest.project.updated_at = utc_now_iso()
        ws_dir = Path(manifest.project.workspace_dir)
        ws_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = ws_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self.db.upsert_project(manifest.project)
        return manifest_path

    def ingest_source_media(
        self,
        project_id: str,
        source_path: Path,
        copy_to_workspace: bool = True,
    ) -> tuple[ProjectManifest, AssetRecord]:
        if not source_path.exists():
            raise FileNotFoundError(f"Source media file not found: {source_path}")
        manifest = self.get_manifest(project_id)
        ws_dir = Path(manifest.project.workspace_dir)
        dest_dir = ws_dir / "source"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if copy_to_workspace:
            dest_path = dest_dir / source_path.name
            if source_path.resolve() != dest_path.resolve():
                shutil.copy2(source_path, dest_path)
        else:
            dest_path = source_path.resolve()

        digest = sha256_file(dest_path)
        manifest.project.source_video_path = str(dest_path)
        manifest.project.source_sha256 = digest
        manifest.project.status = "ingested"

        run_id = f"run_{uuid.uuid4().hex[:8]}"
        asset = self.register_asset(
            manifest,
            run_id=run_id,
            role=AssetRole.SOURCE_MEDIA,
            path=dest_path,
            source_attribution="creator_original_footage",
            license_info="Creator copyright preserved; original file unmodified",
        )
        run = RunRecord(
            run_id=run_id,
            project_id=manifest.project.project_id,
            command="ingest_source_media",
            status="completed",
            finished_at=utc_now_iso(),
            duration_sec=0.0,
            parameters={"source_path": str(source_path), "copied": copy_to_workspace},
            output_asset_ids=[asset.asset_id],
        )
        manifest.runs.append(run)
        self.db.upsert_run(run)
        self.save_manifest(manifest)
        return manifest, asset

    def register_asset(
        self,
        manifest: ProjectManifest,
        run_id: str,
        role: AssetRole,
        path: Path,
        source_attribution: str = "generated_by_symposium_studio",
        license_info: str = "Apache-2.0 / Creator-owned output",
        metadata: dict[str, Any] | None = None,
    ) -> AssetRecord:
        ws_dir = Path(manifest.project.workspace_dir).resolve()
        abs_path = path.resolve()
        try:
            rel_path = str(abs_path.relative_to(ws_dir))
        except ValueError:
            rel_path = str(abs_path)

        mime, _ = mimetypes.guess_type(str(abs_path))
        asset = AssetRecord(
            asset_id=f"asset_{uuid.uuid4().hex[:10]}",
            project_id=manifest.project.project_id,
            run_id=run_id,
            role=role,
            relative_path=rel_path,
            absolute_path=str(abs_path),
            sha256=sha256_file(abs_path) if abs_path.exists() and abs_path.is_file() else "",
            size_bytes=abs_path.stat().st_size if abs_path.exists() and abs_path.is_file() else 0,
            mime_type=mime or "application/octet-stream",
            source_attribution=source_attribution,
            license_info=license_info,
            metadata=metadata or {},
        )
        manifest.assets = [a for a in manifest.assets if a.relative_path != rel_path]
        manifest.assets.append(asset)
        self.db.upsert_asset(asset)
        return asset

    @contextmanager
    def track_run(
        self,
        project_id: str,
        command: str,
        parameters: dict[str, Any] | None = None,
    ) -> Iterator[tuple[ProjectManifest, RunRecord]]:
        manifest = self.get_manifest(project_id)
        t0 = time.monotonic()
        run = RunRecord(
            run_id=f"run_{uuid.uuid4().hex[:8]}",
            project_id=manifest.project.project_id,
            command=command,
            status="running",
            parameters=parameters or {},
            tool_versions=detect_tool_versions(),
        )
        try:
            yield manifest, run
            run.status = "completed"
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            manifest.runs.append(run)
            self.db.upsert_run(run)
            self.save_manifest(manifest)
            raise
        finally:
            run.finished_at = utc_now_iso()
            run.duration_sec = round(time.monotonic() - t0, 3)

        manifest.runs.append(run)
        manifest.decisions.extend(run.decisions)
        self.db.upsert_run(run)
        self.save_manifest(manifest)

    def add_decision(
        self,
        run: RunRecord,
        step: str,
        actor: str,
        decision: str,
        rationale: str,
        evidence_ids: list[str] | None = None,
    ) -> None:
        run.decisions.append(
            DecisionTraceEntry(
                step=step,
                actor=actor,
                decision=decision,
                rationale=rationale,
                evidence_ids=evidence_ids or [],
            )
        )
