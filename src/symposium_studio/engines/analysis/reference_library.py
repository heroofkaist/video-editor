"""Multi-reference library manager: multi-file ingest, workspace picks, URL handling, and SHA256 cache."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import httpx

from symposium_studio.engines.analysis.probe import probe_media
from symposium_studio.engines.research.connectors import infer_platform_from_url
from symposium_studio.models.reference import (
    AudioStyleObservation,
    PacingRhythmObservation,
    ReferenceAnalysisStatus,
    ReferenceItem,
    ReferenceSourceType,
    StyleDNA,
    SubtitleStyleObservation,
    VisualStyleObservation,
    _utc_now,
)
from symposium_studio.storage.db import StudioDatabase
from symposium_studio.storage.workspace import sha256_file

SUPPORTED_REFERENCE_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}


class ReferenceLibraryManager:
    """Handles ingestion, deduplication, persistent cache, and workspace binding of reference videos."""

    def __init__(self, workspaces_root: Path, db: StudioDatabase) -> None:
        self.workspaces_root = Path(workspaces_root)
        self.db = db
        self.global_cache_root = self.workspaces_root / "_references"
        self.global_cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def project_refs_dir(workspace_dir: Path) -> Path:
        d = workspace_dir / "references"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def load_project_manifest(workspace_dir: Path) -> list[ReferenceItem]:
        mp = workspace_dir / "references" / "manifest.json"
        if not mp.is_file():
            return []
        raw = json.loads(mp.read_text(encoding="utf-8"))
        return [ReferenceItem.model_validate(item) for item in (raw.get("references") or [])]

    @staticmethod
    def save_project_manifest(workspace_dir: Path, items: list[ReferenceItem]) -> Path:
        refs_dir = ReferenceLibraryManager.project_refs_dir(workspace_dir)
        mp = refs_dir / "manifest.json"
        mp.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "updated_at": _utc_now(),
                    "count": len(items),
                    "references": [i.model_dump(mode="json") for i in items],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return mp

    def _upsert_project_manifest_item(self, workspace_dir: Path, item: ReferenceItem) -> None:
        items = self.load_project_manifest(workspace_dir)
        updated = [x for x in items if x.reference_id != item.reference_id]
        updated.append(item)
        self.save_project_manifest(workspace_dir, updated)

    def _cache_dir_for_sha(self, sha256: str) -> Path:
        d = self.global_cache_root / "cache" / sha256[:20]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _copy_cached_artifacts_to_project(
        self,
        cached: ReferenceItem,
        ref_dir: Path,
        new_ref_id: str,
        project_id: str | None,
    ) -> ReferenceItem:
        ref_dir.mkdir(parents=True, exist_ok=True)
        cloned = cached.model_copy(deep=True)
        cloned.reference_id = new_ref_id
        cloned.project_id = project_id
        cloned.updated_at = _utc_now()

        for attr, target_name in (
            ("edit_events_path", "edit_events.json"),
            ("contact_sheet_path", "contact_sheet.jpg"),
            ("evidence_path", "evidence.json"),
        ):
            src_str = getattr(cached, attr)
            if src_str and Path(src_str).is_file():
                dst = ref_dir / target_name
                if Path(src_str).resolve() != dst.resolve():
                    shutil.copy2(src_str, dst)
                setattr(cloned, attr, str(dst))
        if cloned.style_dna is not None:
            cloned.style_dna.reference_id = new_ref_id
            ev_arts: dict[str, str] = {}
            for k, v in (cloned.style_dna.evidence_artifacts or {}).items():
                if v and Path(v).is_file():
                    dst = ref_dir / Path(v).name
                    if Path(v).resolve() != dst.resolve():
                        shutil.copy2(v, dst)
                    ev_arts[k] = str(dst)
            cloned.style_dna.evidence_artifacts = ev_arts
        return cloned

    def ingest_local_file(
        self,
        source_path: Path,
        project_id: str | None = None,
        workspace_dir: Path | None = None,
        title: str | None = None,
        source_type: ReferenceSourceType = ReferenceSourceType.LOCAL_UPLOAD,
        platform: str = "local",
        creator: str = "",
        copy_video: bool = True,
    ) -> tuple[ReferenceItem, bool]:
        """Add a local MP4/MOV/WebM reference video with SHA256 deduplication and cache reuse.

        Returns `(reference_item, reused_from_cache)`.
        """
        src = Path(source_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Reference video not found: {source_path}")
        if src.suffix.lower() not in SUPPORTED_REFERENCE_EXTS:
            raise ValueError(f"Unsupported reference format '{src.suffix}'; supported: {sorted(SUPPORTED_REFERENCE_EXTS)}")

        digest = sha256_file(src)
        file_size = src.stat().st_size

        # Deduplicate cleanly inside same project if exact SHA256 already present
        if project_id:
            existing_proj = [r for r in self.db.list_references(project_id=project_id, limit=500) if r.sha256 == digest]
            if existing_proj:
                return existing_proj[0], True

        ref_id = f"ref_{uuid.uuid4().hex[:10]}"
        if workspace_dir is not None:
            ref_dir = self.project_refs_dir(workspace_dir) / ref_id
        else:
            ref_dir = self._cache_dir_for_sha(digest) / ref_id
        ref_dir.mkdir(parents=True, exist_ok=True)

        if copy_video:
            stored_video = ref_dir / f"video{src.suffix.lower()}"
            if src != stored_video.resolve():
                shutil.copy2(src, stored_video)
        else:
            stored_video = src

        probe = probe_media(stored_video)
        cached = self.db.get_reference_by_sha256(digest)
        reused_cache = cached is not None and cached.style_dna is not None and cached.analysis_status == ReferenceAnalysisStatus.FULL_VIDEO_AUDIO
        if reused_cache and cached is not None:
            item = self._copy_cached_artifacts_to_project(cached, ref_dir, ref_id, project_id)
            item.source_type = source_type
            item.source_path = str(stored_video)
            item.title = title or cached.title or src.stem
            item.file_size_bytes = file_size
            item.duration_sec = probe.duration_sec
            item.metadata = {**item.metadata, "reused_from_reference_id": cached.reference_id, "cache_hit": True}
        else:
            item = ReferenceItem(
                reference_id=ref_id,
                project_id=project_id,
                title=title or src.stem,
                source_type=source_type,
                source_path=str(stored_video),
                sha256=digest,
                file_size_bytes=file_size,
                duration_sec=probe.duration_sec,
                platform=platform,
                creator=creator,
                analysis_status=ReferenceAnalysisStatus.PENDING,
                metadata={
                    "width": probe.width,
                    "height": probe.height,
                    "fps": probe.fps,
                    "has_audio": probe.has_audio,
                    "cache_hit": False,
                },
            )

        self.db.upsert_reference(item)
        if workspace_dir is not None:
            self._upsert_project_manifest_item(workspace_dir, item)
        return item, reused_cache

    def ingest_workspace_file(
        self,
        project_id: str,
        workspace_dir: Path,
        relative_path: str,
        title: str | None = None,
    ) -> tuple[ReferenceItem, bool]:
        ws_resolved = workspace_dir.resolve()
        candidate = (ws_resolved / relative_path).resolve()
        try:
            candidate.relative_to(ws_resolved)
        except ValueError as exc:
            raise PermissionError(f"Path traversal denied for workspace reference: {relative_path}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Workspace file not found: {relative_path}")
        return self.ingest_local_file(
            candidate,
            project_id=project_id,
            workspace_dir=workspace_dir,
            title=title or candidate.stem,
            source_type=ReferenceSourceType.WORKSPACE_FILE,
            platform="workspace",
            copy_video=False,
        )

    def ingest_url(
        self,
        url: str,
        project_id: str | None = None,
        workspace_dir: Path | None = None,
        title: str | None = None,
        creator: str = "",
        allow_network: bool = False,
        metadata_notes: dict[str, Any] | None = None,
    ) -> tuple[ReferenceItem, bool]:
        """Import a reference URL honestly: fetches only direct/permitted media streams when allowed,
        or marks explicitly as `metadata_only` without fabricating visual edits."""
        plat_raw = infer_platform_from_url(url)
        platform = plat_raw.value if hasattr(plat_raw, "value") else str(plat_raw)
        ref_id = f"ref_{uuid.uuid4().hex[:10]}"
        ref_dir = (self.project_refs_dir(workspace_dir) if workspace_dir else self.global_cache_root) / ref_id
        ref_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path: Path | None = None
        fetch_note = ""
        url_low = url.lower().split("?", 1)[0]
        is_direct_media = any(url_low.endswith(ext) for ext in SUPPORTED_REFERENCE_EXTS)

        if allow_network and is_direct_media:
            try:
                ext = Path(url_low).suffix or ".mp4"
                target = ref_dir / f"downloaded_ref{ext}"
                with httpx.stream("GET", url, follow_redirects=True, timeout=20.0) as resp:
                    resp.raise_for_status()
                    ctype = (resp.headers.get("content-type") or "").lower()
                    if "video" in ctype or "octet-stream" in ctype or is_direct_media:
                        with target.open("wb") as fh:
                            for chunk in resp.iter_bytes(chunk_size=65536):
                                fh.write(chunk)
                        if target.stat().st_size > 1024:
                            downloaded_path = target
            except Exception as exc:
                fetch_note = f"Direct media download unavailable ({type(exc).__name__}); stored metadata-only."
        elif allow_network and shutil.which("yt-dlp"):
            # Only attempt yt-dlp if explicitly available and permitted by user environment
            try:
                out_tpl = str(ref_dir / "ref_media.%(ext)s")
                subprocess.run(
                    [
                        "yt-dlp",
                        "--no-playlist",
                        "-f",
                        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                        "-o",
                        out_tpl,
                        url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=45,
                    check=True,
                )
                cands = [p for p in ref_dir.iterdir() if p.suffix.lower() in SUPPORTED_REFERENCE_EXTS]
                if cands:
                    downloaded_path = cands[0]
            except Exception as exc:
                fetch_note = f"URL media stream inaccessible ({type(exc).__name__}); flagged honestly as metadata-only."

        if downloaded_path is not None and downloaded_path.is_file():
            try:
                item, cache_hit = self.ingest_local_file(
                    downloaded_path,
                    project_id=project_id,
                    workspace_dir=workspace_dir,
                    title=title or downloaded_path.stem,
                    source_type=ReferenceSourceType.URL_VIDEO,
                    platform=platform,
                    creator=creator,
                    copy_video=False,
                )
                item.source_url = url
                self.db.upsert_reference(item)
                if workspace_dir is not None:
                    self._upsert_project_manifest_item(workspace_dir, item)
                return item, cache_hit
            except Exception as exc:
                fetch_note = f"Downloaded file failed probe ({exc}); falling back to metadata-only."

        limitation_msg = (
            fetch_note
            or (
                "Video frames and cut timestamps were not downloaded or accessible for this URL; "
                "entry is stored strictly as metadata-only with zero fabricated visual editing metrics."
            )
        )
        clean_title = title or f"{platform.replace('_', ' ').title()} Reference ({ref_id[-4:]})"
        evidence_path = ref_dir / "evidence.json"
        evidence_path.write_text(
            json.dumps(
                {
                    "reference_id": ref_id,
                    "analysis_status": ReferenceAnalysisStatus.METADATA_ONLY.value,
                    "source_url": url,
                    "platform": platform,
                    "limitation": limitation_msg,
                    "metadata": metadata_notes or {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        item = ReferenceItem(
            reference_id=ref_id,
            project_id=project_id,
            title=clean_title,
            source_type=ReferenceSourceType.URL_METADATA_ONLY,
            source_url=url,
            platform=platform,
            creator=creator,
            analysis_status=ReferenceAnalysisStatus.METADATA_ONLY,
            style_dna=None,
            evidence_path=str(evidence_path),
            metadata={
                "url_import": True,
                "video_stream_accessible": False,
                "honest_limitations": [limitation_msg],
                "editorial_notes": [
                    "Metadata-only import: upload the reference MP4/MOV/WebM locally to unlock full frame-level Style DNA."
                ],
                **(metadata_notes or {}),
            },
        )
        self.db.upsert_reference(item)
        if workspace_dir is not None:
            self._upsert_project_manifest_item(workspace_dir, item)
        return item, False

    def import_existing_reference(
        self,
        source_reference_id: str,
        project_id: str,
        workspace_dir: Path,
    ) -> ReferenceItem:
        src_item = self.db.get_reference(source_reference_id)
        if src_item is None:
            raise FileNotFoundError(f"Reference not found in library: {source_reference_id}")
        if src_item.source_path and Path(src_item.source_path).is_file():
            imported, _ = self.ingest_local_file(
                Path(src_item.source_path),
                project_id=project_id,
                workspace_dir=workspace_dir,
                title=src_item.title,
                source_type=ReferenceSourceType.LIBRARY_CACHE,
                platform=src_item.platform,
                creator=src_item.creator,
                copy_video=False,
            )
            return imported
        new_id = f"ref_{uuid.uuid4().hex[:10]}"
        ref_dir = self.project_refs_dir(workspace_dir) / new_id
        cloned = self._copy_cached_artifacts_to_project(src_item, ref_dir, new_id, project_id)
        cloned.source_type = ReferenceSourceType.LIBRARY_CACHE
        self.db.upsert_reference(cloned)
        self._upsert_project_manifest_item(workspace_dir, cloned)
        return cloned
