"""Isolated platform publishing subsystem with export-only, draft-prep, and explicitly-approved modes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from symposium_studio.engines.production.qa import evaluate_publish_readiness
from symposium_studio.models.social import PublishingMode, PublishRequest, QualityAssuranceReport


def _check_workspace_qa_readiness(workspace_dir: Path, artifact_path: str) -> dict[str, Any]:
    qa_files = sorted((workspace_dir / "reports").glob("qa_*.json"))
    qa_reports = [QualityAssuranceReport.model_validate_json(p.read_text(encoding="utf-8")) for p in qa_files]
    readiness = evaluate_publish_readiness(qa_reports)
    renders_exist = any((workspace_dir / "renders").glob("*.mp4")) or any((workspace_dir / "shorts").glob("*.mp4"))
    art_is_video = Path(artifact_path).suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"} if artifact_path else False
    qa_required = bool(qa_reports) or renders_exist or art_is_video
    readiness["qa_required"] = qa_required
    return readiness


def execute_publish(req: PublishRequest, workspace_dir: Path) -> dict[str, Any]:
    """Execute export_only, draft_prep, or approved_publish strictly honoring explicit user approval and QA readiness."""
    out_dir = workspace_dir / ("telegram" if req.platform == "telegram" else "social")
    out_dir.mkdir(parents=True, exist_ok=True)
    draft_path = out_dir / f"publish_payload_{req.platform}.json"
    readiness = _check_workspace_qa_readiness(workspace_dir, req.artifact_path)

    payload = {
        "platform": req.platform,
        "mode": req.mode.value,
        "artifact_path": req.artifact_path,
        "metadata": req.metadata,
        "approval_confirmed": req.approval_confirmed,
        "publish_readiness": readiness,
    }

    if req.mode == PublishingMode.EXPORT_ONLY:
        draft_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "status": "exported_only",
            "payload_path": str(draft_path),
            "external_posted": False,
            "publish_readiness": readiness,
        }

    # Platform preflight validation for draft_prep / approved_publish
    warnings: list[str] = []
    if readiness.get("qa_required") and not readiness["ready_for_publish"]:
        warnings.append("Critical QA checks have not passed; publishing is blocked until resolved.")
    if req.platform == "threads":
        text = str(req.metadata.get("text", ""))
        if len(text) > 500:
            raise ValueError(f"Threads text exceeds 500-char limit ({len(text)} chars).")
    elif req.platform == "telegram":
        art = Path(req.artifact_path) if req.artifact_path else None
        if art and art.exists() and art.stat().st_size > 50 * 1024 * 1024:
            warnings.append("File exceeds standard 50 MB Telegram Bot API limit; requires local Bot API server.")

    if req.mode == PublishingMode.DRAFT_PREP:
        draft_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "status": "draft_prepared",
            "payload_path": str(draft_path),
            "warnings": warnings,
            "external_posted": False,
            "publish_readiness": readiness,
        }

    # APPROVED_PUBLISH gate: requires explicit human approval AND clean QA readiness
    if not req.approval_confirmed or not req.approval_reason.strip():
        raise PermissionError(
            "External publishing refused: requires explicit `approval_confirmed=True` and non-empty `approval_reason`."
        )
    if readiness.get("qa_required") and not readiness["ready_for_publish"]:
        reasons = "; ".join(readiness.get("blocking_reasons") or ["QA failed"])
        raise PermissionError(
            f"External publishing refused by QA gate (status=requires_review): {reasons}"
        )

    if req.platform == "telegram":
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = req.metadata.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise RuntimeError(
                "Telegram approved_publish requires TELEGRAM_BOT_TOKEN and target chat_id/TELEGRAM_CHAT_ID."
            )
        text = str(req.metadata.get("text", ""))
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=15.0,
        )
        resp.raise_for_status()
        return {"status": "published_telegram", "response": resp.json(), "external_posted": True}

    raise RuntimeError(
        f"Official OAuth credential flow for '{req.platform}' is not configured in environment; "
        "payload validated in draft_prep mode instead."
    )
