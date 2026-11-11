"""Isolated platform publishing subsystem with export-only, draft-prep, and explicitly-approved modes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from symposium_studio.models.social import PublishingMode, PublishRequest


def execute_publish(req: PublishRequest, workspace_dir: Path) -> dict[str, Any]:
    """Execute export_only, draft_prep, or approved_publish strictly honoring explicit user approval."""
    out_dir = workspace_dir / ("telegram" if req.platform == "telegram" else "social")
    out_dir.mkdir(parents=True, exist_ok=True)
    draft_path = out_dir / f"publish_payload_{req.platform}.json"

    payload = {
        "platform": req.platform,
        "mode": req.mode.value,
        "artifact_path": req.artifact_path,
        "metadata": req.metadata,
        "approval_confirmed": req.approval_confirmed,
    }

    if req.mode == PublishingMode.EXPORT_ONLY:
        draft_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"status": "exported_only", "payload_path": str(draft_path), "external_posted": False}

    # Platform preflight validation for draft_prep / approved_publish
    warnings: list[str] = []
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
        }

    # APPROVED_PUBLISH gate
    if not req.approval_confirmed or not req.approval_reason.strip():
        raise PermissionError(
            "External publishing refused: requires explicit `approval_confirmed=True` and non-empty `approval_reason`."
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
