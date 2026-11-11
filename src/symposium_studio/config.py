"""Global configuration and safety/privacy policies for Symposium Studio."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


class SafetyPolicy(BaseModel):
    """Explicit guardrails for external AI, media privacy, licensing, and publishing."""

    local_processing_default: bool = True
    allow_external_media_upload: bool = Field(
        default=False,
        description="Must be explicitly opted in before sending raw source media or transcripts to third-party cloud APIs.",
    )
    allow_cloud_ai_reasoning: bool = Field(
        default=False,
        description="Opt-in gate allowing Symposium LLM agents to send scrubbed highlight metadata/summaries to cloud LLM APIs.",
    )
    allow_remotion_adapter: bool = Field(
        default=False,
        description="Opt-in flag acknowledging Remotion's source-available company license restrictions.",
    )
    allow_unapproved_publishing: bool = Field(
        default=False,
        description="Never publish externally without explicit user confirmation.",
    )
    strict_factuality_separation: bool = Field(
        default=True,
        description="Enforce separation between factual_scientific animations and artistic_illustration scenes.",
    )


class StudioConfig(BaseModel):
    """Runtime paths and engine defaults."""

    repo_root: Path = Field(default_factory=lambda: Path.cwd())
    workspaces_root: Path = Field(default_factory=lambda: Path.cwd() / "workspaces")
    db_path: Path = Field(default_factory=lambda: Path.cwd() / "workspaces" / "symposium.db")
    whisper_model: str = Field(
        default="mlx-community/whisper-small-mlx",
        description="Default local MLX Whisper model (or Whisper model name for fallback).",
    )
    ollama_url: str = Field(default="http://localhost:11434")
    ollama_model: str | None = Field(default=None)
    local_llm_url: str | None = Field(default=None)
    local_llm_model: str | None = Field(default=None)
    cloud_llm_model: str | None = Field(default=None)
    safety: SafetyPolicy = Field(default_factory=SafetyPolicy)

    @classmethod
    def from_env(cls, root: Path | None = None) -> "StudioConfig":
        base = (root or Path(os.environ.get("SYMPOSIUM_ROOT", Path.cwd()))).resolve()
        ws = Path(os.environ.get("SYMPOSIUM_WORKSPACES", base / "workspaces")).resolve()
        db = Path(os.environ.get("SYMPOSIUM_DB", ws / "symposium.db")).resolve()
        return cls(
            repo_root=base,
            workspaces_root=ws,
            db_path=db,
            whisper_model=os.environ.get("SYMPOSIUM_WHISPER_MODEL", "mlx-community/whisper-small-mlx"),
            ollama_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.environ.get("SYMPOSIUM_OLLAMA_MODEL"),
            local_llm_url=os.environ.get("SYMPOSIUM_LOCAL_LLM_URL"),
            local_llm_model=os.environ.get("SYMPOSIUM_LOCAL_LLM_MODEL"),
            cloud_llm_model=os.environ.get("SYMPOSIUM_CLOUD_LLM_MODEL"),
            safety=SafetyPolicy(
                allow_external_media_upload=os.environ.get("SYMPOSIUM_ALLOW_CLOUD_UPLOAD") == "1",
                allow_cloud_ai_reasoning=os.environ.get("SYMPOSIUM_ALLOW_CLOUD_AI") == "1",
                allow_remotion_adapter=os.environ.get("SYMPOSIUM_ALLOW_REMOTION") == "1",
            ),
        )
