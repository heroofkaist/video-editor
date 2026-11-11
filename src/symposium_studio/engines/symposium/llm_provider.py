"""Optional LLM provider integration with strict local-first privacy gating."""

from __future__ import annotations

import httpx

from symposium_studio.config import StudioConfig


class SymposiumLLMRouter:
    """Queries local Ollama if configured/running; never sends raw media to cloud without opt-in."""

    def __init__(self, config: StudioConfig) -> None:
        self.config = config

    def detect_backend(self) -> str:
        if self.config.ollama_model:
            try:
                r = httpx.get(f"{self.config.ollama_url}/api/tags", timeout=1.5)
                if r.status_code == 200:
                    models = [m.get("name") for m in r.json().get("models", [])]
                    if self.config.ollama_model in models:
                        return f"ollama:{self.config.ollama_model}"
            except Exception:
                pass
        return "deterministic_analytical"

    def maybe_refine_text(self, prompt: str, fallback_text: str) -> str:
        backend = self.detect_backend()
        if backend.startswith("ollama:") and self.config.ollama_model:
            try:
                resp = httpx.post(
                    f"{self.config.ollama_url}/api/generate",
                    json={"model": self.config.ollama_model, "prompt": prompt, "stream": False},
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    txt = str(resp.json().get("response", "")).strip()
                    if txt:
                        return txt
            except Exception:
                pass
        return fallback_text
