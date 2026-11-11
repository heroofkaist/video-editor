"""Multi-provider structured LLM router for AI Symposium with strict local-first privacy gating."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

from symposium_studio.config import StudioConfig


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    txt = raw_text.strip()
    if not txt:
        return None
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


class SymposiumLLMRouter:
    """Routes structured agent prompts to local Ollama/vLLM first, or opt-in cloud APIs when explicitly enabled."""

    def __init__(self, config: StudioConfig) -> None:
        self.config = config
        self.calls_attempted = 0
        self.calls_succeeded = 0
        self.fallback_reasons: list[str] = []

    def _detect_ollama_model(self) -> str | None:
        try:
            r = httpx.get(f"{self.config.ollama_url.rstrip('/')}/api/tags", timeout=1.5)
            if r.status_code != 200:
                return None
            models = [str(m.get("name", "")) for m in r.json().get("models", []) if m.get("name")]
            if not models:
                return None
            if self.config.ollama_model and self.config.ollama_model in models:
                return self.config.ollama_model
            return models[0]
        except Exception:
            return None

    def detect_backend(self) -> str:
        ollama_m = self._detect_ollama_model()
        if ollama_m:
            return f"ollama:{ollama_m}"
        if self.config.local_llm_url:
            return f"local_openai:{self.config.local_llm_model or 'local'}"
        if self.config.safety.allow_cloud_ai_reasoning:
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                return f"gemini:{self.config.cloud_llm_model or 'gemini-2.5-flash'}"
            if os.environ.get("OPENAI_API_KEY"):
                return f"openai:{self.config.cloud_llm_model or 'gpt-4.1-mini'}"
            if os.environ.get("ANTHROPIC_API_KEY"):
                return f"anthropic:{self.config.cloud_llm_model or 'claude-sonnet-4-20250514'}"
        return "deterministic_analytical"

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_sec: float = 20.0,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Invoke configured LLM backend and parse strict JSON object, returning fallback cleanly if unavailable."""
        backend = self.detect_backend()
        if backend == "deterministic_analytical":
            return None, {"backend": backend, "used_llm": False}

        self.calls_attempted += 1
        t0 = time.monotonic()
        try:
            if backend.startswith("ollama:"):
                model_name = backend.split(":", 1)[1]
                resp = httpx.post(
                    f"{self.config.ollama_url.rstrip('/')}/api/chat",
                    json={
                        "model": model_name,
                        "stream": False,
                        "format": "json",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=timeout_sec,
                )
                resp.raise_for_status()
                raw = str(resp.json().get("message", {}).get("content", ""))
                parsed = _extract_json_object(raw)
                if parsed is not None:
                    self.calls_succeeded += 1
                    return parsed, {"backend": backend, "used_llm": True, "latency_ms": round((time.monotonic() - t0) * 1000, 1)}

            elif backend.startswith("local_openai:") and self.config.local_llm_url:
                model_name = backend.split(":", 1)[1]
                resp = httpx.post(
                    f"{self.config.local_llm_url.rstrip('/')}/chat/completions",
                    json={
                        "model": model_name,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=timeout_sec,
                )
                resp.raise_for_status()
                raw = str(resp.json()["choices"][0]["message"]["content"])
                parsed = _extract_json_object(raw)
                if parsed is not None:
                    self.calls_succeeded += 1
                    return parsed, {"backend": backend, "used_llm": True, "latency_ms": round((time.monotonic() - t0) * 1000, 1)}

            elif backend.startswith("gemini:") and self.config.safety.allow_cloud_ai_reasoning:
                key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
                model_name = backend.split(":", 1)[1]
                resp = httpx.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                    params={"key": key},
                    json={
                        "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                        "generationConfig": {"responseMimeType": "application/json"},
                    },
                    timeout=timeout_sec,
                )
                resp.raise_for_status()
                raw = str(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
                parsed = _extract_json_object(raw)
                if parsed is not None:
                    self.calls_succeeded += 1
                    return parsed, {"backend": backend, "used_llm": True, "latency_ms": round((time.monotonic() - t0) * 1000, 1)}

            elif backend.startswith("openai:") and self.config.safety.allow_cloud_ai_reasoning:
                key = os.environ.get("OPENAI_API_KEY") or ""
                model_name = backend.split(":", 1)[1]
                resp = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model_name,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=timeout_sec,
                )
                resp.raise_for_status()
                raw = str(resp.json()["choices"][0]["message"]["content"])
                parsed = _extract_json_object(raw)
                if parsed is not None:
                    self.calls_succeeded += 1
                    return parsed, {"backend": backend, "used_llm": True, "latency_ms": round((time.monotonic() - t0) * 1000, 1)}

            elif backend.startswith("anthropic:") and self.config.safety.allow_cloud_ai_reasoning:
                key = os.environ.get("ANTHROPIC_API_KEY") or ""
                model_name = backend.split(":", 1)[1]
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                    json={
                        "model": model_name,
                        "max_tokens": 900,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                    timeout=timeout_sec,
                )
                resp.raise_for_status()
                raw = str(resp.json()["content"][0]["text"])
                parsed = _extract_json_object(raw)
                if parsed is not None:
                    self.calls_succeeded += 1
                    return parsed, {"backend": backend, "used_llm": True, "latency_ms": round((time.monotonic() - t0) * 1000, 1)}
        except Exception as exc:
            self.fallback_reasons.append(f"{backend}: {exc}")
        return None, {"backend": "deterministic_analytical", "used_llm": False}

    def maybe_refine_text(self, prompt: str, fallback_text: str) -> str:
        parsed, _ = self.complete_structured(
            "Respond strictly as JSON: {\"text\": \"...\"}",
            prompt,
        )
        if parsed and isinstance(parsed.get("text"), str) and parsed["text"].strip():
            return parsed["text"].strip()
        return fallback_text
