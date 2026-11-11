"""Download, normalize, validate, and cache open-license internet click SFX."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from symposium_studio.engines.effects.sfx_synth import synthesize_sfx

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SymposiumStudio/0.2 (https://github.com/heroofkaist/video-editor; open-source-studio)"

# Curated CC0 / Public Domain click sounds hosted on Wikimedia Commons
DEFAULT_CLICK_SOUNDS: dict[str, dict[str, str]] = {
    "mouse_click": {
        "title": "Computer Mouse Click",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/ac/Computer_Mouse_Click.wav",
        "license": "CC0",
        "search_query": "computer mouse click",
        "description": "Crisp real computer mouse button click",
    },
    "mouse_double_click": {
        "title": "Computer Mouse Double Click",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Computer_mouse_double_click.ogg",
        "license": "CC0",
        "search_query": "computer mouse double click",
        "description": "Snappy double mouse click accent",
    },
    "mechanical_click": {
        "title": "Mechanical Switch Tack",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/56/Mechanical_tack.ogg",
        "license": "Public domain",
        "search_query": "mechanical tack",
        "description": "Short tactile mechanical switch click",
    },
    "switch_click": {
        "title": "UI Tactile Switch Click",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8e/Clickick_switch.ogg",
        "license": "Public domain",
        "search_query": "switch click",
        "description": "Clean toggle switch click for card dismissals and cuts",
    },
}


def default_sfx_cache_dir() -> Path:
    return Path.home() / ".cache" / "symposium-studio" / "sfx"


def _http_get_bytes(url: str, timeout_sec: float = 15.0, retries: int = 3) -> bytes:
    """Fetch raw bytes from URL with Wikimedia-compliant User-Agent and 429 retry handling."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in {429, 502, 503, 504} and attempt < retries - 1:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait_sec = float(retry_after) if (retry_after and retry_after.isdigit()) else (1.0 + 1.5 * attempt)
                time.sleep(min(6.0, wait_sec))
                continue
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(0.8 * (attempt + 1))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to download {url}")


def search_internet_clicks(
    query: str = "mouse click",
    limit: int = 8,
    timeout_sec: float = 15.0,
) -> list[dict[str, Any]]:
    """Search Wikimedia Commons API for short permissive-license audio click/SFX files."""
    clean_q = query.strip() or "mouse click"
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{clean_q} filetype:audio",
            "gsrnamespace": 6,
            "gsrlimit": max(1, min(25, limit)),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|mediatype|extmetadata",
            "format": "json",
        }
    )
    payload = json.loads(_http_get_bytes(f"{COMMONS_API_URL}?{params}", timeout_sec=timeout_sec).decode("utf-8"))
    pages = list(((payload.get("query") or {}).get("pages") or {}).values())
    results: list[dict[str, Any]] = []
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        url = str(ii.get("url") or "").split("?")[0]
        mime = str(ii.get("mime") or "")
        size_bytes = int(ii.get("size") or 0)
        if not url or ("audio" not in mime and str(ii.get("mediatype") or "") != "AUDIO"):
            continue
        if "midi" in mime:
            continue
        if size_bytes <= 0 or size_bytes > 750_000:
            continue
        lic = str(((ii.get("extmetadata") or {}).get("LicenseShortName") or {}).get("value") or "Unknown")
        lic_low = lic.lower()
        permissive_rank = 0 if ("cc0" in lic_low or "public domain" in lic_low) else (1 if "cc by" in lic_low else 2)
        results.append(
            {
                "title": str(p.get("title") or "").removeprefix("File:"),
                "url": url,
                "mime": mime,
                "size_bytes": size_bytes,
                "license": lic,
                "_rank": permissive_rank,
            }
        )
    results.sort(key=lambda r: (r["_rank"], r["size_bytes"]))
    for r in results:
        r.pop("_rank", None)
    return results


def _normalize_audio_to_wav(raw_path: Path, out_wav: Path, trim_silence: bool = True, max_duration_sec: float = 0.75) -> float:
    """Decode arbitrary audio file into 48kHz mono PCM WAV with trimmed leading silence and clean fade-out."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if trim_silence:
        filters.append("silenceremove=start_periods=1:start_silence=0.005:start_threshold=-42dB")
    filters.extend(
        [
            "aresample=48000",
            "loudnorm=I=-14:TP=-1.5:LRA=7",
        ]
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(raw_path),
        "-af",
        ",".join(filters),
        "-ac",
        "1",
        "-t",
        f"{max(0.12, min(2.0, max_duration_sec)):.3f}",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    with wave.open(str(out_wav), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or 48000
        dur = frames / float(rate)
    if frames < 240:
        raise ValueError(f"Decoded SFX {out_wav} is empty ({frames} frames)")
    return round(dur, 3)


def download_internet_sound(
    url: str,
    out_wav: Path,
    trim_silence: bool = True,
    max_duration_sec: float = 0.75,
    cache_dir: Path | None = None,
    timeout_sec: float = 15.0,
) -> dict[str, Any]:
    """Download an internet audio file (cached by URL hash) and convert it into 48kHz mono WAV."""
    out_wav = Path(out_wav)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    resolved_cache = cache_dir if cache_dir is not None else default_sfx_cache_dir()
    resolved_cache.mkdir(parents=True, exist_ok=True)
    cached_wav = resolved_cache / f"net_{url_hash}.wav"

    if cached_wav.is_file() and cached_wav.stat().st_size > 1000:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(cached_wav.read_bytes())
        with wave.open(str(out_wav), "rb") as wf:
            dur = round(wf.getnframes() / float(wf.getframerate() or 48000), 3)
        return {
            "path": str(out_wav.resolve()),
            "source_url": url,
            "origin": "cached_internet",
            "duration_sec": dur,
            "sha256": hashlib.sha256(out_wav.read_bytes()).hexdigest()[:16],
        }

    raw_bytes = _http_get_bytes(url, timeout_sec=timeout_sec)
    if len(raw_bytes) < 256:
        raise ValueError(f"Downloaded payload too small ({len(raw_bytes)} bytes) from {url}")
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".ogg"
    with tempfile.TemporaryDirectory(prefix="symposium_dl_sfx_") as tmpdir:
        raw_file = Path(tmpdir) / f"download{suffix}"
        raw_file.write_bytes(raw_bytes)
        dur = _normalize_audio_to_wav(
            raw_file,
            out_wav,
            trim_silence=trim_silence,
            max_duration_sec=max_duration_sec,
        )
    cached_wav.write_bytes(out_wav.read_bytes())
    return {
        "path": str(out_wav.resolve()),
        "source_url": url,
        "origin": "internet_download",
        "duration_sec": dur,
        "sha256": hashlib.sha256(out_wav.read_bytes()).hexdigest()[:16],
    }


def download_click_sounds(
    effects_dir: Path,
    presets: list[str] | None = None,
    custom_urls: dict[str, str] | None = None,
    search_query: str | None = None,
    allow_network: bool = True,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Download open-license click SFX from the internet into effects_dir with safe DSP fallback."""
    effects_dir = Path(effects_dir)
    effects_dir.mkdir(parents=True, exist_ok=True)

    targets: dict[str, dict[str, str]] = {}
    selected = presets or list(DEFAULT_CLICK_SOUNDS.keys())
    for preset in selected:
        if preset in DEFAULT_CLICK_SOUNDS:
            targets[preset] = dict(DEFAULT_CLICK_SOUNDS[preset])
    if custom_urls:
        for preset, url in custom_urls.items():
            targets[preset] = {
                "title": preset,
                "url": url,
                "license": "custom_url",
                "search_query": preset.replace("_", " "),
            }

    if search_query and allow_network:
        try:
            hits = search_internet_clicks(search_query, limit=3)
            for idx, h in enumerate(hits[:2]):
                key = f"search_click_{idx + 1}"
                targets[key] = {
                    "title": h["title"],
                    "url": h["url"],
                    "license": h["license"],
                    "search_query": search_query,
                }
        except Exception:
            pass

    click_paths: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    downloaded_count = 0
    cached_count = 0
    fallback_count = 0

    for idx, (preset, meta) in enumerate(targets.items()):
        out_wav = effects_dir / f"{preset}.wav"
        if allow_network:
            try:
                if idx > 0 and cached_count == 0:
                    time.sleep(0.35)
                info = download_internet_sound(meta["url"], out_wav, cache_dir=cache_dir)
                info["license"] = meta.get("license", "Unknown")
                info["title"] = meta.get("title", preset)
                if info["origin"] == "internet_download":
                    downloaded_count += 1
                else:
                    cached_count += 1
                click_paths[preset] = str(out_wav.resolve())
                sources[preset] = info
                continue
            except Exception as exc:
                err_msg = str(exc)
        else:
            err_msg = "network_disabled"

        # Graceful fallback if network unavailable or preset download failed
        if not out_wav.exists() or out_wav.stat().st_size < 1000:
            synth_key = preset if preset in {"mouse_click", "mechanical_click", "switch_click", "ui_pop"} else "mouse_click"
            synthesize_sfx(synth_key, out_wav, duration_sec=0.14)
        fallback_count += 1
        click_paths[preset] = str(out_wav.resolve())
        sources[preset] = {
            "path": str(out_wav.resolve()),
            "source_url": meta.get("url"),
            "title": meta.get("title", preset),
            "license": meta.get("license", "Unknown"),
            "origin": "synthesized_fallback",
            "fallback_reason": err_msg,
        }

    manifest_path = effects_dir / "sfx_sources.json"
    manifest_payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "downloaded_count": downloaded_count,
        "cached_count": cached_count,
        "fallback_count": fallback_count,
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return {
        "click_paths": click_paths,
        "downloaded_count": downloaded_count,
        "cached_count": cached_count,
        "fallback_count": fallback_count,
        "manifest_path": str(manifest_path),
        "sources": sources,
    }
