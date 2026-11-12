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

import numpy as np

from symposium_studio.engines.effects.sfx_synth import synthesize_sfx

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SymposiumStudio/0.2 (https://github.com/heroofkaist/video-editor; open-source-studio)"
_SFX_NORM_VERSION = "norm_v2_peak08"

# Curated open-license (CC0 / MIT / Public Domain) punchy ADHD click/pop/key SFX
# hosted on Kenney Audio (CC0), Glitch MMO Tiny Speck / Internet Archive (CC0),
# and Keesound mechanical switches (MIT).
DEFAULT_CLICK_SOUNDS: dict[str, dict[str, str]] = {
    "mouse_click": {
        "title": "Kenney Crisp Mouse Click",
        "url": "https://raw.githubusercontent.com/Calinou/kenney-ui-audio/master/addons/kenney_ui_audio/mouseclick1.wav",
        "license": "CC0-1.0",
        "search_query": "mouse click",
        "description": "Ultra-crisp dry tactile mouse click",
    },
    "mouse_double_click": {
        "title": "Kenney Snappy Double Switch",
        "url": "https://raw.githubusercontent.com/Calinou/kenney-ui-audio/master/addons/kenney_ui_audio/switch2.wav",
        "license": "CC0-1.0",
        "search_query": "mouse double click",
        "description": "Fast double-impulse switch tap for mid-card beats",
    },
    "mechanical_click": {
        "title": "Keesound Holy Panda Tactile Thock",
        "url": "https://raw.githubusercontent.com/nirajrajgor/keesound/main/Samples/holy-panda/down/generic_1.wav",
        "license": "MIT",
        "search_query": "mechanical switch",
        "description": "Punchy deep Holy Panda mechanical keycap thock",
    },
    "switch_click": {
        "title": "Kenney Tactile UI Switch",
        "url": "https://raw.githubusercontent.com/Calinou/kenney-ui-audio/master/addons/kenney_ui_audio/switch10.wav",
        "license": "CC0-1.0",
        "search_query": "switch click",
        "description": "Snappy toggle switch click for card dismissals and cuts",
    },
    "camera_shutter": {
        "title": "Glitch MMO Studio Camera Shutter",
        "url": "https://archive.org/download/glitch-mmo-sfx-tiny-speck/camera%20shutter.mp3",
        "license": "CC0-1.0",
        "search_query": "camera shutter",
        "description": "Crisp DSLR shutter snap for zoom punch-ins and reveals",
    },
    "typewriter_key": {
        "title": "Keesound Cherry MX Blue Key Click",
        "url": "https://raw.githubusercontent.com/nirajrajgor/keesound/main/Samples/cherry-mx-blue/down/generic_1.wav",
        "license": "MIT",
        "search_query": "cherry mx click",
        "description": "Single crisp Cherry MX Blue keystroke click for rapid cuts",
    },
    "pop_click": {
        "title": "Kenney Clean UI Bubble Drop",
        "url": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/drop_001.wav",
        "license": "CC0-1.0",
        "search_query": "bubble pop",
        "description": "Snappy CapCut-style UI pop accent for meme cards and callouts",
    },
    "bubble_pop": {
        "title": "Kenney Rising Bubble Pop",
        "url": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/drop_003.wav",
        "license": "CC0-1.0",
        "search_query": "bubble pop",
        "description": "Upward bouncy bubble pop for ADHD captions and reactions",
    },
    "scroll_tick": {
        "title": "Glitch MMO Scroll Wheel Tick",
        "url": "https://archive.org/download/glitch-mmo-sfx-tiny-speck/scroll%20clicking%202.mp3",
        "license": "CC0-1.0",
        "search_query": "scroll click",
        "description": "Hyper-snappy mouse scroll ratchet tick for kinetic phrasing",
    },
    "ui_tick": {
        "title": "Kenney Stopwatch UI Tick",
        "url": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/tick_001.wav",
        "license": "CC0-1.0",
        "search_query": "ui tick",
        "description": "Tight digital UI tick for 2-word subtitle bursts",
    },
    "glitch_tick": {
        "title": "Kenney Glitch Micro-Hit",
        "url": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/glitch_001.wav",
        "license": "CC0-1.0",
        "search_query": "glitch tick",
        "description": "Sharp glitch transient for aggressive punch-zoom cuts",
    },
    "creamy_key": {
        "title": "Keesound NK Cream Keycap Tap",
        "url": "https://raw.githubusercontent.com/nirajrajgor/keesound/main/Samples/nk-cream/down/generic_1.wav",
        "license": "MIT",
        "search_query": "mechanical keyboard tap",
        "description": "Creamy linear mechanical switch tap for rapid pacing",
    },
    "pluck_pop": {
        "title": "Kenney Bouncy UI Pluck",
        "url": "https://raw.githubusercontent.com/Calinou/kenney-interface-sounds/master/addons/kenney_interface_sounds/pluck_001.wav",
        "license": "CC0-1.0",
        "search_query": "pluck pop",
        "description": "Tactile harmonic UI pluck for hook words and meme pops",
    },
}

# Real open-license accent/meme/whoosh SFX that upgrade DSP fallbacks when network is enabled
DEFAULT_ACCENT_SOUNDS: dict[str, dict[str, str]] = {
    "vine_boom": {
        "title": "Viral Meme Vine Boom",
        "url": "https://raw.githubusercontent.com/Vamsiindugu/vine-boom-extension/main/sounds/vine-boom.mp3",
        "license": "MIT",
        "search_query": "vine boom",
        "description": "Punchy viral bass boom impact for shock/meme beats",
        "max_duration_sec": "1.30",
    },
    "ui_pop": {
        "title": "Glitch MMO Cartoon Plop",
        "url": "https://archive.org/download/glitch-mmo-sfx-tiny-speck/piggy%20plop.mp3",
        "license": "CC0-1.0",
        "search_query": "cartoon plop",
        "description": "Squishy bubbly pop-in sound for overlays and reactions",
        "max_duration_sec": "0.45",
    },
    "whoosh_transition": {
        "title": "Glitch MMO Whip Shoop",
        "url": "https://archive.org/download/glitch-mmo-sfx-tiny-speck/friend-exits%20%28shoop%29.mp3",
        "license": "CC0-1.0",
        "search_query": "whoosh swipe",
        "description": "Snappy cartoon swipe/whoosh transition",
        "max_duration_sec": "0.55",
    },
    "kaching_ding": {
        "title": "Glitch MMO Ka-Ching Reward Bell",
        "url": "https://archive.org/download/glitch-mmo-sfx-tiny-speck/buy%20kaching.mp3",
        "license": "CC0-1.0",
        "search_query": "kaching ding",
        "description": "Bright jackpot cash-register chime for unboxing/reveal wins",
        "max_duration_sec": "0.95",
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


def _normalize_audio_to_wav(raw_path: Path, out_wav: Path, trim_silence: bool = True, max_duration_sec: float = 0.55) -> float:
    """Decode audio file to 48kHz mono PCM WAV, strip dead air, peak-normalize to -0.8 dBFS."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    dur_limit = f"{max(0.12, min(2.0, max_duration_sec)):.3f}"

    def _run_decode(strip_silence: bool) -> np.ndarray:
        filters: list[str] = []
        if strip_silence:
            filters.append(
                "silenceremove=start_periods=1:start_silence=0.002:start_threshold=-45dB,"
                "areverse,silenceremove=start_periods=1:start_silence=0.010:start_threshold=-48dB,areverse"
            )
        filters.append("aresample=48000")
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
            dur_limit,
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        with wave.open(str(out_wav), "rb") as wf:
            return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    pcm = _run_decode(trim_silence)
    if trim_silence and len(pcm) < 240:
        pcm = _run_decode(False)
    if len(pcm) < 240:
        raise ValueError(f"Decoded SFX {out_wav} is empty ({len(pcm)} frames)")

    peak = float(np.max(np.abs(pcm)))
    if peak > 1e-4:
        # Peak-normalize to -0.8 dBFS so short 20-120ms transients punch through speech cleanly
        pcm = pcm * (0.912 / peak)

    fade_n = min(192, max(16, len(pcm) // 6))
    if fade_n > 0 and len(pcm) > fade_n:
        tail = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        pcm[-fade_n:] *= tail

    out_pcm = (np.clip(pcm, -0.96, 0.96) * 32767.0).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(out_pcm.tobytes())
    return round(len(out_pcm) / 48000.0, 3)


def download_internet_sound(
    url: str,
    out_wav: Path,
    trim_silence: bool = True,
    max_duration_sec: float = 0.55,
    cache_dir: Path | None = None,
    timeout_sec: float = 15.0,
) -> dict[str, Any]:
    """Download an internet audio file (cached by URL + norm hash) and convert it into 48kHz mono WAV."""
    out_wav = Path(out_wav)
    cache_key = f"{url}|{_SFX_NORM_VERSION}|{max_duration_sec:.2f}|{int(trim_silence)}"
    url_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
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


def _download_preset_map(
    effects_dir: Path,
    targets: dict[str, dict[str, str]],
    *,
    allow_network: bool = True,
    cache_dir: Path | None = None,
    fallback_duration_sec: float = 0.14,
    merge_manifest: bool = False,
) -> dict[str, Any]:
    """Download a preset map into effects_dir with safe DSP fallback and provenance manifest."""
    effects_dir = Path(effects_dir)
    effects_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    downloaded_count = 0
    cached_count = 0
    fallback_count = 0

    for idx, (preset, meta) in enumerate(targets.items()):
        out_wav = effects_dir / f"{preset}.wav"
        max_dur = float(meta.get("max_duration_sec", 0.55))
        if allow_network:
            try:
                if idx > 0 and cached_count == 0 and "wikimedia.org" in str(meta.get("url", "")):
                    time.sleep(0.25)
                info = download_internet_sound(
                    meta["url"],
                    out_wav,
                    max_duration_sec=max_dur,
                    cache_dir=cache_dir,
                )
                info["license"] = meta.get("license", "Unknown")
                info["title"] = meta.get("title", preset)
                if info["origin"] == "internet_download":
                    downloaded_count += 1
                else:
                    cached_count += 1
                paths[preset] = str(out_wav.resolve())
                sources[preset] = info
                continue
            except Exception as exc:
                err_msg = str(exc)
        else:
            err_msg = "network_disabled"

        if not out_wav.exists() or out_wav.stat().st_size < 1000:
            synth_key = (
                preset
                if preset in set(DEFAULT_CLICK_SOUNDS.keys()) | {"ui_pop", "vine_boom", "whoosh_transition", "kaching_ding"}
                else "mouse_click"
            )
            synthesize_sfx(synth_key, out_wav, duration_sec=fallback_duration_sec)
        fallback_count += 1
        paths[preset] = str(out_wav.resolve())
        sources[preset] = {
            "path": str(out_wav.resolve()),
            "source_url": meta.get("url"),
            "title": meta.get("title", preset),
            "license": meta.get("license", "Unknown"),
            "origin": "synthesized_fallback",
            "fallback_reason": err_msg,
        }

    manifest_path = effects_dir / "sfx_sources.json"
    merged_sources = dict(sources)
    if merge_manifest and manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            prev_sources = prev.get("sources") or {}
            merged_sources = {**prev_sources, **sources}
        except Exception:
            merged_sources = dict(sources)

    manifest_payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "downloaded_count": sum(1 for v in merged_sources.values() if v.get("origin") == "internet_download")
        if merge_manifest
        else downloaded_count,
        "cached_count": sum(1 for v in merged_sources.values() if v.get("origin") == "cached_internet")
        if merge_manifest
        else cached_count,
        "fallback_count": sum(1 for v in merged_sources.values() if v.get("origin") == "synthesized_fallback")
        if merge_manifest
        else fallback_count,
        "sources": merged_sources,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return {
        "click_paths": paths,
        "downloaded_count": downloaded_count,
        "cached_count": cached_count,
        "fallback_count": fallback_count,
        "manifest_path": str(manifest_path),
        "sources": sources,
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

    return _download_preset_map(
        effects_dir,
        targets,
        allow_network=allow_network,
        cache_dir=cache_dir,
        fallback_duration_sec=0.14,
        merge_manifest=False,
    )


def download_accent_sounds(
    effects_dir: Path,
    presets: list[str] | None = None,
    allow_network: bool = True,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Download open-license viral accent SFX (vine boom, cartoon plop, whoosh, ka-ching)."""
    selected = presets or list(DEFAULT_ACCENT_SOUNDS.keys())
    targets = {p: dict(DEFAULT_ACCENT_SOUNDS[p]) for p in selected if p in DEFAULT_ACCENT_SOUNDS}
    res = _download_preset_map(
        effects_dir,
        targets,
        allow_network=allow_network,
        cache_dir=cache_dir,
        fallback_duration_sec=0.55,
        merge_manifest=True,
    )
    res["accent_paths"] = res["click_paths"]
    return res
