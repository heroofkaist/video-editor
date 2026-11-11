"""Headless Chromium HTML5 Canvas / SVG animation renderer via Playwright.

Produces deterministic frame-accurate RGBA PNG sequences (with true alpha channel)
and encodes alpha-preserving ProRes4444 MOV or H.264 MP4 deliverables via FFmpeg.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from symposium_studio.engines.animation.base import AnimationAdapter
from symposium_studio.models.animation import (
    AdapterLicenseTier,
    AnimationAdapterCapability,
    AnimationDomain,
    AnimationRenderResult,
    SceneSpec,
)


def _build_canvas_html(spec: SceneSpec) -> str:
    spec_json = json.dumps(spec.model_dump(mode="json"))
    bg_css = "transparent" if spec.transparent_background else spec.theme_palette.get("bg", "#0B1020")
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html, body {{
    margin: 0; padding: 0; overflow: hidden;
    background: {bg_css};
    width: {spec.width}px; height: {spec.height}px;
    font-family: 'Helvetica Neue', 'SF Pro Display', system-ui, sans-serif;
  }}
  canvas {{ display: block; width: {spec.width}px; height: {spec.height}px; }}
</style>
</head>
<body>
<canvas id="stage" width="{spec.width}" height="{spec.height}"></canvas>
<script>
const SPEC = {spec_json};
const canvas = document.getElementById('stage');
const ctx = canvas.getContext('2d');
const W = SPEC.width;
const H = SPEC.height;
const PAL = SPEC.theme_palette || {{}};

function easeInOut(t) {{
  return t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2)/2;
}}

function clamp01(v) {{ return Math.max(0, Math.min(1, v)); }}

function drawRoundedRect(x, y, w, h, r, fill, stroke, lw=2) {{
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  if (fill) {{ ctx.fillStyle = fill; ctx.fill(); }}
  if (stroke) {{ ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }}
}}

function drawArrow(x1, y1, x2, y2, color, lw=3) {{
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 12*Math.cos(angle - Math.PI/6), y2 - 12*Math.sin(angle - Math.PI/6));
  ctx.lineTo(x2 - 12*Math.cos(angle + Math.PI/6), y2 - 12*Math.sin(angle + Math.PI/6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}}

function drawBackdropCard(alpha, title, badgeText) {{
  const padX = W * 0.06;
  const padY = H * 0.08;
  const cardW = W - padX * 2;
  const cardH = H - padY * 2;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.shadowColor = 'rgba(0,0,0,0.55)';
  ctx.shadowBlur = 32;
  drawRoundedRect(padX, padY, cardW, cardH, 28, 'rgba(11, 16, 32, 0.92)', 'rgba(56, 189, 248, 0.42)', 2.5);
  ctx.shadowBlur = 0;

  // Subtle grid lines
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
  ctx.lineWidth = 1;
  for (let gx = padX + 40; gx < padX + cardW; gx += 48) {{
    ctx.beginPath(); ctx.moveTo(gx, padY + 80); ctx.lineTo(gx, padY + cardH - 20); ctx.stroke();
  }}
  for (let gy = padY + 90; gy < padY + cardH; gy += 48) {{
    ctx.beginPath(); ctx.moveTo(padX + 20, gy); ctx.lineTo(padX + cardW - 20, gy); ctx.stroke();
  }}

  // Header badge & title
  drawRoundedRect(padX + 36, padY + 28, 250, 36, 18, 'rgba(56, 189, 248, 0.16)', '#38BDF8', 1.5);
  ctx.fillStyle = '#7DD3FC';
  ctx.font = '700 18px Helvetica Neue, sans-serif';
  ctx.fillText(badgeText, padX + 54, padY + 52);

  ctx.fillStyle = '#F8FAFC';
  ctx.font = '700 34px Helvetica Neue, sans-serif';
  ctx.fillText(title, padX + 310, padY + 56);
  ctx.restore();
  return {{ padX, padY, cardW, cardH }};
}}

function renderEmInduction(t, dur, box) {{
  const {{ padX, padY, cardW, cardH }} = box;
  const p = t / Math.max(0.01, dur);
  const omega = 2 * Math.PI * (SPEC.parameters.oscillations || 2.0);
  // Magnet X oscillates smoothly near coil center
  const coilX = padX + cardW * 0.56;
  const midY = padY + cardH * 0.44;
  const magnetOffset = -Math.cos(omega * p) * (cardW * 0.16) - cardW * 0.14;
  const vel = Math.sin(omega * p); // derivative sign
  const magnetX = coilX + magnetOffset;
  const magnetW = cardW * 0.16;
  const magnetH = 68;

  // Magnetic field lines (curved B loops around magnet)
  ctx.save();
  for (let k = -3; k <= 3; k++) {{
    if (k === 0) continue;
    const spread = k * 34;
    ctx.strokeStyle = 'rgba(56, 189, 248, ' + (0.28 + 0.18 * Math.abs(Math.sin(omega * p))) + ')';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(magnetX + magnetW * 0.5, midY + k * 6);
    ctx.bezierCurveTo(
      magnetX + magnetW * 1.45, midY + spread * 1.8,
      magnetX - magnetW * 0.45, midY + spread * 1.8,
      magnetX - magnetW * 0.5, midY + k * 6
    );
    ctx.stroke();
  }}
  ctx.restore();

  // Copper coil turns (back halves)
  const turns = SPEC.parameters.turns_N || 6;
  const coilW = cardW * 0.18;
  const coilR = 86;
  ctx.save();
  for (let i = 0; i < turns; i++) {{
    const tx = coilX - coilW * 0.5 + (i / Math.max(1, turns - 1)) * coilW;
    ctx.strokeStyle = '#D97706';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.ellipse(tx, midY, 18, coilR, 0, Math.PI * 0.5, Math.PI * 1.5);
    ctx.stroke();
  }}

  // Bar Magnet
  ctx.shadowColor = 'rgba(0,0,0,0.5)';
  ctx.shadowBlur = 16;
  drawRoundedRect(magnetX - magnetW * 0.5, midY - magnetH * 0.5, magnetW * 0.5, magnetH, 8, '#334155', '#94A3B8', 2);
  drawRoundedRect(magnetX, midY - magnetH * 0.5, magnetW * 0.5, magnetH, 8, '#EF4444', '#FCA5A5', 2);
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#FFFFFF';
  ctx.font = '800 26px Helvetica Neue, sans-serif';
  ctx.fillText('S', magnetX - magnetW * 0.30, midY + 9);
  ctx.fillText('N', magnetX + magnetW * 0.18, midY + 9);

  // Copper coil front halves + induced current glow
  const emfNorm = vel; // E = -dPhi/dt proportional to velocity
  const glowAmt = Math.min(1, Math.abs(emfNorm));
  for (let i = 0; i < turns; i++) {{
    const tx = coilX - coilW * 0.5 + (i / Math.max(1, turns - 1)) * coilW;
    ctx.strokeStyle = glowAmt > 0.25 ? '#FBBF24' : '#F59E0B';
    ctx.shadowColor = '#F59E0B';
    ctx.shadowBlur = 14 * glowAmt;
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.ellipse(tx, midY, 18, coilR, 0, -Math.PI * 0.5, Math.PI * 0.5);
    ctx.stroke();
  }}
  ctx.shadowBlur = 0;

  // Induced current arrow along wire loop
  if (Math.abs(emfNorm) > 0.12) {{
    const dir = emfNorm > 0 ? 1 : -1;
    const ax1 = coilX - 60 * dir;
    const ax2 = coilX + 60 * dir;
    drawArrow(ax1, midY + coilR + 28, ax2, midY + coilR + 28, '#34D399', 4.5);
    ctx.fillStyle = '#34D399';
    ctx.font = '700 21px Helvetica Neue, sans-serif';
    ctx.fillText('Induced Current I (opposes dPhi_B/dt)', coilX - 185, midY + coilR + 58);
  }}

  // Live Faraday Waveform & Flux plot bottom panel
  const chartX = padX + 48;
  const chartY = padY + cardH * 0.67;
  const chartW = cardW - 96;
  const chartH = cardH * 0.24;
  drawRoundedRect(chartX, chartY, chartW, chartH, 16, 'rgba(15, 23, 42, 0.9)', 'rgba(148, 163, 184, 0.3)', 1.5);

  const midLine = chartY + chartH * 0.55;
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(chartX + 20, midLine); ctx.lineTo(chartX + chartW - 20, midLine); ctx.stroke();

  // Plot Flux Phi_B(t) in cyan and EMF E(t) = -N dPhi_B/dt in amber
  const samples = 180;
  const maxIdx = Math.max(2, Math.floor(samples * p));
  ctx.strokeStyle = '#38BDF8';
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let s = 0; s <= maxIdx; s++) {{
    const u = s / samples;
    const x = chartX + 24 + u * (chartW - 48);
    const phi = 0.5 + 0.45 * (-Math.cos(omega * u));
    const y = midLine - (phi - 0.5) * (chartH * 0.62);
    if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  ctx.strokeStyle = '#F59E0B';
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  for (let s = 0; s <= maxIdx; s++) {{
    const u = s / samples;
    const x = chartX + 24 + u * (chartW - 48);
    const emf = -Math.sin(omega * u);
    const y = midLine - emf * (chartH * 0.34);
    if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  // Labels & Faraday Equation
  ctx.fillStyle = '#38BDF8';
  ctx.font = '700 20px Helvetica Neue, sans-serif';
  ctx.fillText('Flux Phi_B(t)', chartX + 26, chartY + 30);
  ctx.fillStyle = '#F59E0B';
  ctx.fillText('Induced EMF = -N dPhi_B / dt', chartX + 210, chartY + 30);

  ctx.fillStyle = '#F8FAFC';
  ctx.font = '700 26px Helvetica Neue, sans-serif';
  ctx.fillText('E = -N (dPhi_B / dt)   |   N = ' + turns + ' turns', padX + 48, padY + 122);
  ctx.restore();
}}

function renderRlcCircuit(t, dur, box) {{
  const {{ padX, padY, cardW, cardH }} = box;
  const p = t / Math.max(0.01, dur);
  const lx = padX + 80;
  const ly = padY + 150;
  const lw = cardW * 0.42;
  const lh = cardH * 0.58;

  ctx.save();
  ctx.strokeStyle = '#38BDF8';
  ctx.lineWidth = 5;
  ctx.strokeRect(lx, ly, lw, lh);

  // Moving electron current pulses along perimeter
  const alpha = SPEC.parameters.damping_alpha || 0.45;
  const freq = SPEC.parameters.freq_hz || 1.6;
  const env = Math.exp(-alpha * t * 0.5);
  for (let k = 0; k < 12; k++) {{
    const u = (k / 12 + p * 0.85) % 1.0;
    let px = lx, py = ly;
    if (u < 0.25) {{ px = lx + (u / 0.25) * lw; py = ly; }}
    else if (u < 0.5) {{ px = lx + lw; py = ly + ((u - 0.25) / 0.25) * lh; }}
    else if (u < 0.75) {{ px = lx + lw - ((u - 0.5) / 0.25) * lw; py = ly + lh; }}
    else {{ px = lx; py = ly + lh - ((u - 0.75) / 0.25) * lh; }}
    ctx.fillStyle = '#34D399';
    ctx.beginPath();
    ctx.arc(px, py, 7 + 3 * env, 0, Math.PI * 2);
    ctx.fill();
  }}

  // Component labels
  drawRoundedRect(lx + lw * 0.35, ly - 24, 140, 48, 10, '#1E293B', '#F59E0B', 2.5);
  ctx.fillStyle = '#FDE68A'; ctx.font = '700 22px Helvetica Neue';
  ctx.fillText('R + L', lx + lw * 0.35 + 38, ly + 8);

  drawRoundedRect(lx + lw - 65, ly + lh * 0.42, 130, 52, 10, '#1E293B', '#38BDF8', 2.5);
  ctx.fillStyle = '#BAE6FD';
  ctx.fillText('C (Load)', lx + lw - 45, ly + lh * 0.42 + 33);

  // Right-side oscilloscope trace
  const ox = padX + cardW * 0.55;
  const oy = ly;
  const ow = cardW * 0.38;
  const oh = lh;
  drawRoundedRect(ox, oy, ow, oh, 18, 'rgba(15,23,42,0.95)', '#38BDF8', 2);
  ctx.strokeStyle = '#34D399'; ctx.lineWidth = 4;
  ctx.beginPath();
  for (let i = 0; i <= 160; i++) {{
    const u = i / 160;
    if (u > p) break;
    const x = ox + 20 + u * (ow - 40);
    const val = Math.exp(-alpha * u * dur * 0.55) * Math.sin(2 * Math.PI * freq * u * dur);
    const y = oy + oh * 0.5 - val * (oh * 0.36);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();
  ctx.fillStyle = '#F8FAFC'; ctx.font = '700 22px Helvetica Neue';
  ctx.fillText('Oscilloscope: Damped V_out(t)', ox + 24, oy + 36);
  ctx.restore();
}}

function renderTreeTrace(t, dur, box) {{
  const {{ padX, padY, cardW, cardH }} = box;
  const p = t / Math.max(0.01, dur);
  const step = Math.min(3, Math.floor(p * 4));
  const nodes = [
    {{ val: 32, x: 0.32, y: 0.25, activeAt: 0 }},
    {{ val: 16, x: 0.19, y: 0.48, activeAt: 99 }},
    {{ val: 48, x: 0.45, y: 0.48, activeAt: 1 }},
    {{ val: 40, x: 0.38, y: 0.72, activeAt: 2 }},
    {{ val: 56, x: 0.52, y: 0.72, activeAt: 99 }},
    {{ val: 42, x: 0.42, y: 0.88, activeAt: 3 }},
  ];
  const edges = [[0,1],[0,2],[2,3],[2,4],[3,5]];
  ctx.save();
  for (const [a, b] of edges) {{
    const na = nodes[a], nb = nodes[b];
    const lit = step >= nb.activeAt && step >= na.activeAt && nb.activeAt < 90;
    ctx.strokeStyle = lit ? '#34D399' : 'rgba(148,163,184,0.4)';
    ctx.lineWidth = lit ? 5 : 2.5;
    ctx.beginPath();
    ctx.moveTo(padX + cardW * na.x, padY + cardH * na.y);
    ctx.lineTo(padX + cardW * nb.x, padY + cardH * nb.y);
    ctx.stroke();
  }}
  for (const n of nodes) {{
    const cx = padX + cardW * n.x;
    const cy = padY + cardH * n.y;
    const active = step >= n.activeAt && n.activeAt < 90;
    ctx.fillStyle = active ? '#059669' : '#1E293B';
    ctx.strokeStyle = active ? '#34D399' : '#38BDF8';
    ctx.lineWidth = active ? 4 : 2;
    ctx.beginPath(); ctx.arc(cx, cy, 32, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#F8FAFC'; ctx.font = '700 22px Helvetica Neue';
    ctx.fillText(String(n.val), cx - 13, cy + 8);
  }}
  // Complexity panel on right
  const rx = padX + cardW * 0.62;
  const ry = padY + 140;
  const rw = cardW * 0.33;
  const rh = cardH * 0.65;
  drawRoundedRect(rx, ry, rw, rh, 18, 'rgba(15,23,42,0.95)', '#38BDF8', 2);
  ctx.fillStyle = '#F8FAFC'; ctx.font = '700 24px Helvetica Neue';
  ctx.fillText('BST Lookup Trace: key = 42', rx + 24, ry + 42);
  const lines = [
    'Step 1: 42 > 32  -> go RIGHT',
    'Step 2: 42 < 48  -> go LEFT',
    'Step 3: 42 > 40  -> go RIGHT',
    'Step 4: Found 42 in O(log n)!',
  ];
  lines.forEach((ln, idx) => {{
    ctx.fillStyle = idx <= step ? '#34D399' : '#64748B';
    ctx.font = '600 20px Menlo, monospace';
    ctx.fillText(ln, rx + 24, ry + 100 + idx * 54);
  }});
  ctx.restore();
}}

function renderTimeline(t, dur, box) {{
  const {{ padX, padY, cardW, cardH }} = box;
  const p = easeInOut(clamp01(t / Math.max(0.01, dur)));
  const items = SPEC.parameters.milestones || [];
  const yRail = padY + cardH * 0.52;
  const x0 = padX + 90;
  const x1 = padX + cardW - 90;

  ctx.save();
  ctx.strokeStyle = 'rgba(148,163,184,0.3)';
  ctx.lineWidth = 6;
  ctx.beginPath(); ctx.moveTo(x0, yRail); ctx.lineTo(x1, yRail); ctx.stroke();

  ctx.strokeStyle = '#38BDF8';
  ctx.lineWidth = 6;
  ctx.beginPath(); ctx.moveTo(x0, yRail); ctx.lineTo(x0 + (x1 - x0) * p, yRail); ctx.stroke();

  items.forEach((m, idx) => {{
    const frac = items.length <= 1 ? 0.5 : idx / (items.length - 1);
    const cx = x0 + (x1 - x0) * frac;
    const lit = p >= frac * 0.85;
    ctx.fillStyle = lit ? '#F59E0B' : '#1E293B';
    ctx.strokeStyle = lit ? '#FDE68A' : '#64748B';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(cx, yRail, 24, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#0B1020'; ctx.font = '800 18px Helvetica Neue';
    ctx.fillText(String(m.tag || idx + 1), cx - 10, yRail + 6);

    const cardWItem = 280;
    const cardY = idx % 2 === 0 ? yRail - 170 : yRail + 55;
    drawRoundedRect(cx - cardWItem/2, cardY, cardWItem, 110, 14, 'rgba(30,41,59,0.92)', lit ? '#F59E0B' : '#334155', 2);
    ctx.fillStyle = '#F8FAFC'; ctx.font = '700 22px Helvetica Neue';
    ctx.fillText(String(m.title || ''), cx - cardWItem/2 + 18, cardY + 42);
    ctx.fillStyle = '#CBD5E1'; ctx.font = '500 18px Helvetica Neue';
    ctx.fillText(String(m.detail || '').slice(0, 30), cx - cardWItem/2 + 18, cardY + 78);
  }});
  ctx.restore();
}}

window.renderAtTime = function(t, dur) {{
  ctx.clearRect(0, 0, W, H);
  const fade = Math.min(1, t / 0.35, Math.max(0.0, (dur - t) / 0.35));
  const box = drawBackdropCard(
    0.96 * Math.max(0.15, fade),
    SPEC.title,
    SPEC.factuality_mode === 'factual_scientific' ? 'FACTUAL MODEL' : 'STORY GRAPHIC'
  );
  if (SPEC.template_id === 'em_induction_faraday') {{
    renderEmInduction(t, dur, box);
  }} else if (SPEC.template_id === 'rlc_signal_flow') {{
    renderRlcCircuit(t, dur, box);
  }} else if (SPEC.template_id === 'binary_tree_trace') {{
    renderTreeTrace(t, dur, box);
  }} else {{
    renderTimeline(t, dur, box);
  }}
}};

window.renderAtTime(0.0, SPEC.duration_sec);
</script>
</body>
</html>"""


def _find_node_playwright_core() -> Path | None:
    """Locate playwright-core ESM entrypoint in repo web/node_modules or parent workspace."""
    for base in Path(__file__).resolve().parents:
        for candidate in (
            base / "web" / "node_modules" / "playwright-core" / "index.mjs",
            base / "node_modules" / "playwright-core" / "index.mjs",
        ):
            if candidate.exists():
                return candidate
    return None


def _find_chromium_executable() -> str | None:
    """Find locally cached Playwright Chromium headless shell or system Chrome."""
    env_path = os.environ.get("SYMPOSIUM_CHROMIUM_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    cache_roots = [
        Path.home() / "Library" / "Caches" / "ms-playwright",
        Path.home() / ".cache" / "ms-playwright",
    ]
    for root in cache_roots:
        if not root.exists():
            continue
        for pattern in (
            "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell",
            "chromium_headless_shell-*/chrome-linux/headless_shell",
            "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
            "chromium-*/chrome-linux/chrome",
        ):
            matches = sorted(root.glob(pattern), reverse=True)
            for m in matches:
                if m.is_file() and os.access(m, os.X_OK):
                    return str(m)

    for sys_bin in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("chromium") or "",
        shutil.which("google-chrome") or "",
    ):
        if sys_bin and Path(sys_bin).exists():
            return sys_bin
    return None


class ChromiumCanvasAdapter(AnimationAdapter):
    """Primary built-in HTML5 Canvas/SVG animation renderer via headless Chromium."""

    def _runtime_backend(self) -> tuple[str | None, str]:
        try:
            import playwright  # noqa: F401
            return "python_playwright", "Headless Chromium Playwright (Python) ready."
        except Exception:
            pass

        node_bin = shutil.which("node")
        pw_core = _find_node_playwright_core()
        chrome_bin = _find_chromium_executable()
        if node_bin and pw_core and chrome_bin:
            return (
                "node_playwright_core",
                f"Headless Chromium via Node playwright-core + cached binary ({Path(chrome_bin).name}).",
            )
        missing = []
        if not node_bin:
            missing.append("node")
        if not pw_core:
            missing.append("playwright-core")
        if not chrome_bin:
            missing.append("chromium executable")
        return None, f"Unavailable: neither Python playwright nor Node backend ({', '.join(missing)}) found."

    def capability(self) -> AnimationAdapterCapability:
        backend, msg = self._runtime_backend()
        available = backend is not None

        return AnimationAdapterCapability(
            adapter_id="chromium_canvas",
            display_name="Chromium Canvas / SVG Renderer (Playwright)",
            version="1.0.0",
            available=available,
            license_name="Apache-2.0 / BSD",
            license_tier=AdapterLicenseTier.PERMISSIVE_OPEN,
            supports_alpha=True,
            supported_outputs=["mov_prores4444_alpha", "mp4_h264", "png_rgba_sequence"],
            best_domains=[
                AnimationDomain.PHYSICS,
                AnimationDomain.ELECTRICAL_ENGINEERING,
                AnimationDomain.PROGRAMMING,
                AnimationDomain.STORYTELLING,
                AnimationDomain.DATA_CHART,
            ],
            constraints=["Requires headless Chromium executable cached locally."],
            relative_cost="low",
            health_message=msg,
        )

    def render(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        backend, msg = self._runtime_backend()
        if backend is None:
            raise RuntimeError(f"ChromiumCanvasAdapter cannot render: {msg}")

        t0 = time.monotonic()
        output_dir.mkdir(parents=True, exist_ok=True)
        spec_path = output_dir / f"{scene_spec.scene_id}.spec.json"
        spec_path.write_text(scene_spec.model_dump_json(indent=2), encoding="utf-8")

        html_path = output_dir / f"{scene_spec.scene_id}.html"
        html_path.write_text(_build_canvas_html(scene_spec), encoding="utf-8")

        frames_dir = output_dir / f"{scene_spec.scene_id}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        total_frames = max(1, int(round(scene_spec.duration_sec * scene_spec.fps)))
        if backend == "python_playwright":
            from playwright.sync_api import sync_playwright

            launch_kwargs: dict[str, object] = {"headless": True}
            chrome_bin = _find_chromium_executable()
            if chrome_bin:
                launch_kwargs["executable_path"] = chrome_bin
            with sync_playwright() as p:
                browser = p.chromium.launch(**launch_kwargs)
                page = browser.new_page(viewport={"width": scene_spec.width, "height": scene_spec.height})
                page.goto(html_path.resolve().as_uri())
                for frame_idx in range(total_frames):
                    t_sec = frame_idx / scene_spec.fps
                    page.evaluate("([t, dur]) => window.renderAtTime(t, dur)", [t_sec, scene_spec.duration_sec])
                    frame_file = frames_dir / f"frame_{frame_idx:05d}.png"
                    page.screenshot(path=str(frame_file), omit_background=scene_spec.transparent_background)
                browser.close()
        else:
            node_bin = shutil.which("node") or "node"
            pw_core = _find_node_playwright_core()
            chrome_bin = _find_chromium_executable() or ""
            script_path = Path(__file__).with_name("render_canvas.mjs")
            subprocess.run(
                [
                    node_bin,
                    str(script_path),
                    "--html", str(html_path),
                    "--out", str(frames_dir),
                    "--width", str(scene_spec.width),
                    "--height", str(scene_spec.height),
                    "--fps", str(scene_spec.fps),
                    "--duration", str(scene_spec.duration_sec),
                    "--transparent", "true" if scene_spec.transparent_background else "false",
                    "--pwModule", str(pw_core),
                    "--executable", chrome_bin,
                ],
                check=True,
            )

        # Encode video with alpha when transparent_background=True (ProRes4444 MOV) or MP4
        if scene_spec.transparent_background:
            out_video = output_dir / f"{scene_spec.scene_id}.mov"
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(scene_spec.fps),
                "-i",
                str(frames_dir / "frame_%05d.png"),
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4444",
                "-pix_fmt",
                "yuva444p10le",
                str(out_video),
            ]
        else:
            out_video = output_dir / f"{scene_spec.scene_id}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(scene_spec.fps),
                "-i",
                str(frames_dir / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(out_video),
            ]
        subprocess.run(cmd, check=True)

        return AnimationRenderResult(
            scene_id=scene_spec.scene_id,
            adapter_used="chromium_canvas",
            output_path=str(out_video),
            frame_sequence_dir=str(frames_dir),
            has_alpha=scene_spec.transparent_background,
            width=scene_spec.width,
            height=scene_spec.height,
            fps=float(scene_spec.fps),
            duration_sec=scene_spec.duration_sec,
            scene_spec_path=str(spec_path),
            render_time_sec=round(time.monotonic() - t0, 3),
            factuality_mode=scene_spec.factuality_mode,
        )
