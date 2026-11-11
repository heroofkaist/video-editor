"""Lightweight Matplotlib scientific & vector animation adapter (fast deterministic fallback)."""

from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symposium_studio.engines.animation.base import AnimationAdapter
from symposium_studio.models.animation import (
    AdapterLicenseTier,
    AnimationAdapterCapability,
    AnimationDomain,
    AnimationRenderResult,
    SceneSpec,
)


class MatplotlibVectorAdapter(AnimationAdapter):
    """Pure-Python scientific plotting and schematic renderer (PSF/BSD)."""

    def capability(self) -> AnimationAdapterCapability:
        return AnimationAdapterCapability(
            adapter_id="matplotlib_vector",
            display_name="Matplotlib Scientific Vector Renderer",
            version=matplotlib.__version__,
            available=True,
            license_name="PSF / Matplotlib License",
            license_tier=AdapterLicenseTier.PERMISSIVE_OPEN,
            supports_alpha=True,
            supported_outputs=["mov_prores4444_alpha", "mp4_h264", "png_rgba_sequence"],
            best_domains=[
                AnimationDomain.PHYSICS,
                AnimationDomain.ELECTRICAL_ENGINEERING,
                AnimationDomain.DATA_CHART,
            ],
            constraints=["Best for quantitative equations, waveforms, and factual plots."],
            relative_cost="very_low",
            health_message="Matplotlib Agg ready.",
        )

    def render(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        t0 = time.monotonic()
        output_dir.mkdir(parents=True, exist_ok=True)
        spec_path = output_dir / f"{scene_spec.scene_id}.spec.json"
        spec_path.write_text(scene_spec.model_dump_json(indent=2), encoding="utf-8")

        frames_dir = output_dir / f"{scene_spec.scene_id}_mpl_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        dpi = 100
        fig_w = scene_spec.width / dpi
        fig_h = scene_spec.height / dpi
        total_frames = max(1, int(round(scene_spec.duration_sec * scene_spec.fps)))
        osc = float(scene_spec.parameters.get("oscillations", 2.0))
        turns = int(scene_spec.parameters.get("turns_N", 6))

        t_grid = np.linspace(0, scene_spec.duration_sec, 300)
        omega = 2.0 * math.pi * osc / max(0.1, scene_spec.duration_sec)
        phi_grid = 1.0 - np.cos(omega * t_grid)
        emf_grid = -turns * np.sin(omega * t_grid)

        for f_idx in range(total_frames):
            t_cur = f_idx / scene_spec.fps
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            fig.patch.set_alpha(0.0 if scene_spec.transparent_background else 1.0)
            fig.patch.set_facecolor("#0B1020")
            ax.set_facecolor((11 / 255, 16 / 255, 32 / 255, 0.90))

            mask = t_grid <= t_cur + 1e-6
            ax.plot(t_grid[mask], phi_grid[mask], color="#38BDF8", linewidth=3.5, label=r"Flux $\Phi_B(t)$")
            ax.plot(t_grid[mask], emf_grid[mask] / max(1, turns), color="#F59E0B", linewidth=3.5, label=r"Normalized EMF $\mathcal{E} = -N d\Phi_B/dt$")
            ax.set_xlim(0, scene_spec.duration_sec)
            ax.set_ylim(-1.4, 2.3)
            ax.set_title(scene_spec.title, color="#F8FAFC", fontsize=22, fontweight="bold", pad=16)
            ax.tick_params(colors="#94A3B8", labelsize=13)
            ax.grid(True, color="#334155", alpha=0.4)
            ax.legend(loc="upper right", facecolor="#0F172A", edgecolor="#38BDF8", labelcolor="#F8FAFC", fontsize=14)
            fig.tight_layout(pad=2.5)
            fig.savefig(frames_dir / f"frame_{f_idx:05d}.png", dpi=dpi, transparent=scene_spec.transparent_background)
            plt.close(fig)

        out_path = output_dir / (f"{scene_spec.scene_id}.mov" if scene_spec.transparent_background else f"{scene_spec.scene_id}.mp4")
        if scene_spec.transparent_background:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(scene_spec.fps),
                "-i", str(frames_dir / "frame_%05d.png"),
                "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                str(out_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(scene_spec.fps),
                "-i", str(frames_dir / "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(out_path),
            ]
        subprocess.run(cmd, check=True)

        return AnimationRenderResult(
            scene_id=scene_spec.scene_id,
            adapter_used="matplotlib_vector",
            output_path=str(out_path),
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
