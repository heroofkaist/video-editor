"""Optional external & plugin animation adapters: Motion Canvas (MIT), Manim (MIT), Blender (GPL-3.0), Remotion (commercial-gated)."""

from __future__ import annotations

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


class MotionCanvasAdapter(AnimationAdapter):
    """Generates TypeScript Motion Canvas scenes (@motion-canvas/core, MIT) and delegates rendering when project deps exist."""

    def __init__(self, motion_canvas_workspace: Path | None = None) -> None:
        self.workspace = motion_canvas_workspace

    def capability(self) -> AnimationAdapterCapability:
        node_ok = bool(shutil.which("node"))
        proj_ok = bool(self.workspace and (self.workspace / "package.json").exists())
        return AnimationAdapterCapability(
            adapter_id="motion_canvas",
            display_name="Motion Canvas (TypeScript 2D)",
            version="3.17.x",
            available=node_ok and proj_ok,
            license_name="MIT",
            license_tier=AdapterLicenseTier.PERMISSIVE_OPEN,
            supports_alpha=True,
            supported_outputs=["png_rgba_sequence", "mp4_h264"],
            best_domains=[AnimationDomain.PROGRAMMING, AnimationDomain.ELECTRICAL_ENGINEERING, AnimationDomain.STORYTELLING],
            constraints=["Requires initialized Motion Canvas Vite workspace (`npm install @motion-canvas/core @motion-canvas/2d`)."],
            relative_cost="medium",
            health_message="Node present; Motion Canvas project workspace optional." if node_ok else "Node binary missing.",
        )

    def emit_typescript_scene(self, scene_spec: SceneSpec, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        tsx_path = out_dir / f"{scene_spec.scene_id}.tsx"
        tsx_path.write_text(
            f"""import {{makeScene2D, Rect, Txt}} from '@motion-canvas/2d';
import {{all, createRef, waitFor}} from '@motion-canvas/core';

export default makeScene2D(function* (view) {{
  const title = createRef<Txt>();
  const card = createRef<Rect>();
  view.add(
    <Rect ref={{card}} width={{1400}} height={{240}} radius={{24}} fill="#0B1020" stroke="#38BDF8" lineWidth={{3}}>
      <Txt ref={{title}} text="{scene_spec.title}" fill="#F8FAFC" fontSize={{48}} fontWeight={{700}} />
    </Rect>
  );
  yield* all(card().scale(1.03, 0.8), waitFor({max(0.5, scene_spec.duration_sec - 0.8):.2f}));
}});
""",
            encoding="utf-8",
        )
        return tsx_path

    def render(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        cap = self.capability()
        tsx = self.emit_typescript_scene(scene_spec, output_dir)
        if not cap.available:
            raise RuntimeError(
                f"Motion Canvas TypeScript source generated at {tsx}, but Motion Canvas runtime workspace is not initialized."
            )
        raise RuntimeError("Attach Motion Canvas headless Vite render runner to execute workspace.")


class ManimAdapter(AnimationAdapter):
    """Optional Manim Community (MIT) adapter for mathematical plots and derivations."""

    def capability(self) -> AnimationAdapterCapability:
        manim_bin = shutil.which("manim")
        available = bool(manim_bin)
        if not available:
            try:
                import manim  # noqa: F401
                available = True
            except Exception:
                available = False
        return AnimationAdapterCapability(
            adapter_id="manim",
            display_name="Manim Community",
            version="0.22.x",
            available=available,
            license_name="MIT",
            license_tier=AdapterLicenseTier.PERMISSIVE_OPEN,
            supports_alpha=True,
            supported_outputs=["mov_prores4444_alpha", "mp4_h264"],
            best_domains=[AnimationDomain.PHYSICS, AnimationDomain.PROGRAMMING, AnimationDomain.DATA_CHART],
            constraints=["Requires `cairo` + `pango` + `pip install manim` (and MacTeX for MathTex)."],
            relative_cost="medium",
            health_message="Manim ready." if available else "Optional adapter not installed (requires brew pango + pip manim).",
        )

    def emit_manim_script(self, scene_spec: SceneSpec, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        py_path = out_dir / f"{scene_spec.scene_id}_manim.py"
        py_path.write_text(
            f"""from manim import *

class GeneratedScene(Scene):
    def construct(self):
        title = Text("{scene_spec.title}", font_size=40, color=BLUE_B)
        eq = Text("E = -N dPhi_B / dt", font_size=34, color=YELLOW).next_to(title, DOWN, buff=0.6)
        self.play(Write(title))
        self.play(FadeIn(eq))
        self.wait({max(0.5, scene_spec.duration_sec - 2.0):.2f})
""",
            encoding="utf-8",
        )
        return py_path

    def render(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        t0 = time.monotonic()
        py_path = self.emit_manim_script(scene_spec, output_dir)
        if not self.capability().available:
            raise RuntimeError(f"Manim script emitted to {py_path}, but `manim` is not installed.")
        spec_path = output_dir / f"{scene_spec.scene_id}.spec.json"
        spec_path.write_text(scene_spec.model_dump_json(indent=2), encoding="utf-8")
        subprocess.run(
            ["manim", "-qh", "--media_dir", str(output_dir / "manim_media"), str(py_path), "GeneratedScene"],
            check=True,
        )
        mp4s = list((output_dir / "manim_media").rglob("GeneratedScene.mp4"))
        if not mp4s:
            raise RuntimeError("Manim finished without emitting GeneratedScene.mp4")
        return AnimationRenderResult(
            scene_id=scene_spec.scene_id,
            adapter_used="manim",
            output_path=str(mp4s[0]),
            has_alpha=False,
            width=scene_spec.width,
            height=scene_spec.height,
            fps=float(scene_spec.fps),
            duration_sec=scene_spec.duration_sec,
            scene_spec_path=str(spec_path),
            render_time_sec=round(time.monotonic() - t0, 3),
            factuality_mode=scene_spec.factuality_mode,
        )


class BlenderAdapter(AnimationAdapter):
    """Isolated GPL-3.0 external subprocess adapter for procedural 3D animation via Blender."""

    def capability(self) -> AnimationAdapterCapability:
        blender_bin = shutil.which("blender")
        return AnimationAdapterCapability(
            adapter_id="blender",
            display_name="Blender 3D (Isolated Subprocess)",
            version="external",
            available=bool(blender_bin),
            license_name="GPL-3.0 (isolated CLI subprocess)",
            license_tier=AdapterLicenseTier.COPYLEFT_ISOLATED,
            supports_alpha=True,
            supported_outputs=["png_rgba_sequence", "mp4_h264"],
            best_domains=[AnimationDomain.PHYSICS, AnimationDomain.STORYTELLING],
            constraints=["GPL-3.0 boundary: invoked solely via `blender -b -P`; requires Blender installed."],
            relative_cost="high",
            health_message=f"Blender found at {blender_bin}" if blender_bin else "Blender binary not installed on host.",
        )

    def emit_blender_script(self, scene_spec: SceneSpec, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        script = out_dir / f"{scene_spec.scene_id}_blender.py"
        script.write_text(
            f"""import bpy, math
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_torus_add(major_radius=1.4, minor_radius=0.22)
coil = bpy.context.active_object
coil.name = "InductionCoil"
bpy.ops.object.camera_add(location=(0, -5, 2.2), rotation=(math.radians(72), 0, 0))
bpy.context.scene.camera = bpy.context.active_object
bpy.ops.object.light_add(type='AREA', location=(2, -3, 4))
bpy.context.scene.render.resolution_x = {scene_spec.width}
bpy.context.scene.render.resolution_y = {scene_spec.height}
bpy.context.scene.frame_end = {max(1, int(scene_spec.duration_sec * scene_spec.fps))}
""",
            encoding="utf-8",
        )
        return script

    def render(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        script = self.emit_blender_script(scene_spec, output_dir)
        blender_bin = shutil.which("blender")
        if not blender_bin:
            raise RuntimeError(f"Blender script generated at {script}, but Blender is not installed.")
        raise RuntimeError("Configure output frames/render target in Blender script before production invoke.")


class RemotionAdapter(AnimationAdapter):
    """Opt-in React Remotion adapter; strictly gated by license policy (`allow_remotion_adapter`)."""

    def __init__(self, allow_remotion: bool = False) -> None:
        self.allow_remotion = allow_remotion

    def capability(self) -> AnimationAdapterCapability:
        node_ok = bool(shutil.which("node"))
        return AnimationAdapterCapability(
            adapter_id="remotion",
            display_name="Remotion (React Video)",
            version="4.x",
            available=self.allow_remotion and node_ok,
            license_name="Source-available (Company License required for 4+ employees)",
            license_tier=AdapterLicenseTier.COMMERCIAL_RESTRICTED,
            supports_alpha=True,
            supported_outputs=["mp4_h264", "mov_prores4444_alpha"],
            best_domains=[AnimationDomain.STORYTELLING, AnimationDomain.DATA_CHART, AnimationDomain.PROGRAMMING],
            constraints=["Disabled by default unless SYMPOSIUM_ALLOW_REMOTION=1 due to commercial license thresholds."],
            relative_cost="medium",
            health_message="Enabled via policy flag." if self.allow_remotion else "Disabled by default license guard (SYMPOSIUM_ALLOW_REMOTION=1 to enable).",
        )

    def render(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        if not self.allow_remotion:
            raise PermissionError("Remotion adapter blocked by default license policy; set SYMPOSIUM_ALLOW_REMOTION=1 to opt in.")
        raise RuntimeError("Install a local Remotion project directory before invoking Remotion CLI.")
