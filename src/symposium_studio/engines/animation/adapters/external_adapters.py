"""External & opt-in animation adapters: Motion Canvas (MIT), Manim (MIT), Blender (GPL-3.0), Remotion (commercial-gated).

Security guarantee: all SceneSpec fields (titles, subtitles, parameters) are serialized via
`json.dumps` / Python `repr(json_str)` rather than unescaped string interpolation, preventing
arbitrary TypeScript/Python/HTML injection via user or transcript metadata.
"""

from __future__ import annotations

import json
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
    """Generates parameterized TypeScript Motion Canvas scenes (@motion-canvas/core, MIT) and executes Vite export when initialized."""

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
            health_message="Node present; Motion Canvas project workspace ready." if (node_ok and proj_ok) else (
                "Node present; Motion Canvas project workspace optional." if node_ok else "Node binary missing."
            ),
        )

    def emit_typescript_scene(self, scene_spec: SceneSpec, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        tsx_path = out_dir / f"{scene_spec.scene_id}.tsx"
        safe_spec_lit = json.dumps(scene_spec.model_dump(mode="json"))
        duration = max(0.5, float(scene_spec.duration_sec))
        tsx_path.write_text(
            f"""import {{Circle, Line, Rect, Txt, makeScene2D}} from '@motion-canvas/2d';
import {{all, createRef, waitFor}} from '@motion-canvas/core';

const SCENE_SPEC = {safe_spec_lit} as const;

export default makeScene2D(function* (view) {{
  const titleRef = createRef<Txt>();
  const cardRef = createRef<Rect>();
  const pulseRef = createRef<Circle>();
  const dur = Math.max(0.8, Number(SCENE_SPEC.duration_sec) || {duration:.2f});
  const params: Record<string, any> = (SCENE_SPEC.parameters as any) || {{}};
  const stages = Array.isArray(params.stages)
    ? params.stages
    : Array.isArray(params.milestones)
    ? params.milestones
    : [
        {{name: SCENE_SPEC.template_id, metric: SCENE_SPEC.domain}},
        {{name: 'Analytical Step', metric: 'Synchronized'}},
        {{name: 'Verified Output', metric: SCENE_SPEC.factuality_mode}},
      ];

  view.add(
    <Rect ref={{cardRef}} width={{1680}} height={{880}} radius={{28}} fill="#0B1020" stroke="#38BDF8" lineWidth={{3}}>
      <Txt
        ref={{titleRef}}
        y={{-340}}
        text={{String(SCENE_SPEC.title)}}
        fill="#F8FAFC"
        fontSize={{44}}
        fontWeight={{700}}
      />
      <Line points={{[[-620, 40], [620, 40]]}} stroke="#334155" lineWidth={{6}} />
      <Circle ref={{pulseRef}} x={{-620}} y={{40}} size={{26}} fill="#38BDF8" />
    </Rect>,
  );

  const count = Math.max(1, stages.length);
  for (let i = 0; i < count; i++) {{
    const xPos = count === 1 ? 0 : -540 + (1080 * i) / (count - 1);
    const label = String((stages[i] && (stages[i].name || stages[i].title)) || 'Stage');
    const sub = String((stages[i] && (stages[i].metric || stages[i].detail)) || '');
    cardRef().add(
      <Rect x={{xPos}} y={{40}} width={{260}} height={{130}} radius={{16}} fill="#1E293B" stroke="#38BDF8" lineWidth={{2}}>
        <Txt y={{-18}} text={{label}} fill="#F8FAFC" fontSize={{24}} fontWeight={{700}} />
        <Txt y={{24}} text={{sub}} fill="#94A3B8" fontSize={{18}} />
      </Rect>,
    );
  }}

  yield* all(
    cardRef().scale(1.02, Math.min(1.0, dur * 0.25)),
    pulseRef().position.x(620, dur * 0.75),
  );
  yield* waitFor(Math.max(0.2, dur * 0.25));
}});
""",
            encoding="utf-8",
        )
        return tsx_path

    def render(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        t0 = time.monotonic()
        tsx = self.emit_typescript_scene(scene_spec, output_dir)
        cap = self.capability()
        if not cap.available or not self.workspace:
            raise RuntimeError(
                f"Motion Canvas TypeScript source generated at {tsx}, but Motion Canvas runtime workspace is not initialized."
            )

        spec_path = output_dir / f"{scene_spec.scene_id}.spec.json"
        spec_path.write_text(scene_spec.model_dump_json(indent=2), encoding="utf-8")
        # Copy scene into Motion Canvas workspace src/scenes and invoke project exporter script if present
        target_scene = self.workspace / "src" / "scenes" / tsx.name
        target_scene.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tsx, target_scene)

        render_script = self.workspace / "render.mjs"
        if not render_script.exists():
            raise RuntimeError(
                f"Motion Canvas workspace '{self.workspace}' lacks headless `render.mjs`; TSX exported cleanly to {target_scene}."
            )
        out_mp4 = output_dir / f"{scene_spec.scene_id}.mp4"
        subprocess.run(
            ["node", str(render_script), "--scene", tsx.stem, "--out", str(out_mp4)],
            cwd=str(self.workspace),
            check=True,
        )
        return AnimationRenderResult(
            scene_id=scene_spec.scene_id,
            adapter_used="motion_canvas",
            output_path=str(out_mp4),
            has_alpha=False,
            width=scene_spec.width,
            height=scene_spec.height,
            fps=float(scene_spec.fps),
            duration_sec=scene_spec.duration_sec,
            scene_spec_path=str(spec_path),
            render_time_sec=round(time.monotonic() - t0, 3),
            factuality_mode=scene_spec.factuality_mode,
        )


class ManimAdapter(AnimationAdapter):
    """Optional Manim Community (MIT) adapter with parameterized plots, circuits, trees, and equations."""

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
        spec_repr = repr(scene_spec.model_dump_json())
        py_path.write_text(
            f"""import json
import math
from manim import *

SPEC = json.loads({spec_repr})


class GeneratedScene(Scene):
    def construct(self):
        tmpl = SPEC.get("template_id", "em_induction_faraday")
        params = SPEC.get("parameters", {{}})
        dur = max(2.0, float(SPEC.get("duration_sec", 6.0)))

        title = Text(str(SPEC.get("title", "Symposium Visual"))[:64], font_size=36, color=BLUE_B).to_edge(UP)
        self.play(Write(title), run_time=0.8)

        if tmpl in ("em_induction_faraday", "rlc_signal_flow", "wave_interference_sim"):
            axes = Axes(x_range=[0, 6.28, 1.57], y_range=[-2, 2, 1], x_length=9.5, y_length=4.0).shift(DOWN * 0.3)
            freq = float(params.get("freq_hz", 1.5))
            alpha = float(params.get("damping_alpha", 0.35)) if tmpl == "rlc_signal_flow" else 0.0
            curve = axes.plot(
                lambda x: math.exp(-alpha * x) * math.sin(freq * x) + (0.6 * math.sin(1.7 * freq * x) if tmpl == "wave_interference_sim" else 0.0),
                color=TEAL_C,
            )
            eq_txt = params.get("equation") or (
                "V(t) = exp(-alpha*t) cos(w_d*t)"
                if tmpl == "rlc_signal_flow"
                else "y(x,t) = y1 + y2"
                if tmpl == "wave_interference_sim"
                else "EMF = -N dPhi_B / dt"
            )
            eq = Text(str(eq_txt)[:48], font_size=28, color=YELLOW).next_to(title, DOWN, buff=0.3)
            self.play(Create(axes), Write(eq), run_time=0.8)
            self.play(Create(curve), run_time=1.4)
        elif tmpl == "binary_tree_trace":
            vals = params.get("values", [32, 16, 48, 8, 24, 40, 56])[:7]
            nodes = VGroup(*[
                VGroup(Circle(radius=0.38, color=BLUE_C), Text(str(v), font_size=22))
                for v in vals
            ]).arrange_in_grid(rows=2, buff=0.7)
            self.play(FadeIn(nodes), run_time=1.2)
        else:
            stages = params.get("stages") or params.get("milestones") or [{{"name": SPEC.get("title", "Stage")}}]
            cards = VGroup(*[
                VGroup(
                    RoundedRectangle(width=2.8, height=1.5, corner_radius=0.2, color=TEAL_B),
                    Text(str(s.get("name") or s.get("title") or "Beat")[:18], font_size=22),
                )
                for s in stages[:4]
            ]).arrange(RIGHT, buff=0.45)
            self.play(FadeIn(cards), run_time=1.2)

        self.wait(max(0.4, dur - 2.6))
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

        cmd = ["manim", "-qh", "--media_dir", str(output_dir / "manim_media")]
        if scene_spec.transparent_background:
            cmd.extend(["-t", "--format", "mov"])
        cmd.extend([str(py_path), "GeneratedScene"])
        subprocess.run(cmd, check=True)

        ext = "*.mov" if scene_spec.transparent_background else "*.mp4"
        outs = sorted((output_dir / "manim_media").rglob(ext))
        if not outs:
            outs = sorted((output_dir / "manim_media").rglob("GeneratedScene.*"))
        if not outs:
            raise RuntimeError("Manim finished without emitting rendered video file.")
        out_file = outs[-1]
        return AnimationRenderResult(
            scene_id=scene_spec.scene_id,
            adapter_used="manim",
            output_path=str(out_file),
            has_alpha=out_file.suffix.lower() == ".mov",
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
            supported_outputs=["mov_prores4444_alpha", "mp4_h264", "png_rgba_sequence"],
            best_domains=[AnimationDomain.PHYSICS, AnimationDomain.STORYTELLING, AnimationDomain.PROGRAMMING],
            constraints=["GPL-3.0 boundary: invoked solely via `blender -b --factory-startup -P`; requires Blender installed."],
            relative_cost="high",
            health_message=f"Blender found at {blender_bin}" if blender_bin else "Blender binary not installed on host.",
        )

    def emit_blender_script(self, scene_spec: SceneSpec, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        script = out_dir / f"{scene_spec.scene_id}_blender.py"
        frames_dir = (out_dir / f"{scene_spec.scene_id}_blender_frames").resolve()
        spec_repr = repr(scene_spec.model_dump_json())
        frames_repr = repr(str(frames_dir / "frame_"))
        script.write_text(
            f"""import json
import math
from pathlib import Path
import bpy

SPEC = json.loads({spec_repr})
FRAMES_PREFIX = {frames_repr}
Path(FRAMES_PREFIX).parent.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = int(SPEC.get("width", 1920))
scene.render.resolution_y = int(SPEC.get("height", 1080))
scene.render.fps = int(SPEC.get("fps", 30))
total_frames = max(1, int(round(float(SPEC.get("duration_sec", 5.0)) * scene.render.fps)))
scene.frame_start = 1
scene.frame_end = total_frames
scene.render.film_transparent = bool(SPEC.get("transparent_background", True))
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA' if scene.render.film_transparent else 'RGB'
scene.render.filepath = FRAMES_PREFIX

# Camera & Area Key Light
bpy.ops.object.camera_add(location=(0.0, -7.2, 2.5), rotation=(math.radians(72.0), 0.0, 0.0))
scene.camera = bpy.context.active_object
bpy.ops.object.light_add(type='AREA', location=(2.5, -4.0, 4.5))
light = bpy.context.active_object
light.data.energy = 320.0

tmpl = SPEC.get("template_id", "em_induction_faraday")
if tmpl == "em_induction_faraday":
    # Copper induction coil + reciprocating magnet bar
    bpy.ops.mesh.primitive_torus_add(major_radius=1.35, minor_radius=0.18, location=(0.0, 0.0, 0.0))
    coil = bpy.context.active_object
    coil.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0.0, -2.0, 0.0))
    magnet = bpy.context.active_object
    magnet.scale = (0.45, 1.4, 0.45)
    for f in range(1, total_frames + 1, max(1, total_frames // 12)):
        frac = (f - 1) / max(1, total_frames - 1)
        magnet.location.y = 2.1 * math.sin(2.0 * math.pi * frac)
        magnet.keyframe_insert(data_path="location", frame=f)
elif tmpl == "system_architecture_flow":
    stages = SPEC.get("parameters", {{}}).get("stages", [1, 2, 3, 4])[:5]
    for i, _s in enumerate(stages):
        x = (i - (len(stages) - 1) / 2.0) * 1.9
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, 0.0, 0.0))
        box = bpy.context.active_object
        box.scale = (0.65, 0.65, 0.85)
        box.keyframe_insert(data_path="rotation_euler", frame=1)
        box.rotation_euler.z = math.radians(360.0)
        box.keyframe_insert(data_path="rotation_euler", frame=total_frames)
else:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.25, location=(0.0, 0.0, 0.0))
    core = bpy.context.active_object
    core.keyframe_insert(data_path="rotation_euler", frame=1)
    core.rotation_euler.z = math.radians(180.0)
    core.keyframe_insert(data_path="rotation_euler", frame=total_frames)

bpy.ops.render.render(animation=True)
""",
            encoding="utf-8",
        )
        return script

    def render(self, scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult:
        t0 = time.monotonic()
        script = self.emit_blender_script(scene_spec, output_dir)
        blender_bin = shutil.which("blender")
        if not blender_bin:
            raise RuntimeError(f"Blender script generated at {script}, but Blender is not installed.")

        spec_path = output_dir / f"{scene_spec.scene_id}.spec.json"
        spec_path.write_text(scene_spec.model_dump_json(indent=2), encoding="utf-8")

        subprocess.run(
            [blender_bin, "-b", "--factory-startup", "-P", str(script)],
            check=True,
        )
        frames_dir = output_dir / f"{scene_spec.scene_id}_blender_frames"
        pattern = str(frames_dir / "frame_%04d.png")
        has_alpha = scene_spec.transparent_background
        if has_alpha:
            out_file = output_dir / f"{scene_spec.scene_id}.mov"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(scene_spec.fps),
                "-i", pattern,
                "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le",
                str(out_file),
            ]
        else:
            out_file = output_dir / f"{scene_spec.scene_id}.mp4"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(scene_spec.fps),
                "-i", pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out_file),
            ]
        subprocess.run(cmd, check=True)
        return AnimationRenderResult(
            scene_id=scene_spec.scene_id,
            adapter_used="blender",
            output_path=str(out_file),
            has_alpha=has_alpha,
            width=scene_spec.width,
            height=scene_spec.height,
            fps=float(scene_spec.fps),
            duration_sec=scene_spec.duration_sec,
            scene_spec_path=str(spec_path),
            render_time_sec=round(time.monotonic() - t0, 3),
            factuality_mode=scene_spec.factuality_mode,
        )


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
