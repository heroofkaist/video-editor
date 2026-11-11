"""Unified FFmpeg video compositor & thumbnail renderer for RenderTimeline."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from symposium_studio.engines.effects.library import EFFECTS_CATALOG
from symposium_studio.engines.effects.sfx_synth import synthesize_sfx
from symposium_studio.engines.production.captions import write_ass, write_srt
from symposium_studio.models.timeline import RenderTimeline, TargetPlatformFormat


def ensure_project_sfx(effects_dir: Path) -> dict[str, str]:
    effects_dir.mkdir(parents=True, exist_ok=True)
    sfx_map: dict[str, str] = {}
    for preset in ("whoosh_transition", "sub_impact", "riser_tension", "ui_pop"):
        p = effects_dir / f"{preset}.wav"
        if not p.exists():
            synthesize_sfx(preset, p)
        sfx_map[preset] = str(p.resolve())
    return sfx_map


def _color_filter_for_style(style_profile: str) -> str:
    if "cinematic" in style_profile or "vlog" in style_profile:
        return "eq=contrast=1.06:saturation=1.05,vignette=PI/7"
    if "high_energy" in style_profile or "short_form" in style_profile or "comedy" in style_profile:
        return "eq=contrast=1.09:saturation=1.14:brightness=0.015,unsharp=3:3:0.35"
    return "eq=contrast=1.05:saturation=1.06,unsharp=3:3:0.25"


def _escape_ffmpeg_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def render_timeline(
    timeline: RenderTimeline,
    output_mp4: Path,
    workspace_dir: Path,
) -> dict[str, str]:
    """Compile RenderTimeline through FFmpeg filtergraphs, animation compositing, SFX mixing, and ASS captions."""
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    if not timeline.clips:
        raise ValueError("Cannot render empty RenderTimeline without clips.")

    is_vertical = timeline.height > timeline.width or timeline.platform_format != TargetPlatformFormat.YOUTUBE_LONG
    w, h = timeline.width, timeline.height
    grade_vf = _color_filter_for_style(timeline.style_profile)

    with tempfile.TemporaryDirectory(prefix="symposium_render_") as tmpdir:
        tmp = Path(tmpdir)
        clip_files: list[Path] = []

        # 1. Render each source segment with face-tracked reframing + subtle editorial zoom + color polish
        for idx, clip in enumerate(timeline.clips):
            clip_out = tmp / f"seg_{idx:03d}.mp4"
            dur = max(0.25, clip.source_end_sec - clip.source_start_sec)
            z = max(1.0, clip.zoom_scale)
            if is_vertical:
                # Crop 9:16 window around tracked face center_x_norm, then apply optional punch zoom
                vf = (
                    f"crop=min(iw\\,ih*9/16):ih:"
                    f"max(0\\,min(iw-min(iw\\,ih*9/16)\\,iw*{clip.crop_center_x:.4f}-min(iw\\,ih*9/16)/2)):0,"
                    f"scale={int(round(w * z))}:{int(round(h * z))}:force_original_aspect_ratio=increase,"
                    f"crop={w}:{h},setsar=1,{grade_vf}"
                )
            else:
                vf = (
                    f"scale={int(round(w * z))}:{int(round(h * z))}:force_original_aspect_ratio=increase,"
                    f"crop={w}:{h},setsar=1,{grade_vf}"
                )

            af = "highpass=f=80,acompressor=threshold=-20dB:ratio=2.4:attack=10:release=120,loudnorm=I=-14:TP=-1.5:LRA=11"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{clip.source_start_sec:.3f}",
                "-t", f"{dur:.3f}",
                "-i", clip.source_path,
                "-vf", vf,
                "-af", af,
                "-r", str(int(round(timeline.fps))),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                str(clip_out),
            ]
            subprocess.run(cmd, check=True)
            clip_files.append(clip_out)

        # 2. Concatenate segments cleanly
        concat_mp4 = tmp / "concat.mp4"
        if len(clip_files) == 1:
            concat_mp4 = clip_files[0]
        else:
            list_txt = tmp / "concat_list.txt"
            list_txt.write_text("\n".join(f"file '{p.as_posix()}'" for p in clip_files), encoding="utf-8")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(list_txt),
                    "-c", "copy",
                    str(concat_mp4),
                ],
                check=True,
            )

        # 3. Composite original animation overlays + mix SFX + burn safe-zone ASS captions
        ass_path = workspace_dir / "timelines" / f"{timeline.timeline_id}.ass"
        srt_path = workspace_dir / "timelines" / f"{timeline.timeline_id}.srt"
        write_srt(timeline, srt_path)
        write_ass(timeline, ass_path)

        inputs = ["-i", str(concat_mp4)]
        filter_parts: list[str] = []
        v_label = "[0:v]"
        a_label = "[0:a]"
        input_idx = 1

        for ov_i, ov in enumerate(timeline.overlays):
            ov_path = Path(ov.asset_path)
            if not ov_path.exists():
                continue
            ov_dur = max(0.5, ov.timeline_end_sec - ov.timeline_start_sec)
            inputs.extend(["-i", str(ov_path)])
            ov_w = max(120, int(round(w * ov.w_norm)))
            ov_h = max(120, int(round(h * ov.h_norm)))
            ov_x = int(round(w * ov.x_norm))
            ov_y = int(round(h * ov.y_norm))
            scaled = f"[ov_s_{ov_i}]"
            next_v = f"[v_ov_{ov_i}]"
            filter_parts.append(
                f"[{input_idx}:v]setpts=PTS-STARTPTS+{ov.timeline_start_sec:.3f}/TB,"
                f"scale={ov_w}:{ov_h}:force_original_aspect_ratio=decrease,"
                f"format=yuva420p{scaled}"
            )
            filter_parts.append(
                f"{v_label}{scaled}overlay=x={ov_x}:y={ov_y}:shortest=0:"
                f"enable='between(t,{ov.timeline_start_sec:.3f},{ov.timeline_end_sec:.3f})'{next_v}"
            )
            v_label = next_v
            input_idx += 1

        if timeline.caption_settings.enabled and timeline.captions:
            next_v = "[v_cap]"
            filter_parts.append(f"{v_label}ass='{_escape_ffmpeg_path(ass_path)}'{next_v}")
            v_label = next_v

        # Mix SFX audio tracks cleanly without attenuating speech
        sfx_inputs_labels: list[str] = []
        for s_i, au in enumerate(timeline.audio_events):
            au_path = Path(au.source_path)
            if not au_path.exists():
                continue
            inputs.extend(["-i", str(au_path)])
            delay_ms = max(0, int(round(au.timeline_start_sec * 1000)))
            s_lbl = f"[sfx_{s_i}]"
            gain_mult = round(10.0 ** (au.gain_db / 20.0), 3)
            filter_parts.append(
                f"[{input_idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
                f"volume={gain_mult},adelay={delay_ms}|{delay_ms}{s_lbl}"
            )
            sfx_inputs_labels.append(s_lbl)
            input_idx += 1

        if sfx_inputs_labels:
            all_a = a_label + "".join(sfx_inputs_labels)
            filter_parts.append(
                f"{all_a}amix=inputs={1 + len(sfx_inputs_labels)}:duration=first:normalize=0,"
                f"loudnorm=I=-14:TP=-1.5:LRA=11[a_final]"
            )
            a_label = "[a_final]"

        if filter_parts:
            filter_complex = ";".join(filter_parts)
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", v_label,
                "-map", a_label,
                "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart",
                str(output_mp4),
            ]
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(concat_mp4), "-c", "copy", "-movflags", "+faststart", str(output_mp4),
                ],
                check=True,
            )

    return {
        "video_path": str(output_mp4),
        "srt_path": str(srt_path),
        "ass_path": str(ass_path),
    }


def render_thumbnail(
    source_video: Path,
    headline: str,
    badge: str,
    out_png: Path,
    capture_sec: float = 2.0,
) -> Path:
    """Extract a sharp frame from rendered/source video and design a high-contrast 1280x720 thumbnail."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="symposium_thumb_") as tmpdir:
        raw_frame = Path(tmpdir) / "frame.png"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{max(0.0, capture_sec):.3f}",
                "-i", str(source_video),
                "-vframes", "1",
                str(raw_frame),
            ],
            check=True,
        )
        base = Image.open(raw_frame).convert("RGBA").resize((1280, 720), Image.Resampling.LANCZOS)

    # Dark left gradient scrim for legible typography
    overlay = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(720):
        alpha = int(205 * max(0.0, 1.0 - x / 720.0))
        draw.line([(x, 0), (x, 720)], fill=(11, 16, 32, alpha))

    combined = Image.alpha_composite(base, overlay)
    d = ImageDraw.Draw(combined)

    font_path = "/System/Library/Fonts/HelveticaNeue.ttc"
    try:
        title_font = ImageFont.truetype(font_path, 66)
        badge_font = ImageFont.truetype(font_path, 28)
    except Exception:
        title_font = ImageFont.load_default()
        badge_font = ImageFont.load_default()

    # Badge pill dynamically sized to actual badge string
    badge_txt = badge[:28].strip().upper()
    bbox = d.textbbox((0, 0), badge_txt, font=badge_font)
    badge_w = max(180, (bbox[2] - bbox[0]) + 44)
    d.rounded_rectangle([58, 74, 58 + badge_w, 128], radius=24, fill=(56, 189, 248, 240))
    d.text((80, 86), badge_txt, font=badge_font, fill=(11, 16, 32, 255))

    # Wrap headline neatly onto max 3 lines within left safe column (<= 580px)
    words = headline.strip().split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        cand_w = d.textbbox((0, 0), cand, font=title_font)[2]
        if cand_w <= 560 or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
        if len(lines) == 3:
            break
    if cur and len(lines) < 3:
        lines.append(cur)

    y = 178
    for ln in lines:
        d.text((62, y + 3), ln, font=title_font, fill=(0, 0, 0, 225))
        d.text((58, y), ln, font=title_font, fill=(248, 250, 252, 255))
        y += 84

    # Accent bottom stripe
    d.rectangle([58, 632, 540, 646], fill=(245, 158, 11, 255))
    combined.convert("RGB").save(out_png, format="PNG")
    return out_png
