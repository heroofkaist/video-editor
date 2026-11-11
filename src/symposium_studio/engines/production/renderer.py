"""Unified FFmpeg video compositor & thumbnail renderer for RenderTimeline."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from symposium_studio.engines.effects.library import EFFECTS_CATALOG
from symposium_studio.engines.effects.sfx_synth import synthesize_ambient_bed, synthesize_sfx
from symposium_studio.engines.effects.sound_downloader import DEFAULT_CLICK_SOUNDS, download_click_sounds
from symposium_studio.engines.production.captions import write_ass, write_srt
from symposium_studio.models.timeline import RenderTimeline, TargetPlatformFormat


def ensure_project_sfx(effects_dir: Path, download_clicks: bool = False) -> dict[str, str]:
    effects_dir.mkdir(parents=True, exist_ok=True)
    sfx_map: dict[str, str] = {}
    for preset in ("whoosh_transition", "sub_impact", "riser_tension", "ui_pop", "vine_boom"):
        p = effects_dir / f"{preset}.wav"
        if not p.exists():
            synthesize_sfx(preset, p)
        sfx_map[preset] = str(p.resolve())
    bed_path = effects_dir / "ambient_bed.wav"
    if not bed_path.exists():
        synthesize_ambient_bed(bed_path, duration_sec=24.0, mood="tech_focus")
    sfx_map["ambient_bed"] = str(bed_path.resolve())
    if download_clicks:
        dl = download_click_sounds(effects_dir, allow_network=True)
        sfx_map.update(dl["click_paths"])
    else:
        for click_preset in DEFAULT_CLICK_SOUNDS:
            cp = effects_dir / f"{click_preset}.wav"
            if cp.is_file():
                sfx_map[click_preset] = str(cp.resolve())
    return sfx_map


def _build_atempo_chain(speed: float) -> str:
    """Decompose arbitrary playback_speed into chained FFmpeg atempo stages in [0.5, 2.0]."""
    spd = max(0.25, min(4.0, float(speed)))
    if abs(spd - 1.0) < 1e-3:
        return ""
    stages: list[str] = []
    cur = spd
    while cur > 2.0:
        stages.append("atempo=2.0")
        cur /= 2.0
    while cur < 0.5:
        stages.append("atempo=0.5")
        cur *= 2.0
    stages.append(f"atempo={cur:.4f}")
    return "," + ",".join(stages)


def _color_filter_for_style(style_profile: str) -> str:
    if "zoomer" in style_profile or "high_energy" in style_profile or "short_form" in style_profile or "comedy" in style_profile:
        return "eq=contrast=1.09:saturation=1.14:brightness=0.015,unsharp=3:3:0.35"
    if "cinematic" in style_profile or "vlog" in style_profile:
        return "eq=contrast=1.06:saturation=1.05,vignette=PI/7"
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

        # 1. Render each source segment with portrait-biased reframing, smooth zoom motion, speed, and color polish
        fps_int = int(round(timeline.fps))
        for idx, clip in enumerate(timeline.clips):
            clip_out = tmp / f"seg_{idx:03d}.mp4"
            dur = max(0.25, clip.source_end_sec - clip.source_start_sec)
            spd = max(0.25, min(4.0, float(clip.playback_speed or 1.0)))
            z = max(1.0, clip.zoom_scale)
            speed_vf = f"setpts=(PTS-STARTPTS)/{spd:.4f}," if abs(spd - 1.0) >= 1e-3 else ""
            trans_vf = ""
            if clip.transition_in and clip.transition_in.type != "cut" and idx > 0:
                td = min(0.35, max(0.12, float(clip.transition_in.duration_sec or 0.22)))
                trans_vf = f",fade=t=in:st=0:d={td:.2f}"

            push_fx = next((fx for fx in clip.effects if fx.preset_id == "smooth_push_in"), None)
            out_frames = max(2, int(round((dur / spd) * fps_int)))

            if is_vertical:
                base_crop = (
                    f"crop=min(iw\\,ih*9/16):ih:"
                    f"max(0\\,min(iw-min(iw\\,ih*9/16)\\,iw*{clip.crop_center_x:.4f}-min(iw\\,ih*9/16)/2)):0,"
                )
            else:
                base_crop = ""

            if push_fx is not None:
                z_delta = float((push_fx.parameters or {}).get("zoom_delta", 0.06))
                vf = (
                    f"{speed_vf}fps={fps_int},{base_crop}"
                    f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                    f"crop={w}:{h}:(iw-{w})/2:(ih-{h})*0.32,"
                    f"zoompan=z='{z:.3f}+{z_delta:.3f}*min(1\\,on/{out_frames})':"
                    f"x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*0.34':d=1:s={w}x{h}:fps={fps_int},"
                    f"setsar=1,{grade_vf}{trans_vf}"
                )
            else:
                vf = (
                    f"{speed_vf}{base_crop}"
                    f"scale={int(round(w * z))}:{int(round(h * z))}:force_original_aspect_ratio=increase,"
                    f"crop={w}:{h}:(iw-{w})/2:(ih-{h})*0.34,setsar=1,{grade_vf}{trans_vf}"
                )

            atempo_chain = _build_atempo_chain(spd)
            af = (
                f"highpass=f=80{atempo_chain},"
                "acompressor=threshold=-20dB:ratio=2.4:attack=10:release=120,"
                "loudnorm=I=-14:TP=-1.5:LRA=11"
            )
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{clip.source_start_sec:.3f}",
                "-t", f"{dur:.3f}",
                "-i", clip.source_path,
                "-vf", vf,
                "-af", af,
                "-r", str(fps_int),
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
            sfx_ext = ov_path.suffix.lower()
            is_static_img = sfx_ext in {".png", ".jpg", ".jpeg", ".webp"}
            is_loop_vid = sfx_ext in {".mp4", ".gif", ".webm"}
            if is_static_img:
                inputs.extend(["-loop", "1", "-t", f"{ov_dur + 0.15:.3f}", "-i", str(ov_path)])
            elif is_loop_vid:
                inputs.extend(["-stream_loop", "-1", "-t", f"{ov_dur + 0.15:.3f}", "-i", str(ov_path)])
            else:
                inputs.extend(["-i", str(ov_path)])

            next_v = f"[v_ov_{ov_i}]"
            if ov.mode == "cutaway_blur_fill":
                fg_w = int(round(w * (0.90 if is_vertical else 0.84)))
                fg_h = int(round(h * (0.62 if is_vertical else 0.78)))
                mbg_lbl = f"[ov_mbg_{ov_i}]"
                mfg_lbl = f"[ov_mfg_{ov_i}]"
                bg_lbl = f"[ov_bg_{ov_i}]"
                fg_lbl = f"[ov_fg_{ov_i}]"
                cut_lbl = f"[ov_cut_{ov_i}]"
                filter_parts.append(
                    f"[{input_idx}:v]fps={fps_int},setpts=PTS-STARTPTS,split=2{mbg_lbl}{mfg_lbl}"
                )
                filter_parts.append(
                    f"{mbg_lbl}scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
                    f"gblur=sigma=28,eq=brightness=-0.22:saturation=1.15{bg_lbl}"
                )
                filter_parts.append(
                    f"{mfg_lbl}scale={fg_w}:{fg_h}:force_original_aspect_ratio=decrease,"
                    f"scale=trunc(iw/2)*2:trunc(ih/2)*2,"
                    f"pad=iw+12:ih+12:6:6:color=black@0.88{fg_lbl}"
                )
                filter_parts.append(
                    f"{bg_lbl}{fg_lbl}overlay=(W-w)/2:(H-h)/2,"
                    f"setpts=PTS-STARTPTS+{ov.timeline_start_sec:.3f}/TB,"
                    f"format=yuva420p{cut_lbl}"
                )
                filter_parts.append(
                    f"{v_label}{cut_lbl}overlay=x=0:y=0:eof_action=pass:"
                    f"enable='between(t,{ov.timeline_start_sec:.3f},{ov.timeline_end_sec:.3f})'{next_v}"
                )
            elif ov.mode == "corner_sticker_clean":
                ov_w = max(160, int(round(w * ov.w_norm)))
                ov_h = max(140, int(round(h * ov.h_norm)))
                ov_x = int(round(w * ov.x_norm))
                ov_y = int(round(h * ov.y_norm))
                scaled = f"[ov_s_{ov_i}]"
                filter_parts.append(
                    f"[{input_idx}:v]fps={fps_int},setpts=PTS-STARTPTS+{ov.timeline_start_sec:.3f}/TB,"
                    f"scale={ov_w}:{ov_h}:force_original_aspect_ratio=decrease,"
                    f"scale=trunc(iw/2)*2:trunc(ih/2)*2,"
                    f"pad=iw+8:ih+8:4:4:color=white@0.88,format=yuva420p{scaled}"
                )
                filter_parts.append(
                    f"{v_label}{scaled}overlay=x={ov_x}:y={ov_y}:eof_action=pass:"
                    f"enable='between(t,{ov.timeline_start_sec:.3f},{ov.timeline_end_sec:.3f})'{next_v}"
                )
            else:
                ov_w = max(120, int(round(w * ov.w_norm)))
                ov_h = max(120, int(round(h * ov.h_norm)))
                ov_x = int(round(w * ov.x_norm))
                ov_y = int(round(h * ov.y_norm))
                scaled = f"[ov_s_{ov_i}]"
                filter_parts.append(
                    f"[{input_idx}:v]setpts=PTS-STARTPTS+{ov.timeline_start_sec:.3f}/TB,"
                    f"scale={ov_w}:{ov_h}:force_original_aspect_ratio=decrease,"
                    f"format=yuva420p{scaled}"
                )
                filter_parts.append(
                    f"{v_label}{scaled}overlay=x={ov_x}:y={ov_y}:eof_action=pass:"
                    f"enable='between(t,{ov.timeline_start_sec:.3f},{ov.timeline_end_sec:.3f})'{next_v}"
                )
            v_label = next_v
            input_idx += 1

        if timeline.caption_settings.enabled and timeline.captions:
            next_v = "[v_cap]"
            filter_parts.append(f"{v_label}ass='{_escape_ffmpeg_path(ass_path)}'{next_v}")
            v_label = next_v

        # Mix SFX + optional sidechain-ducked ambient music beds without attenuating speech
        sfx_inputs_labels: list[str] = []
        music_events = [au for au in timeline.audio_events if au.role == "music" and Path(au.source_path).exists()]
        sfx_events = [au for au in timeline.audio_events if au.role != "music" and Path(au.source_path).exists()]
        has_ducked_music = any(au.duck_under_speech for au in music_events)

        if has_ducked_music:
            filter_parts.append(f"{a_label}asplit=2[a_main][a_sc]")
            speech_mix_label = "[a_main]"
        else:
            speech_mix_label = a_label

        ducked_total = sum(1 for au in music_events if au.duck_under_speech)
        ducked_seen = 0
        for m_i, au in enumerate(music_events):
            au_path = Path(au.source_path)
            inputs.extend(["-stream_loop", "-1", "-i", str(au_path)])
            gain_mult = round(10.0 ** (au.gain_db / 20.0), 3)
            m_pre = f"[mus_pre_{m_i}]"
            m_lbl = f"[mus_{m_i}]"
            filter_parts.append(
                f"[{input_idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume={gain_mult}{m_pre}"
            )
            if au.duck_under_speech:
                ducked_seen += 1
                if ducked_seen >= ducked_total:
                    sc_in = "[a_sc]"
                else:
                    filter_parts.append(f"[a_sc]asplit=2[a_sc_{m_i}][a_sc]")
                    sc_in = f"[a_sc_{m_i}]"
                filter_parts.append(
                    f"{m_pre}{sc_in}sidechaincompress=threshold=0.025:ratio=6:attack=35:release=320:level_in=1{m_lbl}"
                )
            else:
                filter_parts.append(f"{m_pre}anull{m_lbl}")
            sfx_inputs_labels.append(m_lbl)
            input_idx += 1

        for s_i, au in enumerate(sfx_events):
            au_path = Path(au.source_path)
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
            all_a = speech_mix_label + "".join(sfx_inputs_labels)
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
        best_frame: Path | None = None
        best_score = -1.0
        try:
            import cv2  # noqa: PLC0415
        except Exception:
            cv2 = None

        candidates = [max(0.0, capture_sec + off) for off in (0.0, -0.9, 1.3, 2.5)]
        for idx, ts in enumerate(candidates):
            cand = Path(tmpdir) / f"frame_{idx}.png"
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{ts:.3f}",
                    "-i", str(source_video),
                    "-vframes", "1",
                    str(cand),
                ],
                check=(idx == 0),
                capture_output=True,
            )
            if proc.returncode != 0 or not cand.is_file() or cand.stat().st_size == 0:
                continue
            score = float(cand.stat().st_size)
            if cv2 is not None:
                img_cv = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
                if img_cv is not None:
                    score = float(cv2.Laplacian(img_cv, cv2.CV_64F).var())
            if best_frame is None or score > best_score:
                best_frame = cand
                best_score = score
            if cv2 is None:
                break
        if best_frame is None:
            raise RuntimeError(f"Failed to extract thumbnail frame from {source_video}")
        base = Image.open(best_frame).convert("RGBA").resize((1280, 720), Image.Resampling.LANCZOS)

    # Dark left gradient scrim kept compact so subject & right-hand board stay crisp
    overlay = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(520):
        alpha = int(215 * max(0.0, 1.0 - x / 520.0))
        draw.line([(x, 0), (x, 720)], fill=(11, 16, 32, alpha))

    combined = Image.alpha_composite(base, overlay)
    d = ImageDraw.Draw(combined)

    title_font = None
    badge_font = None
    for font_candidate in (
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ):
        if Path(font_candidate).is_file():
            try:
                title_font = ImageFont.truetype(font_candidate, 46)
                badge_font = ImageFont.truetype(font_candidate, 21)
                break
            except Exception:
                continue
    if title_font is None or badge_font is None:
        title_font = ImageFont.load_default()
        badge_font = ImageFont.load_default()

    # Badge pill dynamically sized to fit inside safe left margin (<= 330px)
    badge_txt = badge[:24].strip().upper()
    bbox = d.textbbox((0, 0), badge_txt, font=badge_font)
    badge_w = min(335, max(160, (bbox[2] - bbox[0]) + 36))
    d.rounded_rectangle([48, 68, 48 + badge_w, 114], radius=22, fill=(56, 189, 248, 240))
    d.text((66, 78), badge_txt, font=badge_font, fill=(11, 16, 32, 255))

    # Wrap headline cleanly within left third (<= 330px) to avoid overlapping off-center faces
    words = headline.strip().split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        cand_w = d.textbbox((0, 0), cand, font=title_font)[2]
        if cand_w <= 330 or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
        if len(lines) == 4:
            break
    if cur and len(lines) < 4:
        lines.append(cur)

    y = 152
    for ln in lines:
        d.text((51, y + 3), ln, font=title_font, fill=(0, 0, 0, 225))
        d.text((48, y), ln, font=title_font, fill=(248, 250, 252, 255))
        y += 62

    # Accent bottom stripe within left third
    d.rectangle([48, 636, 360, 648], fill=(245, 158, 11, 255))
    combined.convert("RGB").save(out_png, format="PNG")
    return out_png
