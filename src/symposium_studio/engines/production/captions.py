"""SRT and ASS kinetic/safe-zone subtitle generation from RenderTimeline."""

from __future__ import annotations

from pathlib import Path

from symposium_studio.models.timeline import CaptionCue, RenderTimeline, TargetPlatformFormat


def _fmt_srt_time(sec: float) -> str:
    ms_total = max(0, int(round(sec * 1000)))
    h = ms_total // 3_600_000
    m = (ms_total % 3_600_000) // 60_000
    s = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ass_time(sec: float) -> str:
    cs_total = max(0, int(round(sec * 100)))
    h = cs_total // 360_000
    m = (cs_total % 360_000) // 6000
    s = (cs_total % 6000) // 100
    cs = cs_total % 100
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def write_srt(timeline: RenderTimeline, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for idx, cue in enumerate(timeline.captions, start=1):
        txt = cue.text.strip()
        if cue.low_confidence:
            txt = f"[uncertain] {txt}"
        lines.append(f"{idx}\n{_fmt_srt_time(cue.start_sec)} --> {_fmt_srt_time(cue.end_sec)}\n{txt}\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _format_karaoke_line(cue: CaptionCue, max_words: int, highlight_word: bool) -> str:
    words = cue.words
    if not words or not highlight_word:
        parts = cue.text.strip().split()
        if len(parts) > max_words:
            mid = len(parts) // 2
            return _escape_ass(" ".join(parts[:mid])) + "\\N" + _escape_ass(" ".join(parts[mid:]))
        return _escape_ass(cue.text.strip())

    tokens: list[str] = []
    for i, w in enumerate(words):
        dur_cs = max(8, int(round((float(w.get("end_sec", 0.2)) - float(w.get("start_sec", 0.0))) * 100)))
        word_txt = _escape_ass(str(w.get("word", "")).strip())
        if not word_txt:
            continue
        if i > 0 and i % max_words == 0:
            tokens.append("\\N")
        tokens.append(f"{{\\kf{dur_cs}}}{word_txt}")
    return " ".join(tokens).replace(" \\N ", "\\N")


def write_ass(timeline: RenderTimeline, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cs = timeline.caption_settings
    font_size = max(24, int(round(cs.font_size * cs.scale_factor)))
    is_vertical = timeline.platform_format != TargetPlatformFormat.YOUTUBE_LONG or timeline.height > timeline.width
    # Respect platform safe zones (Shorts/TikTok/Reels put buttons/captions at bottom 220-280px)
    margin_v = max(cs.safe_margin_bottom_px, 240 if is_vertical else 75)
    margin_lr = 70 if is_vertical else 110

    # PrimaryColour = Yellow/Amber highlight for karaoke, SecondaryColour = crisp White before sweep
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {timeline.width}
PlayResY: {timeline.height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cs.font_name},{font_size},&H001AF0FF,&H00FFFFFF,&H00120C06,&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.5,2,{margin_lr},{margin_lr},{margin_v},1
Style: Uncertain,{cs.font_name},{max(22, font_size - 6)},&H0088D8FF,&H00E2E8F0,&H00120C06,&H80000000,0,1,0,0,100,100,0,0,1,2.5,1,2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    for cue in timeline.captions:
        style_name = "Uncertain" if cue.low_confidence else "Default"
        body = _format_karaoke_line(cue, cs.max_words_per_line, cs.highlight_active_word)
        if cue.low_confidence:
            body = f"[?] {body}"
        events.append(
            f"Dialogue: 0,{_fmt_ass_time(cue.start_sec)},{_fmt_ass_time(cue.end_sec)},{style_name},,0,0,0,,{body}"
        )

    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_path
