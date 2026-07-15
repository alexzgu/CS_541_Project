"""Timed-segment data model and subtitle writers (CSV / SRT / VTT / karaoke ASS).

A Segment is one token with absolute times in seconds — the same schema as the
dataset CSVs (start,end,token,exclude), with optional realignment metadata.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .tokens import SILENCE


@dataclass
class Segment:
    start: float
    end: float
    token: str
    exclude: bool = False
    confidence: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_silence(self) -> bool:
        return self.token == SILENCE


def read_csv(path: str | Path) -> list[Segment]:
    segs: list[Segment] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            extras = {
                k: v for k, v in row.items()
                if k not in ("start", "end", "token", "exclude", "confidence") and v not in ("", None)
            }
            conf = row.get("confidence")
            segs.append(
                Segment(
                    start=float(row["start"]),
                    end=float(row["end"]),
                    token=row["token"],
                    exclude=str(row.get("exclude", "False")).strip().lower() == "true",
                    confidence=float(conf) if conf not in (None, "") else None,
                    meta=extras,
                )
            )
    return segs


def write_csv(segments: list[Segment], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_keys: list[str] = []
    for s in segments:
        for k in s.meta:
            if k not in extra_keys:
                extra_keys.append(k)
    cols = ["start", "end", "token", "exclude"]
    if any(s.confidence is not None for s in segments):
        cols.append("confidence")
    cols += extra_keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in segments:
            row: list = [round(s.start, 3), round(s.end, 3), s.token, s.exclude]
            if "confidence" in cols:
                row.append("" if s.confidence is None else round(s.confidence, 4))
            row += [s.meta.get(k, "") for k in extra_keys]
            w.writerow(row)
    return path


# ---------------------------------------------------------------------------
# Line grouping for display formats
# ---------------------------------------------------------------------------

@dataclass
class Line:
    segments: list[Segment]

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def text(self) -> str:
        return "".join(s.token for s in self.segments)


def group_lines(
    segments: list[Segment],
    max_gap_s: float = 1.0,
    max_tokens: int = 22,
) -> list[Line]:
    """Group lyric tokens into display lines, breaking on silence gaps."""
    lines: list[Line] = []
    current: list[Segment] = []
    prev_end: float | None = None
    for s in segments:
        if s.is_silence or s.exclude or s.token == "":
            if current and (s.is_silence and s.duration >= max_gap_s):
                lines.append(Line(current))
                current = []
            prev_end = s.end
            continue
        if current and (
            len(current) >= max_tokens
            or (prev_end is not None and s.start - current[-1].end > max_gap_s)
        ):
            lines.append(Line(current))
            current = []
        current.append(s)
        prev_end = s.end
    if current:
        lines.append(Line(current))
    return lines


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _ts_srt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(t: float) -> str:
    return _ts_srt(t).replace(",", ".")


def to_srt(segments: list[Segment], romaji_line: bool = False) -> str:
    from .tokens import romaji as tok_romaji

    out: list[str] = []
    for i, line in enumerate(group_lines(segments), start=1):
        text = line.text
        if romaji_line:
            try:
                text += "\n" + " ".join(tok_romaji(s.token) for s in line.segments)
            except KeyError:
                pass
        out += [str(i), f"{_ts_srt(line.start)} --> {_ts_srt(line.end)}", text, ""]
    return "\n".join(out)


def to_vtt(segments: list[Segment], romaji_line: bool = False) -> str:
    from .tokens import romaji as tok_romaji

    out = ["WEBVTT", ""]
    for line in group_lines(segments):
        text = line.text
        if romaji_line:
            try:
                text += "\n" + " ".join(tok_romaji(s.token) for s in line.segments)
            except KeyError:
                pass
        out += [f"{_ts_vtt(line.start)} --> {_ts_vtt(line.end)}", text, ""]
    return "\n".join(out)


_ASS_HEADER = """[Script Info]
Title: kashi karaoke subtitles
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Noto Sans CJK JP,48,&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,30,30,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts_ass(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def to_ass(segments: list[Segment]) -> str:
    """ASS with per-syllable karaoke timing ({\\k} tags) — the native karaoke format."""
    events: list[str] = []
    for line in group_lines(segments):
        parts: list[str] = []
        t = line.start
        for s in line.segments:
            gap_cs = int(round((s.start - t) * 100))
            if gap_cs > 0:
                parts.append(f"{{\\k{gap_cs}}}")
            parts.append(f"{{\\k{max(1, int(round(s.duration * 100)))}}}{s.token}")
            t = s.end
        events.append(
            f"Dialogue: 0,{_ts_ass(line.start)},{_ts_ass(line.end)},Karaoke,,0,0,0,," + "".join(parts)
        )
    return _ASS_HEADER + "\n".join(events) + "\n"


WRITERS = {"csv": None, "srt": to_srt, "vtt": to_vtt, "ass": to_ass}


def write_outputs(
    segments: list[Segment],
    out_dir: str | Path,
    stem: str,
    formats: list[str],
    romaji_line: bool = False,
    display_lead_ms: float = 0.0,
) -> dict[str, Path]:
    """display_lead_ms shifts the DISPLAY formats (srt/vtt/ass) earlier so
    subtitles appear slightly before the sung onset — viewers perceive that as
    in-time (S12 feedback). The csv keeps the true model timings: it is the
    data/eval format and must stay honest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lead = max(0.0, float(display_lead_ms)) / 1000.0
    disp = segments
    if lead > 0:
        from dataclasses import replace

        disp = [replace(s, start=max(0.0, s.start - lead), end=max(0.0, s.end - lead))
                for s in segments]
    out: dict[str, Path] = {}
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "csv":
            write_csv(segments, path)
        elif fmt == "srt":
            path.write_text(to_srt(disp, romaji_line), encoding="utf-8")
        elif fmt == "vtt":
            path.write_text(to_vtt(disp, romaji_line), encoding="utf-8")
        elif fmt == "ass":
            path.write_text(to_ass(disp), encoding="utf-8")
        else:
            raise ValueError(f"unknown subtitle format: {fmt}")
        out[fmt] = path
    return out
