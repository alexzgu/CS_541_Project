"""Karaoke-VTT scraping and parsing (P2b).

`parse_vtt` is a byte-identical Python port of the Rust parser in
karaoke_subtitle_dataset/src/parse_vtt.rs (the Rust stays as the differential
-test oracle). Karaoke subs recolour the sung prefix each cue; the parser maps
each colour to a stable 1-based index by order of first appearance and rewrites
<c.colorXXX> tags as <N>, so a later diff of consecutive cues recovers the
newly-sung mora and its timestamp.

Quirks preserved on purpose (oracle parity): the cue section starts after a
line beginning with "##"; a cue is only flushed on a blank line (a file that
does not end with one silently drops its last cue); `</c>` closing tags are
retained; `"` and whitespace/zero-width/pipe characters are stripped.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_COLOR_DEF = re.compile(r"cue\((c\.[^\)]+)\)")
_COLOR_TAG = re.compile(r"<c[^>]+>")
_CLEAN = re.compile("[ ,\n\t​|\"]")


def build_color_map(vtt_text: str) -> dict[str, int]:
    color_map: dict[str, int] = {}
    for line in vtt_text.splitlines():
        for m in _COLOR_DEF.finditer(line):
            color_map.setdefault(m.group(1), len(color_map) + 1)
    return color_map


def _unformatted(text: str, color_map: dict[str, int]) -> str:
    cleaned = _CLEAN.sub("", text)
    return _COLOR_TAG.sub(lambda m: f"<{color_map.get(m.group(0)[1:-1], 0)}>", cleaned)


def _percentage(item: str) -> int | None:
    parts = item.split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1].strip().rstrip("%"))
    except ValueError:
        return None


def parse_vtt(vtt_path: str | Path) -> list[tuple[str, str, int, int, str]]:
    """VTT -> rows (start, end, position, line, text-with-<N>-colour-tags)."""
    text = Path(vtt_path).read_text(encoding="utf-8")
    color_map = build_color_map(text)
    rows: list[tuple[str, str, int, int, str]] = []
    start = end = ""
    position = line_pct = -1
    buf = ""
    in_cue = False
    for line in text.splitlines():
        if line.startswith("##"):
            in_cue = True
            continue
        if not in_cue:
            continue
        if not line.strip():
            if start:
                rows.append((start, end, position, line_pct, _unformatted(buf, color_map)))
                start = end = ""
                position = line_pct = -1
                buf = ""
        elif "-->" in line:
            parts = line.split()
            if len(parts) < 3:
                continue
            start, end = parts[0], parts[2]
            for item in parts[3:]:
                pct = _percentage(item)
                if pct is None:
                    continue
                if item.startswith("position:"):
                    position = pct
                elif item.startswith("line:"):
                    line_pct = pct
        else:
            buf += line.replace('"', "")
    return rows


def parse_vtt_to_csv(vtt_path: str | Path, out_csv: str | Path) -> Path:
    """Write the CSV exactly like the Rust (manual quoting, \\n line ends)."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["start,end,position,line,text"]
    for start, end, pos, lp, text in parse_vtt(vtt_path):
        lines.append(f'{start},{end},{pos},{lp},"{text}"')
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_csv


# ---------------------------------------------------------------------------
# yt-dlp wrapper (labels and/or audio); indexing mirrors index_data.rs
# ---------------------------------------------------------------------------

_YTDLP_NAME = re.compile(r"^(?P<title>.*) \[(?P<vid>[A-Za-z0-9_-]{11})\]\.(?P<lang>[a-z-]+)\.vtt$")


def scrape_playlist(cfg, playlist_url: str, lang: str = "ja", out_dir: str | Path | None = None,
                    audio_only: bool = False, labels_only: bool = True) -> Path:
    """Fetch karaoke subs (and optionally audio) for a playlist into a staging
    dir: <out>/raw/*.vtt + progress.txt. Parsing/import happens separately
    (`kashi dataset import --from-staging`)."""
    out_dir = Path(out_dir) if out_dir else cfg.data_dir / "staging"
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "--ignore-errors", "--continue", "--no-overwrites",
        "--download-archive", str(out_dir / "progress.txt"),
        "-o", str(raw / "%(title)s [%(id)s].%(ext)s"),
    ]
    if audio_only:
        cmd += ["-x", "--audio-format", "mp3"]
    else:
        cmd += ["--write-sub", "--sub-lang", lang]
        if labels_only:
            cmd += ["--skip-download"]
    subprocess.run(cmd + [playlist_url], check=False)
    return raw


def index_staging(staging_raw: Path, index_tsv: Path, vtt_out: Path, start_index: int = 0) -> int:
    """Assign integer indices to scraped VTTs and normalise names (port of
    index_data.rs). Appends to index_tsv; returns count indexed."""
    vtt_out.mkdir(parents=True, exist_ok=True)
    index_tsv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not index_tsv.is_file()
    n = 0
    with open(index_tsv, "a", encoding="utf-8") as f:
        if new_file:
            f.write("Index\tTitle\tID\tLanguage\n")
        for vtt in sorted(staging_raw.glob("*.vtt")):
            m = _YTDLP_NAME.match(vtt.name)
            if not m:
                print(f"[scrape] unrecognised name, skipped: {vtt.name}")
                continue
            idx = start_index + n
            f.write(f"{idx}\t{m['title']}\t{m['vid']}\t{m['lang']}\n")
            (vtt_out / f"{idx}.vtt").write_bytes(vtt.read_bytes())
            n += 1
    return n
