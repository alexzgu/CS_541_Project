"""Gold subset: hand-verified reference labels + the Audacity round-trip.

Layout:
  data/gold/subtitles/<id>.csv   verified rows (start,end,token,exclude)
  data/gold/windows.csv          song_id,start,end,source  (verified intervals)

Only frames inside a song's windows count as gold; metrics clip to them.
`seed` imports the legacy hand-corrected golden CSVs (songs 0/6/16/19):
$breathing/$echo -> <noise>, tokens outside the 110-inventory kept but
exclude=True (real vocals outside the token set, e.g. English).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from ..subtitles import Segment, read_csv, write_csv
from ..tokens import NOISE, SILENCE, TOKENS

GOLDEN_SRC = "data/gold/source/golden_csvs/processed"
GOLDEN_IDS = (0, 6, 16, 19)
_NOISE_ALIASES = {"$breathing", "$echo", "<bre>", "<noise>"}
_SIL_ALIASES = {"<silence>", "<sil>", "<s></s>"}
_TOKEN_SET = set(TOKENS)


def gold_dir(cfg) -> Path:
    return cfg.data_dir / "gold" / "subtitles"


def windows_file(cfg) -> Path:
    return cfg.data_dir / "gold" / "windows.csv"


def read_windows(cfg) -> list[dict]:
    f = windows_file(cfg)
    if not f.is_file():
        return []
    with open(f, newline="", encoding="utf-8") as fh:
        return [
            {"song_id": int(r["song_id"]), "start": float(r["start"]),
             "end": float(r["end"]), "source": r.get("source", "")}
            for r in csv.DictReader(fh)
        ]


def add_window(cfg, song_id: int, start: float, end: float, source: str) -> None:
    f = windows_file(cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    rows = [w for w in read_windows(cfg)
            if not (w["song_id"] == song_id and w["start"] == start and w["end"] == end)]
    rows.append({"song_id": song_id, "start": start, "end": end, "source": source})
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["song_id", "start", "end", "source"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["song_id"], r["start"])))


def gold_ids(cfg) -> list[int]:
    d = gold_dir(cfg)
    return sorted(int(p.stem) for p in d.glob("*.csv")) if d.is_dir() else []


def _normalise_token(tok: str) -> tuple[str, bool]:
    """raw golden token -> (token, exclude)."""
    tok = str(tok).strip()
    if tok in _SIL_ALIASES:
        return SILENCE, False
    if tok in _NOISE_ALIASES:
        return NOISE, False
    if tok in _TOKEN_SET:
        return tok, False
    return tok, True  # real vocal content outside the inventory (e.g. English)


def seed_golden(cfg) -> list[int]:
    """Import the legacy hand-corrected golden CSVs as gold (full-song windows)."""
    src_dir = cfg.root / GOLDEN_SRC
    done = []
    for song_id in GOLDEN_IDS:
        src = src_dir / f"{song_id}.csv"
        if not src.is_file():
            print(f"[gold] golden source missing: {src}")
            continue
        df = pd.read_csv(src)
        segs: list[Segment] = []
        for _, r in df.iterrows():
            tok, excl = _normalise_token(r["token"])
            excl = excl or (str(r.get("exclude", "False")).strip().lower() == "true")
            segs.append(Segment(float(r["start"]), float(r["end"]), tok, exclude=excl))
        out = gold_dir(cfg) / f"{song_id}.csv"
        write_csv(segs, out)
        add_window(cfg, song_id, segs[0].start, segs[-1].end, source="legacy-hand")
        n_noise = sum(1 for s in segs if s.token == NOISE)
        print(f"[gold] song {song_id}: {len(segs)} rows, {n_noise} <noise> spans -> {out}")
        done.append(song_id)
    return done


# ---------------------------------------------------------------------------
# Audacity label-track round-trip
# ---------------------------------------------------------------------------

def export(cfg, song_id: int, window_s: float = 90.0, at: float | None = None,
           version: str | None = None) -> Path:
    """Write an Audacity label track prefilled from the current labels, for the
    densest-lyric window (or one starting at --at). Human corrects by ear, then
    `kashi gold import` reads it back."""
    from ..data import manifest

    src = manifest.subtitles_dir(cfg, version) / f"{song_id}.csv"
    segs = read_csv(src)
    if at is None:
        # densest lyric window: slide in 5 s steps
        end_t = max(s.end for s in segs)
        best, at = -1, 0.0
        t = 0.0
        while t + window_s <= end_t + 5:
            n = sum(1 for s in segs if not s.is_silence and t <= s.start < t + window_s)
            if n > best:
                best, at = n, t
            t += 5.0
    out = cfg.runs_dir / "gold" / f"{song_id}_{int(at)}s_labels.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in segs:
            if s.end <= at or s.start >= at + window_s:
                continue
            f.write(f"{max(s.start, at):.3f}\t{min(s.end, at + window_s):.3f}\t{s.token}\n")
    print(f"[gold] window {at:.0f}-{at + window_s:.0f}s of song {song_id} -> {out}")
    print("        correct it in Audacity (File > Import > Labels), then:")
    print(f"        kashi gold import {song_id} {out} --window-start {at:.0f} --window-end {at + window_s:.0f}")
    return out


def import_labels(cfg, song_id: int, path: str | Path,
                  window_start: float, window_end: float) -> Path:
    """Merge a corrected label track back into the gold CSV for its window."""
    new: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            tok, excl = _normalise_token(parts[2])
            new.append(Segment(float(parts[0]), float(parts[1]), tok, exclude=excl))
    new.sort(key=lambda s: s.start)
    out = gold_dir(cfg) / f"{song_id}.csv"
    existing = read_csv(out) if out.is_file() else []
    kept = [s for s in existing if s.end <= window_start or s.start >= window_end]
    merged = sorted(kept + new, key=lambda s: s.start)
    write_csv(merged, out)
    add_window(cfg, song_id, window_start, window_end, source="audacity")
    print(f"[gold] imported {len(new)} rows into song {song_id} "
          f"[{window_start:.0f}-{window_end:.0f}s] -> {out}")
    return out


def status(cfg) -> None:
    wins = read_windows(cfg)
    ids = gold_ids(cfg)
    total = sum(w["end"] - w["start"] for w in wins)
    print(f"gold subset: {len(ids)} songs, {len(wins)} windows, {total/60:.1f} min verified")
    for w in wins:
        print(f"  song {w['song_id']:>3}  {w['start']:8.1f}-{w['end']:8.1f}s  ({w['source']})")
