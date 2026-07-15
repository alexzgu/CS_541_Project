"""Condense karaoke color-fade styling repeats (S12 follow-up, song 88).

Some source subtitles animate a character's color by re-emitting the SAME
token in many contiguous rows of one animation frame each (song 88: uniform
~67 ms rows = 15 fps, runs up to 89 rows ≈ a 6 s sustained note). Those rows
are presentation frames, not morae. This pass merges each such run into one
row spanning it.

A run is merged iff its rows are contiguous (gap ≤ 20 ms), identical in token,
and SHORT (median row duration ≤ 80 ms) — real repeated morae (いい, ままま)
are sung at 100 ms+ per mora (measured corpus-wide: genuine doubles run
0.1–1.0 s, non-uniform), so duration, not run length, is the discriminator.
A song is touched at all only if it shows unmistakable styling (some run of
≥ 4 such frames); everywhere else the labels are left byte-identical.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

MAX_ROW_DUR = 0.080   # styling frames are <= ~67 ms; real morae are >= ~100 ms
MAX_GAP = 0.020
MIN_STYLED_RUN = 4    # song-level trigger: at least one run this long


def _read(path: str | Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        return [dict(r) for r in rd], list(rd.fieldnames)


def _runs(rows: list[dict]):
    """Yield (i, j) row-index spans of contiguous identical lyric tokens (j incl.)."""
    i = 0
    while i < len(rows):
        j = i
        if rows[i]["token"] != "<silence>":
            while (j + 1 < len(rows)
                   and rows[j + 1]["token"] == rows[i]["token"]
                   and float(rows[j + 1]["start"]) - float(rows[j]["end"]) <= MAX_GAP):
                j += 1
        if j > i:
            yield i, j
        i = j + 1


def _is_styling(rows: list[dict], i: int, j: int) -> bool:
    durs = [float(r["end"]) - float(r["start"]) for r in rows[i: j + 1]]
    return float(np.median(durs)) <= MAX_ROW_DUR


def condense_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Merge styling runs; returns (new_rows, rows_removed)."""
    merged: list[dict] = []
    removed = 0
    skip_until = -1
    run_map = {i: j for i, j in _runs(rows) if _is_styling(rows, i, j)}
    for i, r in enumerate(rows):
        if i <= skip_until:
            continue
        j = run_map.get(i, i)
        if j > i:
            q = dict(r)
            q["end"] = rows[j]["end"]
            if "exclude" in q:  # a run is one sound: excluded iff every frame was
                q["exclude"] = str(all(rows[k].get("exclude", "False").strip().lower() == "true"
                                       for k in range(i, j + 1)))
            merged.append(q)
            removed += j - i
            skip_until = j
        else:
            merged.append(r)
    return merged, removed


def song_report(path: str | Path) -> dict:
    rows, _ = _read(path)
    styled = [(i, j) for i, j in _runs(rows) if _is_styling(rows, i, j)]
    return {"rows": len(rows),
            "styled_runs": len(styled),
            "max_run": max((j - i + 1 for i, j in styled), default=0),
            "rows_removed": sum(j - i for i, j in styled)}


def scan(ids: list[int]) -> list[int]:
    """Report every song; return the ones that qualify for condensing."""
    hit = []
    for sid in ids:
        p = Path(f"data/clean/subtitles/{sid}.csv")
        if not p.exists():
            continue
        rep = song_report(p)
        if rep["max_run"] >= MIN_STYLED_RUN:
            hit.append(sid)
            print(f"song {sid}: {rep['rows']} rows, {rep['styled_runs']} styling runs "
                  f"(longest {rep['max_run']}), would remove {rep['rows_removed']} rows")
    if not hit:
        print("no songs with styling repeats")
    return hit


def apply(ids: list[int]) -> None:
    backup = Path("data/clean/subtitles_pre_condense")
    backup.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        src = Path(f"data/clean/subtitles/{sid}.csv")
        rows, fields = _read(src)
        new, removed = condense_rows(rows)
        if not removed:
            print(f"song {sid}: nothing to condense")
            continue
        if not (backup / src.name).exists():
            shutil.copy2(src, backup / src.name)
        with open(src, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            wr.writeheader()
            wr.writerows(new)
        print(f"song {sid}: {len(rows)} -> {len(new)} rows ({removed} styling frames merged); "
              f"original in {backup}/")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[2:]] or list(range(93))
    if sys.argv[1] == "scan":
        scan(ids)
    elif sys.argv[1] == "apply":
        apply(ids if len(sys.argv) > 2 else scan(ids))
    else:
        raise SystemExit("usage: python -m kashi.data.condense scan|apply [ids...]")
