"""Dataset v2 Tier-1 normalization (S13): recover mechanically-excluded rows.

The importer could only map 1 row = 1 inventory token, so source rows holding
sokuon or multiple morae were excluded (6,131 excluded lyric rows corpus-wide;
`docs/dataset_v2_plan.md`). Two deterministic transforms fix most of them:

* **sokuon merge** — っ is a mora of closure belonging to the preceding
  syllable. A bare `っ` row extends the previous lyric row; a `っ` inside a
  chunk extends its host mora (`かっ` → one `か` row spanning both morae's
  time; `って` → previous row gains the っ share, `て` takes the rest).
* **chunk split** — rows whose text decomposes into inventory morae
  (`ない` → `な`+`い`, youon like `きゃ` staying single tokens) are split,
  dividing the span in proportion to mora count (っ counts as one mora of
  time on its host — Japanese is mora-timed).

Transformed rows become `exclude=False`; anything that does not decompose
(English fragments, `<gap>`) is copied through untouched for Tier 2. Output
goes to `data/clean_v2/subtitles/` — `data/clean` is never modified, and
`--set data.version=clean_v2` switches the whole pipeline/eval over.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from ..tokens import SILENCE, TOKENS

_INVENTORY = {t for t in TOKENS if t != SILENCE}
_TWO = sorted((t for t in _INVENTORY if len(t) == 2), key=len, reverse=True)
SOKUON = "っ"
MERGE_GAP_S = 0.05    # a bare っ row must touch the previous row to merge


def decompose(text: str) -> list[str] | None:
    """Greedy parse into inventory morae and っ; None if any residue is left."""
    units: list[str] = []
    i = 0
    while i < len(text):
        two = text[i:i + 2]
        if len(two) == 2 and two in _INVENTORY:
            units.append(two)
            i += 2
        elif text[i] in _INVENTORY or text[i] == SOKUON:
            units.append(text[i])
            i += 1
        else:
            return None
    return units


def _weights(units: list[str]) -> tuple[float, list[tuple[str, float]]]:
    """(leading っ share, [(mora, share), ...]) — each mora & each っ = 1 unit
    of time; a っ's unit goes to the mora before it."""
    lead = 0.0
    morae: list[tuple[str, float]] = []
    for u in units:
        if u == SOKUON:
            if morae:
                morae[-1] = (morae[-1][0], morae[-1][1] + 1.0)
            else:
                lead += 1.0
        else:
            morae.append((u, 1.0))
    return lead, morae


def normalize_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    stats = {"merged_sokuon": 0, "split_chunks": 0, "recovered_rows": 0, "unmapped": 0}
    out: list[dict] = []
    for r in rows:
        tok = r["token"]
        if tok == SILENCE or tok in _INVENTORY:
            out.append(dict(r))
            continue
        units = decompose(tok)
        if units is None:
            stats["unmapped"] += 1
            out.append(dict(r))
            continue
        was_excluded = r.get("exclude", "False").strip().lower() == "true"
        start, end = float(r["start"]), float(r["end"])
        lead, morae = _weights(units)
        total = lead + sum(w for _, w in morae)
        # leading っ extends the previous lyric row (never across silence/gaps)
        if lead:
            prev = out[-1] if out else None
            grow = (end - start) * (lead / total)
            if (prev is not None and prev["token"] in _INVENTORY
                    and start - float(prev["end"]) <= MERGE_GAP_S):
                # bare っ: absorb the whole row (any tiny gap IS the closure);
                # っ-leading chunk: take the っ share, hand off to the morae
                prev["end"] = str(round(end if not morae else start + grow, 3))
                stats["merged_sokuon"] += 1
            elif not morae:
                out.append(dict(r))   # nothing to attach to — keep as-is
                continue
            start += grow  # closure time belongs to the (merged) predecessor
        t = start
        span = end - start
        total_m = sum(w for _, w in morae)
        for k, (m, w) in enumerate(morae):
            q = dict(r)
            q["token"] = m
            q["start"] = str(round(t, 3))
            t = end if k == len(morae) - 1 else t + span * (w / total_m)
            q["end"] = str(round(t, 3))
            if "exclude" in q:
                q["exclude"] = "False"
            out.append(q)
            if was_excluded:
                stats["recovered_rows"] += 1
        if len(morae) > 1:
            stats["split_chunks"] += 1
        if was_excluded and not morae:
            stats["recovered_rows"] += 1  # bare っ absorbed into its predecessor
    return out, stats


def build(src_dir: str | Path = "data/clean/subtitles",
          dst_dir: str | Path = "data/clean_v2/subtitles") -> dict:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    tot = {"merged_sokuon": 0, "split_chunks": 0, "recovered_rows": 0, "unmapped": 0}
    files = sorted(src_dir.glob("*.csv"), key=lambda p: int(p.stem))
    for p in files:
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames)
            rows = [dict(r) for r in rd]
        new, st = normalize_rows(rows)
        for k in tot:
            tot[k] += st[k]
        with open(dst_dir / p.name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            wr.writeheader()
            wr.writerows(new)
    print(f"[normalize] {len(files)} songs -> {dst_dir}: "
          f"{tot['recovered_rows']} rows recovered "
          f"({tot['merged_sokuon']} sokuon merges, {tot['split_chunks']} chunk splits); "
          f"{tot['unmapped']} rows left for Tier 2 (English/<gap>/stylized)")
    return tot


if __name__ == "__main__":
    build(*sys.argv[1:])
