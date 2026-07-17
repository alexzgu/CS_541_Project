"""S17 clean_v3 builder: phonetic (as-sung) label space + T1 structural repairs.

    .venv/bin/python -m kashi.data.phonetic            # dry run (counts only)
    .venv/bin/python -m kashi.data.phonetic --apply    # write data/clean_v3 + gold_v3

Per approved plan (docs/s17_phonetic_plan.md, SIGNOFFS S17a/b):
1. Worklist relabels (runs/s17_relabel_worklist.csv, flanked=True only): the
   ro-verified instance-level は→わ / へ→え (plus を/づ rows that the blanket
   would catch anyway). Test songs 81/83/85 included (S17b Option A); 89-92
   have no worklist rows.
2. Blanket relabels everywhere (incl. non-ro songs): を→お, づ→ず, ぢ→じ.
3. T1 repairs BEFORE relabeling: non-inventory exclude=False tokens (bare ー,
   ー-composites, bare っ, small kana) are decomposed/merged/folded; unfixable
   rows flip to exclude=True. Lyric-lyric overlaps truncated at the overlap
   midpoint. Zero-duration <silence> rows dropped.
4. gold_v3: blanket + worklist rows matched by |Δstart| <= 0.15 s (gold times
   descend from corrected v1, not identical to clean_v2).

Every change is logged to data/clean_v3/CHANGES.tsv (song, start, kind,
before, after). clean_v2 is never touched.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from ..tokens import TOKENS

TOKSET = set(TOKENS)
BLANKET = {"を": "お", "づ": "ず", "ぢ": "じ"}
SMALL = dict(zip("ぁぃぅぇぉゃゅょゎ", "あいうえおやゆよわ"))
VOWEL_OF = {}
for t in TOKENS:
    if t in ("<silence>",):
        continue
    last = t[-1]
    base = SMALL.get(last, last)
    row = "あかがさざただなはばぱまやらわ"
    for v, cls in zip("あいうえお", ("あぁかがさざただなはばぱまやらわ",
                                     "いぃきぎしじちぢにひびぴみり",
                                     "うぅくぐすずつづぬふぶぷむゆる",
                                     "えぇけげせぜてでねへべぺめれ",
                                     "おぉこごそぞとどのほぼぽもよろ")):
        if base in cls:
            VOWEL_OF[t] = {"あ": "あ", "い": "い", "う": "う", "え": "え", "お": "お"}[v]
            break
    if t == "ん":
        VOWEL_OF[t] = "ん"


def decompose(tok: str, prev_tok: str | None) -> list[str] | None:
    """Split a possibly-composite kana token into inventory morae.
    ー extends the previous vowel; っ contributes nothing; small kana fold.
    Returns None if any char can't be consumed."""
    out: list[str] = []
    i = 0
    s = "".join(SMALL.get(c, c) if c in "ぁぃぅぇぉ" else c for c in tok)
    while i < len(s):
        c = s[i]
        if c == "ー":
            ref = out[-1] if out else prev_tok
            v = VOWEL_OF.get(ref or "", None)
            if v is None:
                return None
            out.append(v)
            i += 1
            continue
        if c == "っ":
            i += 1  # sokuon folds into host (contributes no token)
            continue
        if i + 1 < len(s) and s[i:i + 2] in TOKSET:  # youon pairs like きゃ
            out.append(s[i:i + 2])
            i += 2
            continue
        if c in TOKSET:
            out.append(c)
            i += 1
            continue
        if c in SMALL and SMALL[c] in TOKSET:  # ゃゅょゎ standalone
            out.append(SMALL[c])
            i += 1
            continue
        return None
    return out


def load(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames), [dict(r) for r in rd]


def repair_rows(rows: list[dict], log) -> list[dict]:
    """T1 structural repairs (steps 3): returns a new row list."""
    out: list[dict] = []
    for r in rows:
        tok = r["token"]
        exc = r["exclude"].strip().lower() == "true"
        if tok == "<silence>" and float(r["end"]) - float(r["start"]) <= 0:
            log("drop-zero-silence", r, "")
            continue
        if exc or tok in TOKSET or tok.startswith("<"):  # tags (<silence>/<gap>/<noise>) pass through
            out.append(r)
            continue
        prev_tok = next((q["token"] for q in reversed(out)
                         if q["exclude"].strip().lower() != "true"
                         and q["token"] in TOKSET), None)
        morae = decompose(tok, prev_tok)
        if morae is None:
            q = dict(r)
            q["exclude"] = "True"
            log("unfixable->exclude", r, tok)
            out.append(q)
        elif not morae:  # pure っ/ー-less residue: fold into previous row
            if out and out[-1]["token"] in TOKSET:
                out[-1]["end"] = r["end"]
                log("merge-into-prev", r, out[-1]["token"])
            else:
                q = dict(r)
                q["exclude"] = "True"
                log("unfixable->exclude", r, tok)
                out.append(q)
        else:
            s, e = float(r["start"]), float(r["end"])
            step = (e - s) / len(morae)
            for j, m in enumerate(morae):
                q = dict(r)
                q.update(start=str(round(s + j * step, 3)),
                         end=str(round(s + (j + 1) * step, 3)), token=m)
                out.append(q)
            log("decompose", r, "+".join(morae))
    # overlap midpoint truncation: DIRECTLY adjacent lyric pairs only. Pairs
    # separated by excluded rows are source-inherited simultaneity (duets) —
    # midpointing them would corrupt real timings (audit counted 249 adjacent).
    lyr = [i for i, r in enumerate(out)
           if r["exclude"].strip().lower() != "true" and r["token"] in TOKSET]
    for a, b in zip(lyr, lyr[1:]):
        if b != a + 1:
            continue
        ea, sb = float(out[a]["end"]), float(out[b]["start"])
        if ea > sb:
            mid = round((ea + sb) / 2, 3)
            log("overlap-midpoint", out[a], f"{ea}->{mid}")
            out[a]["end"] = str(mid)
            out[b]["start"] = str(mid)
    return out


def relabel(rows: list[dict], wl: dict, sid: int, log, tol: float = 0.0) -> None:
    """Worklist relabels (exact or tolerance start match), then blanket rules."""
    used = set()
    for r in rows:
        if r["exclude"].strip().lower() == "true" or r["token"] not in TOKSET:
            continue
        st = round(float(r["start"]), 3)
        hit = None
        if tol == 0.0:
            hit = wl.get((sid, st))
        else:
            for (s2, t2), v in wl.items():
                if s2 == sid and abs(t2 - st) <= tol and (s2, t2) not in used:
                    hit = v
                    break
        if hit and r["token"] == hit[0]:
            used.add((sid, st if tol == 0.0 else t2))
            log("worklist", r, hit[1])
            r["token"] = hit[1]
    for r in rows:
        if r["exclude"].strip().lower() == "true":
            continue
        if r["token"] in BLANKET:
            log("blanket", r, BLANKET[r["token"]])
            r["token"] = BLANKET[r["token"]]


def main() -> None:
    apply = "--apply" in sys.argv
    root = Path(__file__).resolve().parents[3]
    wl: dict = {}
    with open(root / "runs/s17_relabel_worklist.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["flanked"].strip().lower() == "true":
                wl[(int(r["song_id"]), round(float(r["start"]), 3))] = (
                    r["label_token"], r["phonetic_token"])
    counts: dict[str, int] = {}
    changes: list[list] = []

    def logger(sid):
        def log(kind, row, after):
            counts[kind] = counts.get(kind, 0) + 1
            changes.append([sid, row["start"], kind, row["token"], after])
        return log

    out_sub = root / "data/clean_v3/subtitles"
    out_gold = root / "data/gold_v3/subtitles"
    # gold gets RELABELS ONLY (approved plan scope): its timings are the S12
    # calibrator and its <noise> tags are realign annotations — no repairs.
    for src_dir, out_dir, tol, repairs in (
            (root / "data/clean_v2/subtitles", out_sub, 0.0, True),
            (root / "data/gold_v2/subtitles", out_gold, 0.15, False)):
        if apply:
            out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_dir.glob("*.csv"), key=lambda q: int(q.stem)):
            sid = int(p.stem)
            fields, rows = load(p)
            log = logger(sid)
            if repairs:
                rows = repair_rows(rows, log)
            relabel(rows, wl, sid, log, tol=tol)
            if apply:
                with open(out_dir / p.name, "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
                    wr.writeheader()
                    wr.writerows(rows)
    if apply:
        with open(root / "data/clean_v3/CHANGES.tsv", "w", newline="") as f:
            wr = csv.writer(f, delimiter="\t", lineterminator="\n")
            wr.writerow(["song", "start", "kind", "before", "after"])
            wr.writerows(changes)
    mode = "APPLIED" if apply else "DRY RUN"
    print(f"[phonetic] {mode}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"[phonetic] worklist entries loaded: {len(wl)}")


if __name__ == "__main__":
    main()
