"""Tier-2 BATCH 3 draft generator. Same policy/machinery as s13_pilot/make_drafts.py
(proportional mora distribution over the source fragments' spans; っ contributes no
token, ー extends the vowel via kana_token_stream), but READINGS come from
runs/s13_batch3_readings.py (2,422 blocks / 125 songs, subagent-drafted 2026-07-17,
383 tagged # LOW-CONF for ear-review).

Output: s13_batch3/{sid}_draft.csv — data/clean_v2 is NOT touched.

    .venv/bin/python s13_batch3/make_drafts.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kashi.stats.lm import kana_token_stream  # noqa: E402
from kashi.tokens import TOKENS  # noqa: E402

ns: dict = {}
exec(compile((ROOT / "s13_batch3/readings.py").read_text(),
             "readings.py", "exec"), ns)
READINGS: dict[tuple[int, float], str] = ns["READINGS"]
OUT = ROOT / "s13_batch3"


def blocks_of(rows):
    out, cur = [], []
    for idx, r in enumerate(rows):
        exc = r.get("exclude", "").strip().lower() == "true"
        if exc and r["token"] not in ("<silence>", "<gap>"):
            cur.append(idx)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def draft(sid: int) -> tuple[int, int]:
    src = ROOT / f"data/clean_v2/subtitles/{sid}.csv"
    with open(src, newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames)
        rows = [dict(r) for r in rd]
    replaced, n_miss = {}, 0
    for block in blocks_of(rows):
        t0 = round(float(rows[block[0]]["start"]), 2)
        key = (sid, t0)
        if key not in READINGS:
            n_miss += 1
            continue
        ids = [i for i in kana_token_stream(READINGS[key]) if i >= 0]
        spans = [(float(rows[i]["start"]), float(rows[i]["end"])) for i in block]
        total = sum(e - s for s, e in spans) or 1e-6
        new_rows, k = [], 0
        for j, (s, e) in enumerate(spans):
            take = (len(ids) - k if j == len(spans) - 1
                    else max(0, round(len(ids) * (e - s) / total)))
            take = min(take, len(ids) - k)
            for m in range(take):
                a = s + (e - s) * m / max(1, take)
                b = s + (e - s) * (m + 1) / max(1, take)
                q = dict(rows[block[0]])
                q.update(start=str(round(a, 3)), end=str(round(b, 3)),
                         token=TOKENS[ids[k]], exclude="False")
                new_rows.append(q)
                k += 1
        replaced[tuple(block)] = new_rows
    out_rows, skip = [], set()
    for idx, r in enumerate(rows):
        if idx in skip:
            continue
        hit = next((b for b in replaced if b[0] == idx), None)
        if hit:
            out_rows.extend(replaced[hit])
            skip.update(hit)
        else:
            out_rows.append(r)
    with open(OUT / f"{sid}_draft.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(out_rows)
    n_new = sum(len(v) for v in replaced.values())
    return len(replaced), n_new


if __name__ == "__main__":
    sids = sorted({k[0] for k in READINGS})
    tot_b = tot_m = 0
    for sid in sids:
        nb, nm = draft(sid)
        tot_b += nb
        tot_m += nm
        print(f"song {sid}: {nb} blocks -> {nm} mora rows")
    print(f"TOTAL: {len(sids)} songs, {tot_b} blocks, {tot_m} drafted mora rows")
