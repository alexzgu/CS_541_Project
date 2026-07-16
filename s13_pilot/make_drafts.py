"""Tier-2 pilot draft generator (S13). Songs 23 & 21 (train side).

Policy applied (docs/dataset_v2_plan.md Tier 2, + pilot findings):
- English is transcribed AS SUNG in standard Japanese loan pronunciation,
  constrained to the 110-mora inventory: missing extended morae take the
  standard two-mora expansion (うぃ→うい, てぃ→てい, うぉ→standalone わ/お
  judgment per word); ふぁ/ふぃ/ふぇ/ふぉ/でぃ exist and are used directly.
- Readings are authored per contiguous BLOCK (a lyric line); morae are
  distributed over the source fragments' time spans in proportion to each
  fragment's duration (Japanese singing is near-mora-timed).
- っ in a reading contributes no token (v2 sokuon policy); ー extends the
  previous vowel (kana_token_stream handles both).
- TEST songs are exempt from model-assisted drafting (contamination); song
  81's items are flagged in DRAFTS.md instead.

Output: s13_pilot/{sid}_draft.csv (full file, drafted rows replacing
the excluded originals, exclude=False) — data/clean_v2 is NOT touched.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kashi.stats.lm import kana_token_stream  # noqa: E402
from kashi.tokens import TOKENS  # noqa: E402

# (song, block start time rounded to 2dp) -> kana reading of the whole block.
READINGS: dict[tuple[int, float], str] = {
    # ---- song 23 ----
    (23, 2.02): "れっつしーざわーるどさーちんぐふぉーざりーずんわいあいあむまいせるふ",
    (23, 9.86): "ういずふぃああんどういずじょいあいぶねばーふぇるとびふぉーあいるしんぐあうととわーずざすかい",
    (23, 46.26): "あいくっどびーかいんどおあこーるどはーてっどばっとあいどんとのうういっちちょいすつめいくあいのうまいはーといずういっしんぐふぉーほーぷ",
    (23, 61.74): ("れっつしーざわーるどさーちんぐふぉーざりーずんわいあいあむまいせるふ"
                  "ういずふぃああんどういずじょいあいぶねばーふぇるとびふぉー"
                  "あいういるすたーとつわーくはうめにーしんぐすういるあいらーん"
                  "はうめにーたいむずういるあいせいぐっどばいおーばーざいやーず"
                  "おーわっとういるあうえいとあいきゃんのっとふぉーしー"
                  "あいぷれいいっとういるびーあほーぷざっといずねばーふぇいでぃんぐ"),
    (23, 94.75): "ばっとわっといずまいつるーほーぷ",
    (23, 125.28): "あいくっどびーかいんどおあこーるどはーてっどばっときゃんあいちゅーずざらいとあんさーあいのうまいはーといずういっしんぐふぉーほーぷ",
    (23, 142.59): "わい",
    (23, 145.6): "ふーあむ",
    (23, 148.56): ("いふあいわずわんおあずぃあざーいっとうっどびーそーいーじー"
                   "きゃんあいびりーぶまいせるふきゃんあいびりーぶまいつーはーつ"
                   "あいわんとつめいくざらいとちょいすふぉーまいせるふふぉーぴーぷるふぉーずぃすわーるどおーるおぶらいふ"),
    (23, 173.56): ("れっつしーざわーるどさーちんぐふぉーざりーずんわいあいあむまいせるふ"
                   "えんぶれいしんぐふぃあつげざーういずじょいらいくあいぶねばーのうん"
                   "あいういるきゃりーおんあいあむほーぷなうあいむしゅあ"
                   "あいむもあざんじゃすとあんえんじぇるおああでーもん"
                   "あいわずぼーんあずざおんりーほーぷつせいぶずぃすわーるどあわーるどそー"),
    (23, 200.38): "あわーるどそーらぶ",
    (23, 203.35): "おーるうぇいずびゅーてぃふるあんどあいういるのっとえばーぎぶいっとあっぷ",
    # ---- song 21 ----
    (21, 5.56): "しー", (21, 6.26): "しー", (21, 6.96): "しー",
    (21, 8.36): "わああ",
    (21, 10.59): "おーいえい",
    (21, 11.83): "しんしょっく", (21, 13.2): "しんしょっく",
    (21, 15.27): "はろーはろー",
    (21, 17.4): "しんしょっく", (21, 18.8): "しんしょっく",
    (21, 20.87): "すいんあんどすいーと",
    (21, 25.58): "みい",
    (21, 28.64): "さささささささあ",
    (21, 44.63): "しゅりんぷ",
    (21, 45.4): "さーもん",
    (21, 48.2): "いーとしゅりんぷういーいーとえぶりーしゅりんぷおんずぃあーすおーまいくすおん",
    (21, 57.61): "ぶるー",
    (21, 62.31): "ふむふむ",
    (21, 63.31): ("わんでぱーとざはーばーつーろーんちあわまりんじぇっと"
                  "すりーすたーとあしゃーくべんちゃーふぉーあんどたこおーばーざわーるどういずしー"),
    (21, 71.72): "はいはい",
    (21, 74.26): "らぶりーおーしゃん",
    (21, 82.87): "いえいいえいいえい",
    (21, 85.47): "えぴっくおーしゃん",
    (21, 89.34): "あくあ",
    (21, 90.71): "ぶるーえすとぷらねっと",
    (21, 98.22): "しー", (21, 98.92): "しー", (21, 99.62): "しー",
    (21, 101.02): "わああ",
    (21, 107.66): "ないすぼーと",
    (21, 111.76): "あー",
    (21, 121.74): "ざっつ",
    (21, 135.99): "はいはい",
    (21, 138.59): "らぶりーおーしゃん",
    (21, 147.13): "いえいいえいいえい",
    (21, 149.77): "えぴっくおーしゃん",
    (21, 153.6): "あくあ",
    (21, 154.97): "ぶるーえすとぷらねっと",
    (21, 162.51): "しー", (21, 163.21): "しー", (21, 163.88): "しー",
    (21, 165.28): "しー", (21, 165.98): "しー", (21, 166.68): "しー",
    (21, 168.42): "わあ",
}


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


def draft(sid: int) -> None:
    src = Path(f"data/clean_v2/subtitles/{sid}.csv")
    with open(src, newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames)
        rows = [dict(r) for r in rd]
    replaced = {}
    n_miss = 0
    for block in blocks_of(rows):
        t0 = round(float(rows[block[0]]["start"]), 2)
        key = (sid, t0)
        if key not in READINGS:
            n_miss += 1
            continue
        ids = [i for i in kana_token_stream(READINGS[key]) if i >= 0]
        spans = [(float(rows[i]["start"]), float(rows[i]["end"])) for i in block]
        total = sum(e - s for s, e in spans)
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
    dst = Path(f"s13_pilot/{sid}_draft.csv")
    with open(dst, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(out_rows)
    n_new = sum(len(v) for v in replaced.values())
    n_old = sum(len(b) for b in replaced)
    print(f"song {sid}: {len(replaced)} blocks drafted ({n_old} fragment rows -> "
          f"{n_new} mora rows); {n_miss} blocks without readings -> {dst}")


if __name__ == "__main__":
    for sid in (23, 21):
        draft(sid)
