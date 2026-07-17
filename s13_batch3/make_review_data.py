"""Build the batch-3 review player's data.js + media symlinks.

Same three lanes as the pilot (SOURCE = excluded rows as written, CURRENT =
what training sees today, DRAFT = proposal). 125 songs, sorted by drafted-mora
volume descending so review effort lands on the biggest wins first. Songs with
LOW-CONF blocks get a warning count in the title. Media: ids 0-92 use the
dataset repo videos; ids 93+ (T1 admission) have audio-only webm — the player's
<video> element plays them with a black canvas, review is by ear.

    .venv/bin/python s13_batch3/make_review_data.py
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "s13_batch3"
DATASET_REPO = ROOT.parent / "karaoke_subtitle_dataset"


def titles() -> dict[int, str]:
    out = {}
    with open(DATASET_REPO / "data" / "indexed" / "index.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[int(r["Index"])] = r["Title"]
    with open(ROOT / "data" / "imported" / "admitted_index.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[int(r["id"])] = r["title"]
    return out


def admitted_yt() -> dict[int, str]:
    with open(ROOT / "data" / "imported" / "admitted_index.tsv", newline="") as f:
        return {int(r["id"]): r["yt"] for r in csv.DictReader(f, delimiter="\t")}


def lowconf_counts() -> dict[int, int]:
    counts: dict[int, int] = {}
    pat = re.compile(r"\(\s*(\d+)\s*,\s*[\d.]+\s*\)\s*:")
    for line in (ROOT / "s13_batch3" / "readings.py").read_text().splitlines():
        if "LOW-CONF" in line:
            m = pat.search(line)
            if m:
                sid = int(m.group(1))
                counts[sid] = counts.get(sid, 0) + 1
    return counts


def lyric(path: Path, want_excluded: bool) -> list:
    out = []
    for r in csv.DictReader(open(path, newline="")):
        if r["token"] == "<silence>":
            continue
        exc = r.get("exclude", "False").strip().lower() == "true"
        if exc == want_excluded:
            out.append((round(float(r["start"]), 3), round(float(r["end"]), 3), r["token"]))
    return out


def main() -> None:
    (OUT / "media").mkdir(exist_ok=True)
    ttl, yt, lc = titles(), admitted_yt(), lowconf_counts()
    sids = sorted(int(p.stem.split("_")[0]) for p in OUT.glob("*_draft.csv"))
    songs = []
    for sid in sids:
        cur = ROOT / f"data/clean_v2/subtitles/{sid}.csv"
        drf = OUT / f"{sid}_draft.csv"
        candidates = [DATASET_REPO / "data" / "indexed" / "videos" / f"{sid}.webm"] if sid <= 92 else [
            ROOT / "data" / "imported" / "audio" / f"{yt[sid]}.webm",
            ROOT / "data" / "raw" / "audio" / f"{sid}.mp3",   # admission-time mp3 (pool-sourced audio)
        ]
        video_src = next((p for p in candidates if p.is_file()), None)
        vocals_src = ROOT / "data" / "clean" / "audio" / "vocals" / f"{sid}.mp3"
        if video_src is None or not vocals_src.is_file():
            print(f"song {sid}: SKIPPED (missing media)")
            continue
        for link, target in ((OUT / "media" / f"{sid}{video_src.suffix}", video_src),
                             (OUT / "media" / f"{sid}_vocals.mp3", vocals_src)):
            link.unlink(missing_ok=True)
            link.symlink_to(target)
        n_src = len(lyric(cur, True))
        n_new = len(lyric(drf, False)) - len(lyric(cur, False))
        title = ttl.get(sid, f"song {sid}")
        if lc.get(sid):
            title += f"  [!{lc[sid]} low-conf]"
        songs.append({
            "id": sid, "title": title,
            "source": lyric(cur, True), "current": lyric(cur, False),
            "draft": lyric(drf, False), "n_src": n_src, "n_new": n_new,
            "video": f"media/{sid}{video_src.suffix}", "vocals": f"media/{sid}_vocals.mp3",
        })
    songs.sort(key=lambda s: -s["n_new"])
    (OUT / "data.js").write_text("const SONGS = " + json.dumps(songs) + ";\n")
    total = sum(s["n_new"] for s in songs)
    nlc = sum(1 for s in songs if "low-conf" in s["title"])
    print(f"wrote {OUT}/data.js: {len(songs)} songs, +{total} drafted morae, "
          f"{nlc} songs carry low-conf flags")


if __name__ == "__main__":
    main()
