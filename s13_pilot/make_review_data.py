"""Build the S13 pilot review player's data.js + media symlinks.

    .venv/bin/python s13_pilot/make_review_data.py

Lanes per song: SOURCE (the excluded rows as-written: English fragments,
stylized kana), CURRENT (clean_v2 non-excluded tokens = what training/eval
see today), DRAFT (the proposed rows from {sid}_draft.csv).
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "s13_pilot"
DATASET_REPO = ROOT.parent / "karaoke_subtitle_dataset"
SONGS = [23, 21]


def titles() -> dict[int, str]:
    with open(DATASET_REPO / "data" / "indexed" / "index.tsv", newline="") as f:
        return {int(r["Index"]): r["Title"] for r in csv.DictReader(f, delimiter="\t")}


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
    ttl = titles()
    songs = []
    for sid in SONGS:
        cur = ROOT / f"data/clean_v2/subtitles/{sid}.csv"
        drf = OUT / f"{sid}_draft.csv"
        for link, target in ((OUT / "media" / f"{sid}.webm",
                              DATASET_REPO / "data" / "indexed" / "videos" / f"{sid}.webm"),
                             (OUT / "media" / f"{sid}_vocals.mp3",
                              ROOT / "data" / "clean" / "audio" / "vocals" / f"{sid}.mp3")):
            link.unlink(missing_ok=True)
            link.symlink_to(target)
        n_src = len(lyric(cur, True))
        n_draft_new = len(lyric(drf, False)) - len(lyric(cur, False))
        songs.append({
            "id": sid, "title": ttl.get(sid, f"song {sid}"),
            "source": lyric(cur, True),          # excluded rows as written
            "current": lyric(cur, False),        # what eval/training see today
            "draft": lyric(drf, False),          # proposal (excluded blocks filled)
            "n_src": n_src, "n_new": n_draft_new,
            "video": f"media/{sid}.webm", "vocals": f"media/{sid}_vocals.mp3",
        })
        print(f"song {sid}: {n_src} source fragments -> +{n_draft_new} drafted morae")
    (OUT / "data.js").write_text("const SONGS = " + json.dumps(songs) + ";\n")
    print(f"wrote {OUT}/data.js")


if __name__ == "__main__":
    main()
