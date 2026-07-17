"""Admit staged T1 songs into the corpus (labels + audio), ids 93+.

T1 label format (data/imported/t1/subtitle_files/<ytid>.csv): per-token rows
`start,end,line,unformatted,token` where token is kana or 漢字(かな) furigana
units. Conversion: extract the reading, then run the standard v2 normalizer
(kanji readings of 2+ morae split proportionally; っ folds into its host;
katakana folded to hiragana; anything unmappable stays as an excluded row for
Tier 2). Explicit <silence> rows are synthesized for gaps >= 0.4 s to match
corpus conventions (crop cutting + eval gap handling).

New ids are assigned 93+ in sorted-ytid order and recorded in
data/imported/admitted_index.tsv. PAPER_TEST_IDS is a frozen literal, so all
admitted songs land in TRAIN automatically. Audio: webm -> data/raw/audio/
<id>.mp3 via ffmpeg (vocals separation is a separate GPU pass).
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

from ..tokens import TOKENS
from .normalize import _INVENTORY, normalize_rows

_KATA = {chr(k): chr(k - 0x60) for k in range(0x30A1, 0x30F7)}
_FURI = re.compile(r"(.)\(([ぁ-ゖァ-ヶー]+)\)")
_KANA_OK = re.compile(r"^[ぁ-ゖー]+$")
MIN_SIL_S = 0.4


def reading_of(token: str) -> str | None:
    """kana reading of a T1 token, or None if unmappable (ascii/symbols)."""
    t = _FURI.sub(lambda m: m.group(2), token)
    t = "".join(_KATA.get(ch, ch) for ch in t)
    return t if _KANA_OK.match(t) else None


def convert_t1(src: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(src, newline="")):
        tok = (r["token"] or "").strip()
        if not tok:
            continue
        kana = reading_of(tok)
        rows.append({"start": r["start"], "end": r["end"],
                     "token": kana if kana else tok,
                     "exclude": "False" if kana else "True"})
    out, _ = normalize_rows(rows)
    # synthesize corpus-convention <silence> rows in the gaps
    final, t = [], 0.0
    for r in out:
        s, e = float(r["start"]), float(r["end"])
        if s - t >= MIN_SIL_S:
            final.append({"start": str(round(t, 3)), "end": str(round(s, 3)),
                          "token": "<silence>", "exclude": "False"})
        final.append(r)
        t = max(t, e)
    return final


def admit(audio_dir: str | Path = "data/imported/audio",
          labels_dir: str | Path = "data/imported/t1/subtitle_files",
          start_id: int = 93, dry_run: bool = False) -> list[tuple[int, str]]:
    audio_dir, labels_dir = Path(audio_dir), Path(labels_dir)
    titles = {}
    for r in csv.DictReader(open("data/imported/index.tsv"), delimiter="\t"):
        titles[r["ID"]] = (r["Set"], r["Title"])
    have_audio = {p.stem: p for p in audio_dir.iterdir() if p.is_file()}
    todo = sorted(yt for yt in have_audio
                  if (labels_dir / f"{yt}.csv").exists())
    idx_path = Path("data/imported/admitted_index.tsv")
    already = set()
    if idx_path.exists():
        already = {r["yt"] for r in csv.DictReader(open(idx_path), delimiter="\t")}
        start_id = 1 + max(int(r["id"]) for r in csv.DictReader(open(idx_path), delimiter="\t"))
    admitted = []
    nid = start_id
    for yt in todo:
        if yt in already:
            continue
        rows = convert_t1(labels_dir / f"{yt}.csv")
        n_lyr = sum(1 for r in rows if r["token"] != "<silence>"
                    and r["exclude"] == "False")
        n_exc = sum(1 for r in rows if r["exclude"] == "True")
        if dry_run:
            print(f"[admit] {yt}: would become id {nid} ({n_lyr} morae, {n_exc} excluded)")
            nid += 1
            continue
        dst = Path(f"data/clean_v2/subtitles/{nid}.csv")
        with open(dst, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["start", "end", "token", "exclude"],
                                lineterminator="\n")
            wr.writeheader()
            wr.writerows(rows)
        mp3 = Path(f"data/raw/audio/{nid}.mp3")
        if not mp3.exists():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y",
                            "-i", str(have_audio[yt]), "-codec:a", "libmp3lame",
                            "-q:a", "2", str(mp3)], check=True)
        st, title = titles.get(yt, ("t1", ""))
        header = not idx_path.exists()
        with open(idx_path, "a", newline="") as f:
            wr = csv.writer(f, delimiter="\t", lineterminator="\n")
            if header:
                wr.writerow(["id", "set", "yt", "title"])
            wr.writerow([nid, st, yt, title])
        admitted.append((nid, yt))
        print(f"[admit] id {nid} <- {yt} ({n_lyr} morae, {n_exc} excluded) {title[:50]}")
        nid += 1
    print(f"[admit] {'would admit' if dry_run else 'admitted'} "
          f"{nid - start_id} songs (ids {start_id}..{nid - 1})")
    return admitted


if __name__ == "__main__":
    admit(dry_run="--dry-run" in sys.argv)
