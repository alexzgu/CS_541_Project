"""Dataset layout and the frozen train/test split.

data/
  raw/audio/<id>.mp3                        scraped full mixes
  raw/subtitles/subtitle_files/<id>.csv     scraped rows (start,end,line,unformatted,token)
  raw/subtitles/clips_to_exclude/<id>.txt   manually flagged bad time ranges
  raw/subtitles/index.tsv                   Index / Title / YouTube ID / Language
  clean/audio/vocals/<id>.mp3               separated vocals
  <version>/subtitles/<id>.csv              labels (start,end,token,exclude[,...])

The paper split is positional over the sorted song ids: [0:80] train,
[80:93] test, intersected with the labeled set — frozen for comparability.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SongPaths:
    song_id: int
    raw_audio: Path
    vocals: Path
    raw_subtitles: Path
    exclude_ranges: Path

    def subtitles(self, cfg, version: str | None = None) -> Path:
        return subtitles_dir(cfg, version) / f"{self.song_id}.csv"


def index_file(cfg) -> Path:
    return cfg.data_dir / "raw" / "subtitles" / "index.tsv"


def subtitles_dir(cfg, version: str | None = None) -> Path:
    version = version or cfg["data.version"]
    return cfg.data_dir / version / "subtitles"


def vocals_dir(cfg) -> Path:
    return cfg.data_dir / "clean" / "audio" / "vocals"


def read_index(cfg) -> list[dict]:
    path = index_file(cfg)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def song_ids(cfg) -> list[int]:
    """All dataset song ids, sorted. Source of truth: index.tsv; fallback: raw audio dir."""
    rows = read_index(cfg)
    if rows:
        return sorted(int(r["Index"]) for r in rows)
    audio_dir = cfg.data_dir / "raw" / "audio"
    if audio_dir.is_dir():
        return sorted(int(p.stem) for p in audio_dir.glob("*.mp3") if p.stem.isdigit())
    return []


def labeled_ids(cfg, version: str | None = None) -> list[int]:
    d = subtitles_dir(cfg, version)
    if not d.is_dir():
        return []
    return sorted(int(p.stem) for p in d.glob("*.csv") if p.stem.isdigit())


def song_paths(cfg, song_id: int) -> SongPaths:
    data = cfg.data_dir
    return SongPaths(
        song_id=song_id,
        raw_audio=data / "raw" / "audio" / f"{song_id}.mp3",
        vocals=vocals_dir(cfg) / f"{song_id}.mp3",
        raw_subtitles=data / "raw" / "subtitles" / "subtitle_files" / f"{song_id}.csv",
        exclude_ranges=data / "raw" / "subtitles" / "clips_to_exclude" / f"{song_id}.txt",
    )


# The paper's test songs, FROZEN as an explicit list: originally the labeled
# ids in positions [80:93]. Pinned so that admitting new labeled songs with
# ids in that range (t2-extra) cannot silently change the test set.
PAPER_TEST_IDS = (81, 83, 85, 89, 90, 91, 92)


def split_ids(cfg, version: str | None = None) -> tuple[list[int], list[int]]:
    """(train_ids, test_ids): frozen paper test set; everything else trains."""
    labeled = set(labeled_ids(cfg, version))
    test = [i for i in PAPER_TEST_IDS if i in labeled]
    train = sorted(labeled - set(PAPER_TEST_IDS))
    return train, test
