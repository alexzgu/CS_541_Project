"""Build clean per-syllable labels from the scraped karaoke subtitles.

Faithful port of the legacy chain
    data_processing/main_functions/clean_subtitles.py
    + change_last_end_to_vid_length.py + utils/reduce_silence.py
collapsed into pure functions (pandas>=2: `pd.concat` replaces the removed
`DataFrame.append`). Raw rows: start,end,line,unformatted,token — `line` is
the vertical position bucket (0-49 top / >=50 bottom / -1 unknown): karaoke
videos render each lyric twice (kana + kanji), so one hemisphere is dropped.
Output rows: start,end,token,exclude.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import audio as audio_mod
from ..tokens import TOKENS, SILENCE, keep_alnum, keep_kana, katakana_to_hiragana
from . import manifest

GAP = "<gap>"
TOKEN_SET = set(TOKENS)


@dataclass
class TimeRange:
    start: float
    end: float


def read_time_ranges(path: str | Path) -> list[TimeRange]:
    """clips_to_exclude format: 'start1:end1,start2:end2,...' (whitespace ignored)."""
    path = Path(path)
    if not path.is_file():
        return []
    data = path.read_text().replace(" ", "").replace("\t", "").replace("\n", "")
    if not data:
        return []
    out = []
    for part in data.split(","):
        if not part:
            continue
        start, end = part.split(":")
        out.append(TimeRange(float(start), float(end)))
    return out


def compute_silence_ranges(df: pd.DataFrame) -> list[TimeRange]:
    if df["start"].isnull().values.any() or df["end"].isnull().values.any():
        raise ValueError("NaN values in 'start'/'end' of raw subtitles")
    df = df.sort_values(["start", "end", "line"]).reset_index(drop=True)
    if len(df) == 0:
        return [TimeRange(0.0, -1)]
    ranges: list[TimeRange] = []
    if float(df.iloc[0]["start"]) > 0:
        ranges.append(TimeRange(0.0, float(df.iloc[0]["start"])))
    for i in range(len(df) - 1):
        cur_end = float(df.iloc[i]["end"])
        nxt_start = float(df.iloc[i + 1]["start"])
        if nxt_start > cur_end:
            ranges.append(TimeRange(cur_end, nxt_start))
    ranges.append(TimeRange(float(df.iloc[-1]["end"]), -1))
    return ranges


def compute_overlaps(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    flag = [False] * n
    starts = df["start"].to_numpy()
    ends = df["end"].to_numpy()
    for i in range(n):
        j = i + 1
        while j < n and starts[j] < ends[i]:
            flag[i] = True
            flag[j] = True
            j += 1
    return pd.Series(flag, index=df.index, dtype=bool)


def remove_hemisphere(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the hemisphere (top/bottom lyric copy) with more kana; unknown-position
    rows (line == -1) follow the smaller hemisphere — the original heuristic."""
    df = df.copy()
    A = df[(df["line"] >= 0) & (df["line"] < 50)]
    B = df[df["line"] >= 50]
    C = df[df["line"] == -1]
    if A.shape[0] < B.shape[0]:
        A = pd.concat([A, C])
        c_with = "A"
    else:
        B = pd.concat([B, C])
        c_with = "B"
    a_count = sum(len(x) for x in A["cleaned_token"])
    b_count = sum(len(x) for x in B["cleaned_token"])
    if a_count == b_count:
        raise ValueError("hemisphere kana counts equal — ambiguous raw file")
    if a_count < b_count:
        if c_with == "A":
            df = df[~((df["line"] < 50) & df["overlap"])]
        else:
            df = df[~((df["line"] >= 0) & (df["line"] < 50) & df["overlap"])]
    else:
        if c_with == "B":
            df = df[~(((df["line"] >= 50) | (df["line"] == -1)) & df["overlap"])]
        else:
            df = df[~((df["line"] >= 50) & df["overlap"])]
    return df.reset_index(drop=True)


def insert_silence_and_excluded(
    df: pd.DataFrame, ignore: list[TimeRange], silences: list[TimeRange]
) -> pd.DataFrame:
    if silences and silences[-1].end == -1:
        silences[-1].end = float("inf")
    sil_rows = [
        {"start": s.start, "end": s.end, "overlap": False, "token": SILENCE, "cleaned_token": SILENCE}
        for s in silences
    ]
    if sil_rows:
        df = pd.concat([df, pd.DataFrame(sil_rows)], ignore_index=True)
    df = df.sort_values(["start", "end"]).reset_index(drop=True)

    df["ignore"] = False
    for tr in ignore:
        overlapping = ~((tr.end <= df["start"]) | (tr.start >= df["end"]))
        df.loc[overlapping, "ignore"] = True
    df = df.sort_values(["start", "end"]).reset_index(drop=True)

    gap_rows = []
    for i in range(1, len(df)):
        if df.at[i, "start"] > df.at[i - 1, "end"]:
            gap_rows.append(
                {
                    "start": df.at[i - 1, "end"],
                    "end": df.at[i, "start"],
                    "overlap": False,
                    "token": GAP,
                    "cleaned_token": GAP,
                    "ignore": False,
                    "gap": True,
                }
            )
    df["gap"] = False
    if gap_rows:
        df = pd.concat([df, pd.DataFrame(gap_rows)], ignore_index=True)
    df = df.sort_values(["start", "end"]).reset_index(drop=True)

    df["other"] = df["overlap"] | df["cleaned_token"].isnull()
    df["exclude"] = df["ignore"] | df["gap"] | df["other"]
    return df.drop(columns=["ignore", "gap", "other"])


def _merge_removed_into_prev(df: pd.DataFrame, remove_mask: pd.Series) -> pd.DataFrame:
    """Remove rows, extending the previous row's end over each removed row."""
    df = df.copy()
    if remove_mask.any():
        for idx in df[remove_mask].index:
            prior = df.index[df.index < idx]
            if len(prior):
                df.at[prior[-1], "end"] = df.at[idx, "end"]
        df = df[~remove_mask]
    return df.reset_index(drop=True)


def filter_long_vowels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["token"] = df["token"].str.replace("ー", "")
    return _merge_removed_into_prev(df, df["token"].str.len() == 0)


def filter_null_tokens(df: pd.DataFrame) -> pd.DataFrame:
    return _merge_removed_into_prev(df.copy(), df["token"].isnull())


def clean_song(raw: pd.DataFrame, ignore: list[TimeRange]) -> pd.DataFrame:
    """The pure-CSV part of the chain: raw rows -> start,end,token,exclude."""
    df = raw.drop(columns=["unformatted"])
    df = df.copy()
    df["cleaned_token"] = df["token"].apply(lambda x: keep_kana(str(x)))
    silences = compute_silence_ranges(df)

    df["overlap"] = compute_overlaps(df)
    df = remove_hemisphere(df)
    df = df.drop(columns=["line"])
    df["cleaned_token"] = df["token"].apply(lambda x: keep_kana(str(x), include_katakana=True))

    df["overlap"] = compute_overlaps(df)
    df = df[~df["overlap"]]

    df = insert_silence_and_excluded(df, ignore, silences)
    df = df.drop(columns=["overlap"])
    df["cleaned_token"] = df["cleaned_token"].replace("", None)
    df["cleaned_token"] = df["cleaned_token"].fillna(df["token"].apply(lambda x: keep_alnum(str(x))))
    df = df.drop(columns=["token"]).rename(columns={"cleaned_token": "token"})

    df = filter_long_vowels(df)
    df = filter_null_tokens(df)
    df["token"] = df["token"].apply(katakana_to_hiragana)
    df["exclude"] = df["exclude"] | ~df["token"].isin(TOKEN_SET)
    return df[["start", "end", "token", "exclude"]].reset_index(drop=True)


def set_last_end_to_duration(df: pd.DataFrame, duration: float) -> pd.DataFrame:
    df = df.copy()
    if len(df):
        df.loc[df.index[-1], "end"] = round(duration, 3)
    return df


# ---------------------------------------------------------------------------
# Silence trimming (port of utils/reduce_silence.py; librosa.effects.trim
# reimplemented with numpy: rms frames, dB relative to max, top_db threshold).
# ---------------------------------------------------------------------------

def _trim_length(segment: np.ndarray, top_db: float = 20.0, frame: int = 2048, hop: int = 512) -> int:
    """Length (samples) of the trailing-trimmed segment, replicating the legacy
    use of librosa.effects.trim (whose leading offset the legacy code ignored)."""
    if len(segment) == 0:
        return 0
    pad = frame // 2
    padded = np.pad(segment.astype(np.float64), pad, mode="reflect") if len(segment) > pad else segment.astype(np.float64)
    n_frames = 1 + max(0, (len(padded) - frame)) // hop
    if n_frames == 0:
        return len(segment)
    idx = np.arange(n_frames)[:, None] * hop + np.arange(frame)[None, :]
    idx = np.minimum(idx, len(padded) - 1)
    rms2 = (padded[idx] ** 2).mean(axis=1)
    ref = rms2.max()
    if ref <= 0:
        return 0
    db = 10.0 * np.log10(rms2 / ref + 1e-20)
    nonsilent = np.flatnonzero(db > -top_db)
    if len(nonsilent) == 0:
        return 0
    start = int(nonsilent[0] * hop)
    end = int(min(len(segment), (nonsilent[-1] + 1) * hop))
    return max(0, end - start)


def reduce_silence(df: pd.DataFrame, wave: np.ndarray, sr: int, top_db: float = 20.0) -> pd.DataFrame:
    """Pull each non-silence row's end in by the amount of silence trimmed from
    its audio span (legacy kept the start fixed), then re-anchor silence rows."""
    df = df.copy()
    for i in df.index:
        if df.at[i, "token"] == SILENCE:
            continue
        s = int(float(df.at[i, "start"]) * sr)
        e = int(float(df.at[i, "end"]) * sr)
        seg = wave[max(0, s):max(0, e)]
        trimmed = _trim_length(seg, top_db=top_db)
        if trimmed > 0 and len(seg) > 0:
            df.at[i, "end"] = float(df.at[i, "start"]) + trimmed / sr
    for pos, i in enumerate(df.index):
        if df.at[i, "token"] == SILENCE:
            if pos > 0:
                df.at[i, "start"] = df.at[df.index[pos - 1], "end"]
            elif pos < len(df) - 1:
                df.at[i, "end"] = df.at[df.index[pos + 1], "start"]
    df["start"] = df["start"].astype(float).round(3)
    df["end"] = df["end"].astype(float).round(3)
    return df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_song(cfg, song_id: int, trim: bool = True) -> pd.DataFrame:
    paths = manifest.song_paths(cfg, song_id)
    raw = pd.read_csv(paths.raw_subtitles)
    df = clean_song(raw, read_time_ranges(paths.exclude_ranges))
    audio_path = paths.vocals if paths.vocals.is_file() else paths.raw_audio
    if audio_path.is_file():
        df = set_last_end_to_duration(df, audio_mod.duration_s(audio_path))
        if trim:
            wave = audio_mod.load_audio(audio_path, sr=cfg.sample_rate)
            df = reduce_silence(df, wave, cfg.sample_rate)
    return df


def build_dataset(cfg, out_version: str = "clean", force: bool = False, trim: bool = True) -> list[int]:
    """raw/subtitles -> <out_version>/subtitles for every song with raw labels."""
    out_dir = manifest.subtitles_dir(cfg, out_version)
    if out_dir.is_dir() and any(out_dir.glob("*.csv")) and not force:
        raise SystemExit(
            f"{out_dir} already contains labels; pass --force to overwrite "
            f"(label versions are append-only by policy — prefer a new --out-version)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    done: list[int] = []
    raw_dir = cfg.data_dir / "raw" / "subtitles" / "subtitle_files"
    for path in sorted(raw_dir.glob("*.csv"), key=lambda p: int(p.stem)):
        song_id = int(path.stem)
        try:
            df = build_song(cfg, song_id, trim=trim)
        except Exception as e:  # keep going; report at the end
            print(f"[build] song {song_id}: FAILED ({e})")
            continue
        df.to_csv(out_dir / f"{song_id}.csv", index=False)
        done.append(song_id)
    print(f"[build] built labels for {len(done)} songs -> {out_dir}")
    return done


def download_audio(cfg, out_dir: Path | None = None) -> None:
    """Fetch dataset audio with yt-dlp from index.tsv (port of download_audio.sh)."""
    import subprocess

    out_dir = out_dir or (cfg.data_dir / "raw" / "audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest.read_index(cfg):
        idx, vid = int(row["Index"]), row["ID"]
        target = out_dir / f"{idx}.mp3"
        if target.is_file():
            continue
        print(f"[download] {idx} <- {vid}")
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", str(out_dir / f"{idx}.%(ext)s"),
             f"https://www.youtube.com/watch?v={vid}"],
            check=False,
        )
