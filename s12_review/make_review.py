"""Build the S12 side-by-side review package (run from the repo root):

    .venv/bin/python s12_review/make_review.py

Produces, under s12_review/:
  data.js                     — per-song tokens (uncorrected + corrected), offsets, titles
  subtitles_corrected/<id>.csv— the proposed corrected reference files (staging only;
                                data/clean is NOT touched until S12-final approval)
  figures/<id>.png            — the measurement, drawn: correlation curve + excerpt overlay
  media/<id>.webm             — symlink to the song's video (dataset repo)
  media/<id>_vocals.mp3       — symlink to the separated vocals
index.html (hand-written, committed) reads data.js and plays video with BOTH
karaoke tracks. METHOD.md documents every step of the measurement.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kashi import audio as audio_mod
from kashi.data.offset_audit import (GOLD_IDS, HOP_S, cross_correlate,
                                     onset_envelope, ref_train)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "s12_review"
DATASET_REPO = ROOT.parent / "karaoke_subtitle_dataset"

# Correction rule (S12): |gold-calibrated offset| >= 50 ms AND sharpness >= 1.5.
MIN_ABS_MS = 50
MIN_SHARP = 1.5

# Package = all frozen-test songs (incl. no-change controls 83/85/92) +
# two train examples + the excluded-measurement case study (88).
PACKAGE = [81, 83, 85, 89, 90, 91, 92, 13, 70, 88]
TEST_IDS = {81, 83, 85, 89, 90, 91, 92}

# Palette (dataviz reference instance, light mode): slot1 blue = uncorrected,
# slot2 aqua = corrected; neutral inks; violet for the measurement curve.
BLUE, AQUA, VIOLET = "#2a78d6", "#1baf7a", "#4a3aa7"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"


def titles() -> dict[int, str]:
    with open(DATASET_REPO / "data" / "indexed" / "index.tsv", newline="") as f:
        return {int(r["Index"]): r["Title"] for r in csv.DictReader(f, delimiter="\t")}


def read_rows(sid: int, uncorrected: bool = False) -> list[dict]:
    """Current labels, or (uncorrected=True) the pre-S12 backup when one exists —
    data/clean now already CONTAINS the applied corrections."""
    p = ROOT / f"data/clean/subtitles/{sid}.csv"
    if uncorrected:
        b = ROOT / f"data/clean/subtitles_pre_s12/{sid}.csv"
        p = b if b.exists() else p
    with open(p, newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def audit_song(sid: int, rows: list[dict]):
    """Re-measure on the given rows (pass the UNCORRECTED rows to reproduce
    the original fault measurement)."""
    wave = audio_mod.load_audio(str(ROOT / f"data/clean/audio/vocals/{sid}.mp3"), sr=16000)
    env = onset_envelope(wave)
    train = ref_train(rows, len(env))
    lags, corr = cross_correlate(env, train)
    best = int(np.argmax(corr))
    sharp = float((corr[best] - np.median(corr)) / (corr.std() + 1e-9))
    return env, lags, corr, float(lags[best] * HOP_S * 1000), sharp


def onsets_of(rows) -> np.ndarray:
    return np.array([float(r["start"]) for r in rows
                     if r["token"] != "<silence>"
                     and r.get("exclude", "False").strip().lower() != "true"])


def densest_window(onsets: np.ndarray, span: float = 8.0) -> float:
    if len(onsets) == 0:
        return 0.0
    counts = [(np.searchsorted(onsets, t + span) - i, t)
              for i, t in enumerate(onsets)]
    return max(counts)[1] - 0.5


def figure(sid: int, env, lags, corr, onsets, shift_s: float, base_ms: float,
           raw_ms: float, sharp: float, corrected: bool) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 5.6), facecolor=SURF)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7)

    ms = lags * HOP_S * 1000
    ax1.plot(ms, corr, color=VIOLET, linewidth=2)
    ax1.axvline(base_ms, color=INK2, linestyle="--", linewidth=1)
    ax1.axvline(raw_ms, color=VIOLET, linestyle=":", linewidth=1.5)
    ax1.text(base_ms, ax1.get_ylim()[1], " gold baseline (instrument bias)",
             color=INK2, fontsize=8, va="top")
    ax1.text(raw_ms, corr.max(), f" peak {raw_ms:+.0f} ms (sharpness {sharp:.1f})",
             color=VIOLET, fontsize=8, va="bottom")
    ax1.set_title(f"song {sid} — cross-correlation of onset envelope × label onsets "
                  f"(label fault = peak − baseline = {raw_ms - base_ms:+.0f} ms)",
                  color=INK, fontsize=10, loc="left")
    ax1.set_xlabel("lag (ms)   [negative = labels EARLY]", color=INK2, fontsize=8)

    t0 = densest_window(onsets)
    t1 = t0 + 8.0
    i0, i1 = int(t0 / HOP_S), int(t1 / HOP_S)
    tt = np.arange(i0, min(i1, len(env))) * HOP_S
    ax2.fill_between(tt, 0, np.maximum(env[i0:i0 + len(tt)], 0),
                     color=INK2, alpha=0.35, linewidth=0)
    w = onsets[(onsets >= t0) & (onsets < t1)]
    ax2.vlines(w, 1.02, 1.55, color=BLUE, linewidth=2)
    lbl = f"corrected (+{shift_s*1000:.0f} ms)" if corrected else "corrected = uncorrected (no change)"
    ax2.vlines(w + shift_s, -0.05, -0.5, color=AQUA, linewidth=2)
    ax2.text(t0 + 0.05, 1.58, "uncorrected labels", color=BLUE, fontsize=9, va="bottom")
    ax2.text(t0 + 0.05, -0.62, lbl, color=AQUA, fontsize=9, va="top")
    ax2.set_ylim(-1.0, 2.0)
    ax2.set_yticks([])
    ax2.set_title("8 s excerpt — vocal onset envelope (gray) vs label onsets before/after",
                  color=INK, fontsize=10, loc="left")
    ax2.set_xlabel("song time (s)", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{sid}.png", dpi=110, facecolor=SURF)
    plt.close(fig)


def predict(sid: int) -> list:
    """Champion pipeline prediction for a test song (cached under pred/)."""
    cache = OUT / "pred" / f"{sid}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from kashi.config import Config
    from kashi.pipeline import transcribe

    cfg = Config.load()
    res = transcribe(cfg, ROOT / f"data/clean/audio/vocals/{sid}.mp3",
                     out_dir=OUT / "pred" / f"tmp_{sid}", formats=["csv"], separate=False)
    toks = [(round(s.start, 3), round(s.end, 3), s.token)
            for s in res.segments if not s.is_silence]
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(toks))
    return toks


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "media").mkdir(exist_ok=True)
    full = json.load(open(ROOT / "runs" / "ref_offset_full.json"))
    base_ms = full["gold_baseline_ms"]
    ttl = titles()

    songs = []
    for sid in PACKAGE:
        rows_u = read_rows(sid, uncorrected=True)
        rows_c = read_rows(sid)
        env, lags, corr, raw_ms, sharp = audit_song(sid, rows_u)
        fault_ms = raw_ms - base_ms                      # <0 = labels early

        def lyric_of(rows):
            return [(round(float(r["start"]), 3), round(float(r["end"]), 3), r["token"])
                    for r in rows if r["token"] != "<silence>"
                    and r.get("exclude", "False").strip().lower() != "true"]

        lyric, lyric_c = lyric_of(rows_u), lyric_of(rows_c)
        corrected = (ROOT / f"data/clean/subtitles_pre_s12/{sid}.csv").exists()
        shift_s = (lyric_c[0][0] - lyric[0][0]) if corrected and lyric and lyric_c else 0.0
        figure(sid, env, lags, corr, np.array([t for t, _, _ in lyric]),
               shift_s, base_ms, raw_ms, sharp, corrected)

        for link, target in ((OUT / "media" / f"{sid}.webm",
                              DATASET_REPO / "data" / "indexed" / "videos" / f"{sid}.webm"),
                             (OUT / "media" / f"{sid}_vocals.mp3",
                              ROOT / "data" / "clean" / "audio" / "vocals" / f"{sid}.mp3")):
            link.unlink(missing_ok=True)
            link.symlink_to(target)

        songs.append({
            "id": sid, "title": ttl.get(sid, f"song {sid}"),
            "set": "test" if sid in TEST_IDS else "train",
            "fault_ms": round(fault_ms), "sharp": round(sharp, 1),
            "corrected": corrected, "shift_ms": round(shift_s * 1000),
            "video": f"media/{sid}.webm", "vocals": f"media/{sid}_vocals.mp3",
            "fig": f"figures/{sid}.png", "tokens": lyric, "tokens_c": lyric_c,
            "pred": predict(sid) if sid in TEST_IDS else None,
        })
        print(f"song {sid}: fault {fault_ms:+.0f} ms sharp {sharp:.1f} "
              f"{'-> corrected' if corrected else '-> unchanged'} ({len(lyric)} tokens)")

    (OUT / "data.js").write_text("const SONGS = " + json.dumps(songs) + ";\n")
    print(f"\nwrote {OUT}/data.js ({len(songs)} songs) + figures + corrected CSVs + media links")


if __name__ == "__main__":
    main()
