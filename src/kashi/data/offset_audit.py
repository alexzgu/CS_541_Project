"""S12 reference-timing audit: per-song constant offset of subtitle labels vs
acoustics — NO model in the loop (avoids circularity with the CTC champion).

Method (every step, no magic):
1. Onset envelope from the separated vocals: STFT (512-sample Hann = 32 ms
   window, 10 ms hop), log-compressed magnitude ``log1p(50|Z|)``, per-bin
   positive first difference summed over bins (spectral flux, half-wave
   rectified), then standardized. Peaks where vocal energy APPEARS.
2. Reference onset train: a Gaussian bump (sigma 20 ms) at every non-silence,
   non-excluded token's start time; standardized.
3. Normalized cross-correlation of (1) and (2) over lags in [-200, +200] ms.
   The argmax lag is the song's constant offset. Positive = labels LATE
   relative to the audio, negative = labels EARLY.
4. Instrument calibration: the same measurement on the hand-corrected gold
   songs {0, 6, 16, 19} reads ~-40 ms even though their labels are correct —
   spectral flux inherently peaks after the true onset (energy has to build
   within the 32 ms window before flux rises). The gold median is therefore
   subtracted from every reading; what remains is the label fault.

`corrected_ms[sid]` < 0 means song `sid`'s labels are that many ms EARLY, and
fixing them means shifting every row LATER by `-corrected_ms[sid]`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import stft

from .. import audio as audio_mod

SR = 16000
HOP_S = 0.010        # 10 ms audit resolution
MAX_LAG_S = 0.20
GOLD_IDS = (0, 6, 16, 19)


def onset_envelope(wave: np.ndarray) -> np.ndarray:
    """Spectral flux at 10 ms hop: log-magnitude STFT, half-wave-rectified
    per-bin first difference, summed over frequency, standardized."""
    nper = 512  # 32 ms window
    hop = int(SR * HOP_S)
    _, _, Z = stft(wave, fs=SR, nperseg=nper, noverlap=nper - hop, padded=True)
    mag = np.log1p(50.0 * np.abs(Z))                     # [F, T]
    flux = np.diff(mag, axis=1)
    flux[flux < 0] = 0
    env = flux.sum(0)
    env = env - env.mean()
    return env / (env.std() + 1e-9)


def ref_train(rows: list[dict], n: int) -> np.ndarray:
    """Gaussian bumps (sigma 20 ms) at each non-silence token onset."""
    train = np.zeros(n)
    sig = 0.020 / HOP_S
    for r in rows:
        tok = r["token"]
        if tok == "<silence>" or r.get("exclude", "False").strip().lower() == "true":
            continue
        t = float(r["start"]) / HOP_S
        lo, hi = int(max(0, t - 4 * sig)), int(min(n, t + 4 * sig + 1))
        if hi > lo:
            idx = np.arange(lo, hi)
            train[idx] += np.exp(-0.5 * ((idx - t) / sig) ** 2)
    train = train - train.mean()
    return train / (train.std() + 1e-9)


def cross_correlate(env: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """corr[k] over lags k in [-MAX_LAG, +MAX_LAG]; k>0 means the reference
    train must be pulled EARLIER to match the envelope (labels late)."""
    max_lag = int(MAX_LAG_S / HOP_S)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([np.dot(env[max(0, -k): len(env) - max(0, k)],
                            train[max(0, k): len(train) - max(0, -k)]) for k in lags])
    return lags, corr / len(env)


def subtitle_path(sid: int) -> str:
    return (f"data/gold/subtitles/{sid}.csv" if sid in GOLD_IDS
            else f"data/clean/subtitles/{sid}.csv")


def song_offset(sid: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    """(raw lag ms, peak sharpness, lags, corr-curve) for one song."""
    wave = audio_mod.load_audio(f"data/clean/audio/vocals/{sid}.mp3", sr=SR)
    env = onset_envelope(wave)
    with open(subtitle_path(sid), newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    train = ref_train(rows, len(env))
    lags, corr = cross_correlate(env, train)
    best = int(np.argmax(corr))
    sharp = float((corr[best] - np.median(corr)) / (corr.std() + 1e-9))
    return float(lags[best] * HOP_S * 1000), sharp, lags, corr


def audit(ids: list[int], out_json: str | Path | None = None) -> dict:
    """Audit `ids` with gold calibration; optionally dump machine-readable JSON."""
    raw: dict[int, float] = {}
    print(f"{'song':>6} {'set':<6} {'raw lag ms':>10} {'sharpness':>9}   (positive = labels late)")
    for sid in list(GOLD_IDS) + [i for i in ids if i not in GOLD_IDS]:
        lag, sharp, _, _ = song_offset(sid)
        raw[sid] = lag
        print(f"{sid:>6} {'gold' if sid in GOLD_IDS else 'v1':<6} {lag:>10.0f} {sharp:>9.1f}", flush=True)
    base = float(np.median([raw[g] for g in GOLD_IDS]))
    corrected = {k: round(v - base) for k, v in raw.items() if k not in GOLD_IDS}
    print(f"\ngold baseline (instrument bias): {base:+.0f} ms")
    report = {"gold_baseline_ms": base, "raw_lag_ms": raw, "corrected_ms": corrected}
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=1))
        print(f"wrote {out_json}")
    return report


MIN_ABS_MS = 50      # correction rule (S12): |gold-calibrated fault| >= this ...
MIN_SHARP = 1.5      # ... AND the correlation peak is unambiguous


def apply_corrections(report: dict | str | Path, dry_run: bool = False) -> dict[int, int]:
    """Shift each qualifying song's reference rows by its measured constant
    (S12-approved). Originals are backed up to data/clean/subtitles_pre_s12/
    once; refuses to run twice. Returns {song: applied_shift_ms}."""
    import shutil

    if not isinstance(report, dict):
        report = json.loads(Path(report).read_text())
    base = report["gold_baseline_ms"]
    backup = Path("data/clean/subtitles_pre_s12")
    applied: dict[int, int] = {}
    for sid_s, corr_ms in sorted(report["corrected_ms"].items(), key=lambda kv: int(kv[0])):
        sid = int(sid_s)
        if abs(corr_ms) < MIN_ABS_MS:
            continue
        if (backup / f"{sid}.csv").exists():
            print(f"[apply] song {sid}: already corrected (backup exists) — skipped")
            continue
        raw_ms, sharp, _, _ = song_offset(sid)          # re-measure for sharpness
        fault_ms = raw_ms - base
        if sharp < MIN_SHARP:
            print(f"[apply] song {sid}: fault {fault_ms:+.0f} ms but sharpness "
                  f"{sharp:.1f} < {MIN_SHARP} — SKIPPED (unreliable measurement)")
            continue
        shift_s = -fault_ms / 1000.0
        src = Path(subtitle_path(sid))
        if dry_run:
            print(f"[apply] song {sid}: would shift {shift_s*1000:+.0f} ms")
            applied[sid] = round(shift_s * 1000)
            continue
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, backup / src.name)
        with open(src, newline="") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames)
            rows = [dict(r) for r in rd]
        for r in rows:
            r["start"] = str(round(max(0.0, float(r["start"]) + shift_s), 3))
            r["end"] = str(round(max(0.0, float(r["end"]) + shift_s), 3))
        with open(src, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            wr.writeheader()
            wr.writerows(rows)
        applied[sid] = round(shift_s * 1000)
        print(f"[apply] song {sid}: shifted {shift_s*1000:+.0f} ms "
              f"(fault {fault_ms:+.0f}, sharpness {sharp:.1f})")
    print(f"[apply] {'would correct' if dry_run else 'corrected'} {len(applied)} songs; "
          f"originals in {backup}/")
    return applied


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("apply", "apply-dry"):
        apply_corrections("runs/ref_offset_full.json", dry_run=sys.argv[1] == "apply-dry")
    else:
        ids = [int(x) for x in sys.argv[1:]] or [i for i in range(93)]
        audit(ids, out_json="runs/ref_offset_full.json")
