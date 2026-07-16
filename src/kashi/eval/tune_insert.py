"""Tune deleted-vowel insertion (theta, min_gap) on gold + train — never test.

    python -m kashi.eval.tune_insert

Caches each tuning song's full CTC log-posteriors (fp16) under
runs/insert_logp/ so the sweep is offline CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import audio as audio_mod
from ..components.decoders import propose_continuations
from ..config import Config
from ..registry import create
from ..tokens import SILENCE_ID, TOKENS
from .tune_beam import GOLD, TRAIN_SAMPLE, lev, refs_for


def full_logp(cfg, decoder, sid: int) -> np.ndarray:
    p = Path("runs/insert_logp") / f"{sid}.npz"
    if p.exists():
        return np.load(p)["logp"].astype(np.float32)
    wave = audio_mod.load_audio(f"data/clean/audio/vocals/{sid}.mp3", sr=cfg.sample_rate)
    frame_s = cfg.frame_ms / 1000.0
    T = max(1, int(len(wave) / cfg.sample_rate / frame_s))
    logp = decoder._ctc_log_probs(wave, cfg.sample_rate, T)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, logp=logp.astype(np.float16))
    return logp


def main() -> None:
    cfg = Config.load()
    decoder = create(cfg, "decoder")
    frame_s = cfg.frame_ms / 1000.0
    songs = {}
    for sid in GOLD + TRAIN_SAMPLE:
        lp = full_logp(cfg, decoder, sid)
        path = lp.argmax(-1)
        spike_f = [t for t in range(len(path))
                   if path[t] != SILENCE_ID and (t == 0 or path[t] != path[t - 1])]
        spike_c = [int(path[t]) for t in spike_f]
        songs[sid] = (lp, spike_f, spike_c, refs_for(sid))
        print(f"cached {sid}: {len(spike_f)} spikes", flush=True)

    n_ref = sum(len(r) for *_, r in songs.values())
    base = sum(lev(r, [TOKENS[c] for c in sc]) for _, _, sc, r in songs.values()) / n_ref
    print(f"\ngreedy baseline SER {base:.4f}")
    print(f"{'':>12}" + "".join(f"  gap={g}s" for g in (0.2, 0.24, 0.3)))
    best = (None, base)
    for th in (0.02, 0.05, 0.1, 0.15, 0.25):
        row = [f"theta={th:<5}"]
        for gap in (0.2, 0.24, 0.3):
            d = n_ins = 0
            for lp, sf, sc, ref in songs.values():
                ins = propose_continuations(lp, sf, sc, frame_s, th, gap)
                n_ins += len(ins)
                merged = sorted([(f, c) for f, c in zip(sf, sc)] + list(ins))
                d += lev(ref, [TOKENS[c] for _, c in merged])
            ser = d / n_ref
            row.append(f"{ser:.4f}({n_ins})")
            if ser < best[1]:
                best = ((th, gap), ser)
        print("  ".join(row), flush=True)
    if best[0]:
        print(f"\nbest: theta={best[0][0]}, gap={best[0][1]} -> SER {best[1]:.4f} "
              f"(baseline {base:.4f})")
    else:
        print("\nno setting beats the greedy baseline")


if __name__ == "__main__":
    main()
