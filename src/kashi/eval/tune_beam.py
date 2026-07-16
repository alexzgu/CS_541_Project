"""Tune the beam decoder's (K, lambda) on gold + train songs — NEVER on test
(the S12 onset-shift episode is why). Usage:

    python -m kashi.eval.tune_beam            # gold {0,6,16,19} + 12 train songs

Caches each song's spike lattice (top-8 candidates + log-probs) under
runs/beam_lattices/ so the sweep itself is pure CPU and re-runs are instant.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .. import audio as audio_mod
from ..components.decoders import beam_pick
from ..config import Config
from ..registry import create
from ..stats.lm import log_bigram
from ..tokens import SILENCE_ID, TOKENS

GOLD = [0, 6, 16, 19]
TRAIN_SAMPLE = [3, 9, 22, 33, 41, 55, 62, 68, 74, 79, 84, 86]
CACHE_K = 8


def refs_for(sid: int) -> list[str]:
    src = (f"data/gold_v2/subtitles/{sid}.csv" if sid in GOLD
           else f"data/clean_v2/subtitles/{sid}.csv")
    return [r["token"] for r in csv.DictReader(open(src, newline=""))
            if r["token"] != "<silence>"
            and r.get("exclude", "False").strip().lower() != "true"]


def lattice(cfg, decoder, sid: int) -> dict:
    p = Path("runs/beam_lattices") / f"{sid}.npz"
    if p.exists():
        z = np.load(p)
        return {"cands": z["cands"], "scores": z["scores"]}
    wave = audio_mod.load_audio(f"data/clean/audio/vocals/{sid}.mp3", sr=cfg.sample_rate)
    frame_s = cfg.frame_ms / 1000.0
    T = max(1, int(len(wave) / cfg.sample_rate / frame_s))
    logp = decoder._ctc_log_probs(wave, cfg.sample_rate, T)
    path = logp.argmax(-1)
    spike_f = [t for t in range(len(path))
               if path[t] != SILENCE_ID and (t == 0 or path[t] != path[t - 1])]
    cands = np.zeros((len(spike_f), CACHE_K), dtype=np.int64)
    scores = np.zeros((len(spike_f), CACHE_K), dtype=np.float32)
    for i, t in enumerate(spike_f):
        order = [int(c) for c in np.argsort(logp[t])[::-1] if c != SILENCE_ID][:CACHE_K]
        cands[i] = order
        scores[i] = logp[t, order]
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, cands=cands, scores=scores)
    return {"cands": cands, "scores": scores}


def lev(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main() -> None:
    import sys

    cfg = Config.load()
    decoder = create(cfg, "decoder")
    if len(sys.argv) > 1:  # alternate LM artifact (e.g. text-trained bigram)
        log_A = np.load(sys.argv[1])["log_bigram"]
        print(f"using LM {sys.argv[1]}")
    else:
        log_A = log_bigram(cfg)
    songs = {}
    for sid in GOLD + TRAIN_SAMPLE:
        songs[sid] = (lattice(cfg, decoder, sid), refs_for(sid))
        print(f"lattice {sid}: {len(songs[sid][0]['cands'])} spikes", flush=True)

    n_ref = sum(len(r) for _, r in songs.values())
    print(f"\n{'':>10}" + "".join(f"  K={k}" for k in (3, 5, 8)))
    best = (None, 9e9)
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        row = [f"lam={lam:<5}"]
        for K in (3, 5, 8):
            d = 0
            for lat, ref in songs.values():
                cands = [list(c[:K]) for c in lat["cands"]]
                scores = [s[:K] for s in lat["scores"]]
                ks = beam_pick(cands, scores, log_A, lam)
                hyp = [TOKENS[cands[i][k]] for i, k in enumerate(ks)]
                d += lev(ref, hyp)
            ser = d / n_ref
            row.append(f"{ser:.4f}")
            if ser < best[1]:
                best = ((lam, K), ser)
        print("  ".join(row), flush=True)
    print(f"\nbest on gold+train: lam={best[0][0]}, K={best[0][1]} (SER {best[1]:.4f}; "
          f"lam=0 row = plain greedy baseline)")


if __name__ == "__main__":
    main()
