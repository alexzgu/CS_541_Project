"""Unsupervised boundary-candidate sources for `kashi realign` (spec §8.1).

Union of: sticky HDP-HMM posterior boundaries, spectral-flux onsets, voicing
transitions. Each source yields (time_s, prob, std_ms); merged within 1 frame.
HMM results are cached per (song, feature-cache) under artifacts/boundaries/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def hmm_boundaries(cfg, feats: np.ndarray, cache_key: str | None = None) -> list[dict]:
    from ..stats.hmm import PCA, StickyHDPHMM

    cache = None
    if cache_key:
        cache = cfg.artifacts_dir / "boundaries" / f"{cache_key}.json"
        if cache.is_file():
            return json.loads(cache.read_text())
    X = PCA(int(cfg["segmenter.hmm.pca_dim"])).fit(feats).transform(feats)
    hmm = StickyHDPHMM(
        L=int(cfg["segmenter.hmm.L"]),
        alpha=float(cfg["segmenter.hmm.alpha"]),
        gamma=float(cfg["segmenter.hmm.gamma"]),
        rho=float(cfg["segmenter.hmm.rho"]),
        sweeps=int(cfg["segmenter.hmm.sweeps"]),
        burnin=int(cfg["segmenter.hmm.burnin"]),
        seed=int(cfg["train.seed"]),
        temperature=float(cfg.get("segmenter.hmm.temperature", 1.0)),
    )
    res = hmm.fit(X, min_prob=float(cfg["boundaries.hmm_p_min"]))
    frame_s = cfg.frame_ms / 1000.0
    out = [
        {"time_s": b["frame"] * frame_s, "prob": b["prob"],
         "std_ms": b["std_frames"] * cfg.frame_ms, "source": "hmm"}
        for b in res.boundaries
    ]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


def onset_boundaries(cfg, wave: np.ndarray, sr: int) -> list[dict]:
    """Spectral-flux onsets: positive per-band magnitude increases, normalised,
    NMS peak-picked."""
    import torch
    import torchaudio

    hop = int(sr * cfg.frame_ms / 1000)
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64, hop_length=hop)(
        torch.from_numpy(wave).float()
    ).numpy()  # [mel, T+1]
    mag = np.log(spec + 1e-9)
    flux = np.maximum(0.0, mag[:, 1:] - mag[:, :-1]).sum(0)  # [T]
    if len(flux) == 0:
        return []
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-9
    score = (flux - med) / (3 * mad)
    frame_s = cfg.frame_ms / 1000.0
    out = []
    for t in range(1, len(score) - 1):
        if score[t] > 1.0 and score[t] >= score[t - 1] and score[t] >= score[t + 1]:
            out.append({"time_s": (t + 1) * frame_s,
                        "prob": float(min(1.0, score[t] / 3)),
                        "std_ms": float(cfg.frame_ms), "source": "onset"})
    return out


def voicing_boundaries(cfg, voicing: np.ndarray) -> list[dict]:
    """Extrema of |Δ voicing| — voiced/unvoiced transitions."""
    dv = np.abs(np.diff(voicing))
    frame_s = cfg.frame_ms / 1000.0
    out = []
    for t in range(1, len(dv) - 1):
        if dv[t] > 0.35 and dv[t] >= dv[t - 1] and dv[t] >= dv[t + 1]:
            out.append({"time_s": (t + 1) * frame_s, "prob": float(min(1.0, dv[t])),
                        "std_ms": float(cfg.frame_ms), "source": "voicing"})
    return out


def merge_candidates(sources: list[list[dict]], merge_within_s: float = 0.02) -> list[dict]:
    """Union of sources; events within one frame merge (max prob wins, sources joined)."""
    allc = sorted((c for src in sources for c in src), key=lambda c: c["time_s"])
    out: list[dict] = []
    for c in allc:
        if out and c["time_s"] - out[-1]["time_s"] <= merge_within_s:
            best = out[-1]
            if c["prob"] > best["prob"]:
                c = dict(c, source=f"{best['source']}+{c['source']}")
                out[-1] = c
            else:
                best["source"] += "+" + c["source"]
        else:
            out.append(dict(c))
    return out
