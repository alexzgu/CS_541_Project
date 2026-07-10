"""Token bigram over label sequences (spec §6). Add-k smoothed; a prior over
token strings learned offline — not a transcript; decoding stays textless.
Off by default (decoder.segmental.lambda_lm = 0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data import manifest
from ..subtitles import read_csv
from ..tokens import NOISE, TOKEN_INDEX, TOKENS


def fit_bigram(cfg, version: str | None = None, add_k: float = 0.1) -> Path:
    version = version or cfg["data.version"]
    V = len(TOKENS)
    counts = np.zeros((V, V))
    for song_id in manifest.labeled_ids(cfg, version):
        prev = None
        for seg in read_csv(manifest.subtitles_dir(cfg, version) / f"{song_id}.csv"):
            if seg.exclude or seg.token == NOISE or seg.token not in TOKEN_INDEX:
                prev = None
                continue
            cur = TOKEN_INDEX[seg.token]
            if prev is not None:
                counts[prev, cur] += 1
            prev = cur
    probs = (counts + add_k) / (counts.sum(1, keepdims=True) + add_k * V)
    path = cfg.artifacts_dir / "lm_bigram.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, log_bigram=np.log(probs))
    print(f"[fit lm] {int(counts.sum())} transitions -> {path}")
    return path


def log_bigram(cfg) -> np.ndarray | None:
    path = cfg.artifacts_dir / "lm_bigram.npz"
    return np.load(path)["log_bigram"] if path.is_file() else None
