"""Per-class segment-duration priors (spec §6): shifted negative binomial
d-1 ~ NB(r_u, p_u), method-of-moments with shrinkage to the pooled fit.
`kashi fit durations` writes artifacts/durations.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from ..data import manifest
from ..subtitles import read_csv
from ..tokens import NOISE, SILENCE, TOKEN_INDEX, TOKENS


def _mom_nb(durs: np.ndarray) -> tuple[float, float]:
    """MoM for d-1 ~ NB(r, p); falls back to geometric when var <= mean."""
    x = durs - 1.0
    m, v = float(x.mean()), float(x.var())
    if m <= 0:
        return 1.0, 0.9
    if v <= m + 1e-9:
        p = 1.0 / (1.0 + m)          # geometric limit (r=1)
        return 1.0, p
    p = m / v
    r = m * m / (v - m)
    return float(np.clip(r, 0.05, 500)), float(np.clip(p, 1e-4, 1 - 1e-4))


def fit_durations(cfg, version: str | None = None, min_count: int = 30) -> Path:
    version = version or cfg["data.version"]
    frame_ms = cfg.frame_ms
    per_class: dict[int, list[float]] = {i: [] for i in range(len(TOKENS))}
    for song_id in manifest.labeled_ids(cfg, version):
        for seg in read_csv(manifest.subtitles_dir(cfg, version) / f"{song_id}.csv"):
            if seg.exclude or seg.token == NOISE or seg.token not in TOKEN_INDEX:
                continue
            d = max(1, round(seg.duration * 1000 / frame_ms))
            per_class[TOKEN_INDEX[seg.token]].append(d)
    pooled = np.array([d for i, ds in per_class.items()
                       for d in ds if TOKENS[i] != SILENCE], dtype=float)
    r0, p0 = _mom_nb(pooled)
    out = {}
    for i, ds in per_class.items():
        if TOKENS[i] == SILENCE:
            r, p = _mom_nb(np.array(ds, dtype=float)) if len(ds) >= min_count else (0.5, 0.02)
        elif len(ds) >= min_count:
            r, p = _mom_nb(np.array(ds, dtype=float))
        else:
            r, p = r0, p0
        out[str(i)] = {"r": r, "p": p, "n": len(ds)}
    path = cfg.artifacts_dir / "durations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frame_ms": frame_ms, "classes": out,
                                "pooled": {"r": r0, "p": p0, "n": len(pooled)}}, indent=1))
    print(f"[fit durations] {len(pooled)} lyric segments, pooled NB(r={r0:.2f}, p={p0:.3f}) -> {path}")
    return path


def log_duration_table(cfg, d_max: int) -> np.ndarray:
    """[num_classes, d_max] log P(d | class) for d = 1..d_max (spec §6);
    silence gets a floor so long holds are never impossible."""
    path = cfg.artifacts_dir / "durations.json"
    if not path.is_file():
        raise SystemExit("no duration fits — run `kashi fit durations` first")
    data = json.loads(path.read_text())
    table = np.empty((len(TOKENS), d_max))
    d = np.arange(1, d_max + 1, dtype=float)
    for i in range(len(TOKENS)):
        c = data["classes"][str(i)]
        r, p = c["r"], c["p"]
        k = d - 1
        logpmf = (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
                  + r * np.log(p) + k * np.log1p(-p))
        table[i] = logpmf
    sil = TOKEN_INDEX[SILENCE]
    table[sil] = np.maximum(table[sil], np.log(1e-4 / d_max))
    # normalise over the truncated support so classes are comparable
    table -= np.log(np.exp(table).sum(axis=1, keepdims=True) + 1e-30)
    return table
