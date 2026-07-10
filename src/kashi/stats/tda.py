"""Voicing (periodicity) score per analysis frame (spec §3.3).

Voiced sounds are quasi-periodic at the fundamental; breath and silence are
aperiodic. Two interchangeable estimators of the same quantity:
- "autocorr" (default): peak of the normalised autocorrelation over plausible
  F0 lags — cheap, robust, no extra deps.
- "tda": longest H1 persistence bar of a Vietoris-Rips filtration on the
  time-delay embedding (requires the [tda] extra / ripser).
"""

from __future__ import annotations

import numpy as np


def autocorr_periodicity(
    window: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 500.0
) -> float:
    """Max normalised autocorrelation over lags for F0 in [fmin, fmax]."""
    w = window.astype(np.float64)
    w = w - w.mean()
    n = len(w)
    lo = int(sr / fmax)
    hi = min(n - 1, int(sr / fmin))
    if hi <= lo or n < 2 * lo or not np.any(w):
        return 0.0
    # FFT autocorrelation
    f = np.fft.rfft(w, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    denom = ac[0]
    if denom <= 0:
        return 0.0
    return float(np.clip(np.max(ac[lo : hi + 1] / denom), 0.0, 1.0))


def h1_periodicity(
    window: np.ndarray, sr: int, fmax: float = 500.0, m: int = 3, max_points: int = 160
) -> float:
    """Longest H1 bar of the delay embedding, normalised to [0, 1]."""
    try:
        from ripser import ripser
    except ImportError as e:
        raise SystemExit("voicing 'tda' needs the [tda] extra (ripser)") from e
    w = window.astype(np.float64)
    w = w - w.mean()
    tau = max(1, int(sr / (2 * fmax)))
    n = len(w) - (m - 1) * tau
    if n < 8:
        return 0.0
    emb = np.stack([w[i * tau : i * tau + n] for i in range(m)], axis=1)
    if len(emb) > max_points:
        emb = emb[:: len(emb) // max_points][:max_points]
    diam = float(np.linalg.norm(emb.max(0) - emb.min(0)))
    if diam <= 0:
        return 0.0
    dgm = ripser(emb, maxdim=1)["dgms"][1]
    if len(dgm) == 0:
        return 0.0
    life = np.nan_to_num(dgm[:, 1] - dgm[:, 0], posinf=0.0)
    return float(np.clip(life.max() / diam, 0.0, 1.0))


def voicing_track(
    wave: np.ndarray,
    sr: int,
    frame_ms: int,
    method: str = "autocorr",
    window_ms: float = 46.0,
    fmin: float = 70.0,
    fmax: float = 500.0,
) -> np.ndarray:
    """[T] voicing score per frame, aligned with the encoder's frame grid."""
    hop = int(sr * frame_ms / 1000)
    half = int(sr * window_ms / 2000)
    T = len(wave) // hop
    est = autocorr_periodicity if method == "autocorr" else h1_periodicity
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        c = t * hop + hop // 2
        piece = wave[max(0, c - half) : c + half]
        if len(piece) >= 4:
            if method == "autocorr":
                out[t] = est(piece, sr, fmin=fmin, fmax=fmax)
            else:
                out[t] = est(piece, sr, fmax=fmax)
    return out
