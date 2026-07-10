"""Model 1: per-frame syllable-break scorer (transformer encoder), corrected.

Fixes vs the recovered legacy notebook (Transformer_Wave2Vec_AAAAA.ipynb):
batch_first=True (attention actually runs over time), sinusoidal positional
encodings, windowed attention (±attn_window frames), edge-guarded label
expansion. Two losses (spec §4.2): the report's soft-kernel BCE and the
latent-offset marginalized likelihood (exact under the ±50 ms jitter model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The report's asymmetric label kernel on offsets -2..+3.
LABEL_KERNEL = {-2: 0.2, -1: 0.35, 0: 1.0, 1: 1.0, 2: 0.8, 3: 0.25}


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int = 40000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, D]
        return x + self.pe[: x.size(1)].to(x.dtype)


class TransformerSegmenter(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 512,
        attn_window: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hparams = dict(
            input_dim=input_dim, d_model=d_model, n_heads=n_heads,
            num_layers=num_layers, ff_dim=ff_dim, attn_window=attn_window,
            dropout=dropout,
        )
        self.in_proj = nn.Linear(input_dim, d_model)
        self.positions = SinusoidalPositions(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            activation="relu", dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)
        self.attn_window = attn_window

    def _window_mask(self, T: int, device) -> torch.Tensor | None:
        if self.attn_window <= 0 or T <= self.attn_window:
            return None
        idx = torch.arange(T, device=device)
        return (idx[None, :] - idx[:, None]).abs() > self.attn_window  # True = masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim] -> per-frame break logits [B, T]."""
        h = self.positions(self.in_proj(x))
        mask = self._window_mask(h.size(1), h.device)
        h = self.encoder(h, mask=mask)
        return self.out(h).squeeze(-1)

    @torch.inference_mode()
    def frame_logits(self, feats: np.ndarray, chunk: int = 2000, overlap: int = 200) -> np.ndarray:
        """Full-song inference with overlapped chunks, seams averaged."""
        T = len(feats)
        out = np.zeros(T, dtype=np.float64)
        weight = np.zeros(T, dtype=np.float64)
        step = chunk - overlap
        device = next(self.parameters()).device
        for s in range(0, max(1, T), step):
            piece = feats[s : s + chunk]
            if len(piece) == 0:
                break
            logits = self(torch.from_numpy(piece)[None].float().to(device))[0].cpu().numpy()
            out[s : s + len(piece)] += logits
            weight[s : s + len(piece)] += 1.0
            if s + chunk >= T:
                break
        weight[weight == 0] = 1.0
        return (out / weight).astype(np.float32)


# ---------------------------------------------------------------------------
# Labels and losses
# ---------------------------------------------------------------------------

def soft_break_labels(breaks: np.ndarray) -> np.ndarray:
    """Edge-guarded placement of the report's kernel around each break frame."""
    T = len(breaks)
    y = np.zeros(T, dtype=np.float32)
    for b in np.flatnonzero(breaks):
        for off, val in LABEL_KERNEL.items():
            t = b + off
            if 0 <= t < T:
                y[t] = min(1.0, y[t] + val)
    return y


def soft_bce_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Masked, positively-weighted BCE against the soft targets. All [T] tensors."""
    logp = F.logsigmoid(logits)
    log1mp = F.logsigmoid(-logits)
    per = -(pos_weight * soft_targets * logp + (1 - soft_targets) * log1mp)
    m = valid.float()
    return (per * m).sum() / m.sum().clamp(min=1.0)


def _truncated_windows(break_frames: np.ndarray, delta: int, T: int) -> list[tuple[int, int, int]]:
    """(lo, hi, b) per break with overlaps resolved at midpoints; [lo, hi] inclusive."""
    out = []
    bs = np.sort(break_frames)
    for i, b in enumerate(bs):
        lo = max(0, b - delta)
        hi = min(T - 1, b + delta)
        if i > 0:
            lo = max(lo, (bs[i - 1] + b) // 2 + 1)
        if i + 1 < len(bs):
            hi = min(hi, (b + bs[i + 1]) // 2)
        if lo <= b <= hi:
            out.append((int(lo), int(hi), int(b)))
    return out


def latent_offset_loss(
    logits: torch.Tensor,
    break_frames: np.ndarray,
    valid: torch.Tensor,
    delta: int = 3,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """Exact marginal likelihood under the jitter model (spec eq. (3)).

    Each labeled break's true position is a latent offset within ±delta frames
    (triangular prior); windows are truncated at midpoints when breaks crowd.
    """
    T = logits.shape[-1]
    logp = F.logsigmoid(logits)      # log y_t
    log1mp = F.logsigmoid(-logits)   # log (1 - y_t)

    windows = _truncated_windows(np.asarray(break_frames, dtype=int), delta, T)
    in_window = torch.zeros(T, dtype=torch.bool, device=logits.device)
    total = logits.sum() * 0.0
    for lo, hi, b in windows:
        idx = torch.arange(lo, hi + 1, device=logits.device)
        in_window[idx] = True
        prior = (delta + 1 - (idx - b).abs()).clamp(min=1).float()
        log_prior = torch.log(prior / prior.sum())
        s_window = log1mp[idx].sum()
        terms = log_prior + logp[idx] + s_window - log1mp[idx]
        total = total - torch.logsumexp(terms, dim=0)

    bg = valid & ~in_window
    total = total - neg_weight * (log1mp * bg.float()).sum()
    denom = max(1, len(windows)) + bg.float().sum().clamp(min=1.0)
    return total / denom


# ---------------------------------------------------------------------------
# Decoding and the tolerance metric
# ---------------------------------------------------------------------------

def pick_boundaries(probs: np.ndarray, threshold: float = 0.45, nms: int = 3) -> list[int]:
    """Threshold + non-maximum suppression -> boundary frame indices."""
    out: list[int] = []
    T = len(probs)
    for t in range(T):
        if probs[t] < threshold:
            continue
        lo, hi = max(0, t - nms), min(T, t + nms + 1)
        if probs[t] >= probs[lo:hi].max():
            if not out or t - out[-1] > nms:
                out.append(t)
    return out


@dataclass
class BoundaryScore:
    precision: float
    recall: float
    f1: float
    mean_abs_ms: float
    n_pred: int
    n_true: int


def boundary_f1(
    pred: list[int], true: list[int], tol_frames: int, frame_ms: int = 20
) -> BoundaryScore:
    """Greedy monotone 1-1 matching within ±tol_frames."""
    i = j = hits = 0
    abs_err: list[int] = []
    pred = sorted(pred)
    true = sorted(true)
    while i < len(pred) and j < len(true):
        d = pred[i] - true[j]
        if abs(d) <= tol_frames:
            hits += 1
            abs_err.append(abs(d))
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_abs = float(np.mean(abs_err) * frame_ms) if abs_err else float("nan")
    return BoundaryScore(precision, recall, f1, mean_abs, len(pred), len(true))
