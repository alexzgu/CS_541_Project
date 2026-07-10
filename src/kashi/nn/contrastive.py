"""Contrastive feature learning: the projection head g_w (spec §3.2, §9).

Trained on cached (frozen) encoder features; the trained head then rides
inside the wav2vec2 encoder (encoder.wav2vec2.projection_head), so every
downstream consumer sees the learned space. Training modes land in P5.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"hparams": {"in_dim": self.in_dim, "hidden": self.hidden, "out_dim": self.out_dim},
             "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProjectionHead":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        head = cls(**payload["hparams"])
        head.load_state_dict(payload["state_dict"])
        return head


def info_nce(anchors: torch.Tensor, positives: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Standard InfoNCE; i-th anchor's positive is the i-th positive row,
    everything else in the batch is a negative. Inputs must be L2-normalised."""
    logits = anchors @ positives.T / temperature
    labels = torch.arange(len(anchors), device=anchors.device)
    return F.cross_entropy(logits, labels)
