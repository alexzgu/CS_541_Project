"""Segmenters: features -> syllable spans (two_stage topology).

Interchangeable framings of the same job:
- "energy": training-free voiced-region splitter; the out-of-the-box default.
- "transformer": Model 1 (needs a trained checkpoint) — per-frame break
  probabilities, NMS peak-picked into boundaries.
- "hmm": unsupervised sticky HDP-HMM — boundaries are latent state changes
  (lands in P2).
"""

from __future__ import annotations

import numpy as np

from ..registry import register
from .base import FrameAux, Span


def _voiced_mask_from_rms(rms_db: np.ndarray, top_db: float) -> np.ndarray:
    if len(rms_db) == 0:
        return np.zeros(0, dtype=bool)
    return rms_db > (rms_db.max() - top_db)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


@register("segmenter", "energy")
class EnergySegmenter:
    """Voiced runs from the energy track, long runs split at energy minima."""

    def __init__(self, cfg):
        self.top_db = float(cfg["segmenter.energy.top_db"])
        self.min_frames = max(1, int(cfg["segmenter.energy.min_dur_ms"]) // cfg.frame_ms)
        self.max_frames = max(2, int(cfg["segmenter.energy.max_dur_ms"]) // cfg.frame_ms)

    def segment(self, feats: np.ndarray, aux: FrameAux | None = None) -> list[Span]:
        if aux is not None and aux.rms_db is not None:
            energy = aux.rms_db[: len(feats)]
        else:  # fall back to feature magnitude as an energy proxy
            energy = np.log(np.abs(feats).mean(axis=1) + 1e-9)
        mask = _voiced_mask_from_rms(np.asarray(energy), self.top_db)
        spans: list[Span] = []
        for s, e in _runs(mask):
            if e - s < self.min_frames:
                continue
            # split over-long runs at interior energy minima
            while e - s > self.max_frames:
                window = energy[s + self.min_frames : s + self.max_frames]
                cut = s + self.min_frames + int(np.argmin(window))
                spans.append(Span(s, cut))
                s = cut
            spans.append(Span(s, e))
        return spans


@register("segmenter", "transformer")
class TransformerSegmenterComponent:
    """Model 1 as a segmenter: boundaries from NMS peaks over break probabilities."""

    def __init__(self, cfg):
        ckpt = cfg.get("segmenter.transformer.checkpoint", "")
        if not ckpt:
            raise SystemExit(
                "segmenter 'transformer' needs a trained checkpoint: run "
                "`kashi train segmenter`, then set segmenter.transformer.checkpoint"
            )
        import torch

        from ..nn.segmenter import TransformerSegmenter

        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        hparams = payload.get("hparams", {})
        self.model = TransformerSegmenter(**hparams)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.threshold = float(cfg["segmenter.transformer.threshold"])
        self.nms = int(cfg["segmenter.transformer.nms_frames"])
        self.min_frames = 1

    def segment(self, feats: np.ndarray, aux: FrameAux | None = None) -> list[Span]:
        import torch

        from ..nn.segmenter import pick_boundaries

        with torch.inference_mode():
            probs = torch.sigmoid(
                self.model(torch.from_numpy(feats)[None].float())[0]
            ).numpy()
        if aux is not None:
            aux.boundary_logits = np.log(probs + 1e-9) - np.log1p(-probs + 1e-9)
        bounds = pick_boundaries(probs, threshold=self.threshold, nms=self.nms)
        edges = [0, *bounds, len(feats)]
        return [Span(s, e) for s, e in zip(edges[:-1], edges[1:]) if e - s >= self.min_frames]


@register("segmenter", "hmm")
class HMMSegmenterComponent:
    def __init__(self, cfg):
        raise SystemExit("segmenter 'hmm' lands in P2 (sticky HDP-HMM) — see ROADMAP.md")
