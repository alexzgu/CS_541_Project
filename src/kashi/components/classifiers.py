"""Per-span syllable classifiers (two_stage topology)."""

from __future__ import annotations

import numpy as np

from ..registry import register
from ..tokens import SILENCE, TOKENS
from .base import Span


@register("classifier", "silence_only")
class SilenceOnlyClassifier:
    """Labels everything <silence>; wiring/debug baseline."""

    def __init__(self, cfg):
        pass

    def classify(self, feats: np.ndarray, spans: list[Span]) -> list[tuple[str, float]]:
        return [(SILENCE, 1.0) for _ in spans]


@register("classifier", "lstm")
class LSTMClassifierComponent:
    """Model 2. Loads legacy checkpoints unchanged (attrs `lstm`, `fc`;
    class order = kashi.tokens.TOKENS)."""

    def __init__(self, cfg):
        import torch

        from ..nn.classifier import LSTMClassifier

        ckpt = cfg.get("classifier.lstm.checkpoint", "")
        self.model = LSTMClassifier(
            input_size=int(cfg["classifier.lstm.input_size"]),
            hidden_size=int(cfg["classifier.lstm.hidden_size"]),
            num_layers=int(cfg["classifier.lstm.num_layers"]),
            dropout=float(cfg["classifier.lstm.dropout"]),
        )
        if ckpt:
            path = cfg.root / ckpt if not str(ckpt).startswith("/") else ckpt
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
        self.model.eval()

    def classify(self, feats: np.ndarray, spans: list[Span]) -> list[tuple[str, float]]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        if not spans:
            return []
        pieces = [torch.from_numpy(feats[s.start : s.end]).float() for s in spans]
        lengths = torch.tensor([max(1, len(p)) for p in pieces])
        order = torch.argsort(lengths, descending=True)
        padded = pad_sequence([pieces[i] for i in order], batch_first=True)
        with torch.inference_mode():
            logits = self.model(padded, lengths[order])
            probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        out: list[tuple[str, float]] = [("", 0.0)] * len(spans)
        for rank, orig in enumerate(order.tolist()):
            out[orig] = (TOKENS[int(pred[rank])], float(conf[rank]))
        return out
