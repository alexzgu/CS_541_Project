"""Model 2: per-segment syllable classifier (LSTM) + phonetic partial-credit loss.

Attribute names (`lstm`, `fc`) are identical to the legacy
models/predict_syllables/model.py so old checkpoints (e.g.
model_20ms_drop_0.5_144_0.5363_test) load without conversion.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 144,
        num_layers: int = 2,
        num_classes: int = 110,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] padded, sorted by length desc; lengths on CPU.
        Returns [B, num_classes] logits from the top layer's final hidden state."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n[-1])


class FrameClassifier(nn.Module):
    """Model 2f (spec §5.2): per-frame posterior over the 110 classes. Additive
    span scores decompose over frames, enabling the exact semi-Markov DP."""

    def __init__(self, input_size: int = 768, hidden: int = 256, num_classes: int = 110):
        super().__init__()
        self.hparams = dict(input_size=input_size, hidden=hidden, num_classes=num_classes)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [.., D] -> [.., C] logits
        return self.net(x)

    @torch.inference_mode()
    def log_probs(self, feats: np.ndarray, batch: int = 8192) -> np.ndarray:
        """[T, C] log posteriors."""
        device = next(self.parameters()).device
        out = []
        for s in range(0, len(feats), batch):
            x = torch.from_numpy(feats[s : s + batch]).float().to(device)
            out.append(torch.log_softmax(self(x), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 110), dtype=np.float32)


class PhoneticCrossEntropy(nn.Module):
    """Cross-entropy against phonetic soft targets (spec §5.3):
    q(u'|u) = (1-alpha)·1[u'=u] + alpha·(k^power renormalised off-diagonal)."""

    def __init__(self, alpha: float = 0.1, power: int = 4):
        super().__init__()
        from ..phonetics import soft_targets

        Q = soft_targets(alpha=alpha, power=power)
        self.register_buffer("targets", torch.from_numpy(np.asarray(Q, dtype=np.float32)))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=-1)
        q = self.targets[labels]
        return -(q * logp).sum(dim=-1).mean()
