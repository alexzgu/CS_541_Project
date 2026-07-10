from pathlib import Path

import numpy as np
import pytest
import torch

from kashi.nn.classifier import LSTMClassifier, PhoneticCrossEntropy
from kashi.nn.segmenter import (
    TransformerSegmenter,
    boundary_f1,
    latent_offset_loss,
    pick_boundaries,
    soft_bce_loss,
    soft_break_labels,
)

REPO = Path(__file__).resolve().parents[1]
LEGACY_M2 = REPO / "models/predict_syllables/pretrained/model_20ms_drop_0.5_144_0.5363_test"


def test_transformer_uses_temporal_context():
    """The legacy bug made every frame independent; the fix must not."""
    torch.manual_seed(0)
    model = TransformerSegmenter(input_dim=8, d_model=16, n_heads=2, num_layers=1,
                                 ff_dim=16, attn_window=50, dropout=0.0).eval()
    x = torch.randn(1, 30, 8)
    with torch.inference_mode():
        base = model(x)
        perturbed = x.clone()
        perturbed[0, 0] += 10.0        # change ONLY frame 0
        out = model(perturbed)
    # frames beyond 0 must feel it through attention
    assert (out[0, 1:] - base[0, 1:]).abs().max() > 1e-4


def test_soft_labels_edge_guarded():
    breaks = np.zeros(6, dtype=bool)
    breaks[0] = True   # would wrap around in the legacy implementation
    y = soft_break_labels(breaks)
    assert y[0] == 1.0 and y[1] == 1.0 and abs(y[2] - 0.8) < 1e-6
    assert y[-1] == 0.0 and y[-2] == 0.0  # no wraparound bleed


def test_losses_finite_and_trainable():
    torch.manual_seed(0)
    logits = torch.randn(200, requires_grad=True)
    breaks = np.array([50, 100, 153])
    valid = torch.ones(200, dtype=torch.bool)
    loss = latent_offset_loss(logits, breaks, valid, delta=3)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    targets = torch.from_numpy(soft_break_labels(np.isin(np.arange(200), breaks)))
    l2 = soft_bce_loss(torch.randn(200), targets, valid, pos_weight=10.0)
    assert torch.isfinite(l2)


def test_latent_offset_prefers_near_miss():
    """A prediction 2 frames off must cost less than one 10 frames off."""
    T = 60
    breaks = np.array([30])
    valid = torch.ones(T, dtype=torch.bool)

    def loss_with_peak(at: int) -> float:
        logits = torch.full((T,), -6.0)
        logits[at] = 6.0
        return float(latent_offset_loss(logits, breaks, valid, delta=3))

    assert loss_with_peak(32) < loss_with_peak(40)
    assert loss_with_peak(30) <= loss_with_peak(32)


def test_pick_boundaries_nms():
    probs = np.zeros(50)
    probs[10] = 0.9
    probs[11] = 0.8   # suppressed by NMS
    probs[30] = 0.6
    assert pick_boundaries(probs, threshold=0.45, nms=3) == [10, 30]


def test_boundary_f1_tolerance():
    s = boundary_f1([10, 30, 52], [11, 29, 70], tol_frames=2)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 3)
    assert s.mean_abs_ms == pytest.approx(20.0)  # (1 + 1) / 2 frames * 20ms


def test_lstm_classifier_shapes_and_phonetic_loss():
    model = LSTMClassifier(input_size=32, hidden_size=16, num_layers=2, dropout=0.0)
    x = torch.randn(4, 12, 32)
    lengths = torch.tensor([12, 9, 5, 2])
    logits = model(x, lengths)
    assert logits.shape == (4, 110)
    crit = PhoneticCrossEntropy(alpha=0.1, power=4)
    loss = crit(logits, torch.tensor([0, 5, 45, 109]))
    assert torch.isfinite(loss)


@pytest.mark.skipif(not LEGACY_M2.exists(), reason="legacy checkpoint not present")
def test_legacy_checkpoint_loads_unchanged():
    model = LSTMClassifier()
    state = torch.load(LEGACY_M2, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # attr names lstm/fc must match exactly
    model.eval()
    with torch.inference_mode():
        out = model(torch.zeros(2, 5, 768), torch.tensor([5, 3]))
    assert out.shape == (2, 110)
