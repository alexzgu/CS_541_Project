"""Segmental decoder DP on synthetic posteriors: planted segments must be
recovered exactly when the evidence is clean."""

import numpy as np
import pytest

from kashi.components.decoders import SegmentalDecoder
from kashi.tokens import SILENCE_ID


def _bare_decoder(d_max=20, lam_d=0.3, lam_lm=0.0, log_A=None):
    dec = SegmentalDecoder.__new__(SegmentalDecoder)  # skip __init__ (no ckpt)
    dec.d_max = d_max
    dec.lam_c, dec.lam_d, dec.lam_b, dec.lam_lm = 1.0, lam_d, 0.0, lam_lm
    # flat-ish NB duration table peaked around 8 frames
    d = np.arange(1, d_max + 1)
    logdur = -0.5 * ((d - 8) / 6.0) ** 2
    dec.log_dur = np.tile(logdur - np.log(np.exp(logdur).sum()), (110, 1))
    dec.log_A = log_A
    return dec


def test_viterbi_recovers_planted_segments():
    rng = np.random.default_rng(0)
    plan = [(0, 10, 5), (10, 18, 40), (18, 30, 7), (30, 42, 5)]
    n = 42
    logp = np.full((n, 110), np.log(0.3 / 109))
    for s, e, u in plan:
        logp[s:e, u] = np.log(0.7)
    logp += rng.normal(0, 0.01, size=logp.shape)

    dec = _bare_decoder()
    segs = dec._viterbi_chunk(logp, beta=np.zeros(n))
    assert [(s, e) for s, e, _ in segs] == [(s, e) for s, e, _ in plan]
    assert [u for *_, u in segs] == [u for *_, u in plan]


def test_viterbi_covers_and_prefers_fewer_segments():
    """Uniform emissions: every duration term is a penalty, so the DP covers
    [0, n) exactly with as few segments as d_max allows."""
    n = 32
    logp = np.zeros((n, 110)) - np.log(110)
    dec = _bare_decoder(lam_d=1.0, d_max=20)
    segs = dec._viterbi_chunk(logp, beta=np.zeros(n))
    assert sum(e - s for s, e, _ in segs) == n
    assert all(e - s <= 20 for s, e, _ in segs)
    assert len(segs) == 2  # ceil(32/20)


def test_viterbi_with_bigram():
    """A strong bigram should flip an ambiguous middle segment."""
    n = 24
    logp = np.full((n, 110), -np.log(110))
    logp[0:8, 1] = np.log(0.8)
    logp[16:24, 3] = np.log(0.8)
    logp[8:16, 2] = np.log(0.02)   # weak evidence for class 2
    logp[8:16, 4] = np.log(0.02)   # equally weak for class 4
    A = np.full((110, 110), np.log(1e-4))
    A[1, 2] = A[2, 3] = np.log(0.9)   # bigram prefers 1 -> 2 -> 3
    dec = _bare_decoder(lam_lm=1.0, log_A=A)
    segs = dec._viterbi_chunk(logp, beta=np.zeros(n))
    labels = [u for *_, u in segs]
    assert 2 in labels and 4 not in labels
    assert labels[0] == 1 and labels[-1] == 3
