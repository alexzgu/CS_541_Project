"""Deleted-vowel insertion: continuation candidates + gap-posterior proposal."""

import numpy as np

from kashi.components.decoders import continuation_candidates, propose_continuations
from kashi.tokens import SILENCE_ID, TOKEN_INDEX

FS = 0.02


def test_continuation_phonotactics():
    assert continuation_candidates("ゆ") == ["う", "ん"]     # う段 -> う
    assert continuation_candidates("ね") == ["い", "え", "ん"]  # え段 -> い/え
    assert continuation_candidates("か") == ["あ", "ん"]
    assert continuation_candidates("ん") == ["ん"]           # no vowel nucleus


def _logp(T, held=None):
    """Mostly-blank posteriors; held=(t0, t1, token_id, p) puts secondary mass there."""
    p = np.full((T, 110), 1e-6)
    p[:, SILENCE_ID] = 0.9
    if held:
        t0, t1, tok, mass = held
        p[t0:t1, tok] = mass
        p[t0:t1, SILENCE_ID] = 0.9 - mass
    return np.log(p / p.sum(1, keepdims=True))


def test_held_vowel_proposed():
    yu, u = TOKEN_INDEX["ゆ"], TOKEN_INDEX["う"]
    logp = _logp(40, held=(8, 30, u, 0.3))     # う mass held in the gap
    ins = propose_continuations(logp, [2, 32], [yu, TOKEN_INDEX["た"]], FS, theta=0.1)
    assert ins and ins[0][1] == u
    assert 8 <= ins[0][0] < 30                  # placed at the posterior peak region


def test_silent_gap_not_proposed():
    yu = TOKEN_INDEX["ゆ"]
    logp = _logp(40)                            # pure blank gap
    assert propose_continuations(logp, [2, 32], [yu, TOKEN_INDEX["た"]], FS, theta=0.1) == []


def test_short_gap_not_proposed():
    yu, u = TOKEN_INDEX["ゆ"], TOKEN_INDEX["う"]
    logp = _logp(20, held=(4, 8, u, 0.4))
    assert propose_continuations(logp, [2, 8], [yu, TOKEN_INDEX["た"]], FS, theta=0.1) == []


def test_illegal_continuation_not_proposed():
    # あ mass after ゆ (う段) is not a legal continuation -> nothing proposed
    yu, a = TOKEN_INDEX["ゆ"], TOKEN_INDEX["あ"]
    logp = _logp(40, held=(8, 30, a, 0.3))
    assert propose_continuations(logp, [2, 32], [yu, TOKEN_INDEX["た"]], FS, theta=0.1) == []
