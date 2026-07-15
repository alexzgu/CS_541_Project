"""spikes_to_crops: grouping/filtering of greedy-CTC spikes into pseudo crops."""

import numpy as np

from kashi.tokens import SILENCE_ID
from kashi.train.pseudo import spikes_to_crops

FS = 0.02  # 20 ms frames
B = SILENCE_ID


def _path(spec):
    """spec = [(class, n_frames), ...] -> (path, probs) with prob 0.9 on spikes."""
    path = np.concatenate([np.full(n, c, dtype=int) for c, n in spec])
    probs = np.full(len(path), 0.9)
    return path, probs


def _tokens(ids, spike_f=1, gap_f=9):
    """Realistic singing density: one spike every (spike_f+gap_f) frames (~5 tok/s)."""
    spec = []
    for c in ids:
        spec += [(c, spike_f), (B, gap_f)]
    return spec


def test_cut_at_long_blank_gap():
    # 5 tokens, a 0.6 s blank gap, 5 more tokens -> two crops
    spec = [(B, 10)] + _tokens(range(5)) + [(B, 30)] + _tokens(range(5, 10)) + [(B, 10)]
    path, probs = _path(spec)
    crops = spikes_to_crops(path, probs, FS, blank_id=B)
    assert len(crops) == 2
    assert crops[0]["tokens"] == [0, 1, 2, 3, 4]
    assert crops[1]["tokens"] == [5, 6, 7, 8, 9]
    # first crop ends inside the separating blank run, before the second starts
    assert crops[0]["t1"] <= crops[1]["t0"] + 1e-6


def test_short_blank_gap_does_not_cut():
    spec = _tokens(range(4)) + [(B, 10)] + _tokens(range(4, 8))  # extra 0.2 s gap only
    path, probs = _path(spec)
    crops = spikes_to_crops(path, probs, FS, blank_id=B)
    assert len(crops) == 1
    assert crops[0]["tokens"] == list(range(8))


def test_min_tokens_filter():
    spec = [(B, 5)] + _tokens([1, 2]) + [(B, 40)] + _tokens([3, 4, 5, 6]) + [(B, 5)]
    path, probs = _path(spec)
    crops = spikes_to_crops(path, probs, FS, blank_id=B, min_tokens=4)
    assert len(crops) == 1  # the 2-token group is dropped
    assert crops[0]["tokens"] == [3, 4, 5, 6]


def test_overflow_split_at_crop_s():
    # 30 tokens, 0.3 s apart (no long blanks) -> must split around crop_s=3 s
    spec = []
    for i in range(30):
        spec += [(i % 20 + 1, 1), (B, 14)]
    path, probs = _path(spec)
    crops = spikes_to_crops(path, probs, FS, blank_id=B, crop_s=3.0)
    assert len(crops) >= 3
    assert sum(len(c["tokens"]) for c in crops) == 30
    # +0.1 s leading pad, +0.5 s trailing pad on the final group
    assert all(c["dur_s"] <= 3.0 + 0.7 for c in crops)


def test_rate_filter_drops_hallucination_bursts():
    # 20 tokens in ~0.8 s = ~25 tok/s -> dropped by max_rate
    spec = [(B, 5)] + [(i % 20 + 1, 2) for i in range(20)] + [(B, 5)]
    path, probs = _path(spec)
    crops = spikes_to_crops(path, probs, FS, blank_id=B, max_rate=12.0)
    assert crops == []


def test_confidence_stats_recorded():
    spec = [(B, 5)] + _tokens([1, 2, 3, 4, 5]) + [(B, 5)]
    path, probs = _path(spec)
    probs[:] = 0.5
    crops = spikes_to_crops(path, probs, FS, blank_id=B)
    assert len(crops) == 1
    assert abs(crops[0]["conf_mean"] - 0.5) < 1e-6
    assert abs(crops[0]["conf_min"] - 0.5) < 1e-6
