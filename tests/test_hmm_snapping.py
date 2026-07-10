import numpy as np

from kashi.stats.hmm import PCA, StickyHDPHMM
from kashi.stats.snapping import snap_events


def test_snapping_monotone_and_bounded():
    events = [1.0, 2.0, 3.0]
    candidates = [
        {"time_s": 1.04, "prob": 0.9, "std_ms": 10},
        {"time_s": 2.5, "prob": 0.9, "std_ms": 10},   # too far from 2.0 (>100ms)
        {"time_s": 2.97, "prob": 0.8, "std_ms": 10},
    ]
    out = snap_events(events, candidates, delta_max_s=0.1)
    assert out[0]["matched"] and abs(out[0]["time_s"] - 1.04) < 1e-9
    assert not out[1]["matched"] and out[1]["time_s"] == 2.0   # unmatched -> unchanged
    assert out[2]["matched"] and abs(out[2]["time_s"] - 2.97) < 1e-9
    times = [o["time_s"] for o in out]
    assert times == sorted(times)                               # order preserved


def test_snapping_prefers_confident_candidate():
    events = [1.0]
    candidates = [
        {"time_s": 0.95, "prob": 0.1, "std_ms": 10},
        {"time_s": 1.06, "prob": 0.9, "std_ms": 10},
    ]
    out = snap_events(events, candidates, delta_max_s=0.1, eta=0.5)
    assert abs(out[0]["time_s"] - 1.06) < 1e-9  # slightly farther but much more confident


def test_hmm_recovers_synthetic_boundaries():
    rng = np.random.default_rng(0)
    means = rng.normal(0, 3, size=(4, 6))
    z_true, X = [], []
    state = 0
    for seg in range(10):
        state = (state + 1 + rng.integers(0, 3)) % 4
        dur = 25 + int(rng.integers(0, 10))
        z_true += [state] * dur
        X.append(means[state] + rng.normal(0, 0.5, size=(dur, 6)))
    X = np.concatenate(X)
    true_bounds = [t for t in range(1, len(z_true)) if z_true[t] != z_true[t - 1]]

    hmm = StickyHDPHMM(L=12, alpha=2, gamma=2, rho=0.9, sweeps=10, burnin=5, seed=1)
    res = hmm.fit(X, min_prob=0.3)
    found = [b["frame"] for b in res.boundaries]
    hits = sum(1 for tb in true_bounds if any(abs(tb - f) <= 2 for f in found))
    assert hits / len(true_bounds) >= 0.8           # recall
    assert len(found) <= len(true_bounds) * 2       # not shattering
    assert res.n_active_states >= 3


def test_pca_shapes():
    X = np.random.default_rng(0).normal(size=(500, 20))
    Z = PCA(5).fit(X).transform(X)
    assert Z.shape == (500, 5)
    assert abs(Z.std(axis=0).mean() - 1.0) < 0.2    # whitened
