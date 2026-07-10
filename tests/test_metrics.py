import pytest

from kashi.eval import metrics as M
from kashi.subtitles import Segment
from kashi.tokens import NOISE, SILENCE


def segs(rows):
    return [Segment(s, e, t, exclude=x) for s, e, t, x in rows]


REF = segs([
    (0.0, 1.0, SILENCE, False),
    (1.0, 1.2, "か", False),
    (1.2, 1.5, "た", False),
    (1.5, 2.0, SILENCE, False),
    (2.0, 2.2, "す", False),
])


def test_ser_exact_and_edits():
    assert M.ser(REF, REF)[0] == 0.0
    pred = segs([(1.0, 1.2, "か", False), (1.2, 1.5, "な", False), (2.0, 2.2, "す", False)])
    s, d, n = M.ser(pred, REF)                      # 1 substitution / 3 ref tokens
    assert (s, d, n) == (pytest.approx(1 / 3), 1, 3)
    assert M.ser([], REF)[0] == 1.0                  # all deletions


def test_ser_ignores_silence_noise_excluded():
    pred = segs([
        (0.0, 1.0, SILENCE, False), (0.5, 0.7, NOISE, False),
        (1.0, 1.2, "か", False), (1.2, 1.5, "た", True),  # excluded
    ])
    assert M.lyric_tokens(pred) == ["か"]


def test_boundary_metrics_tolerance():
    ref = [1.0, 2.0, 3.0]
    pred = [1.03, 2.2, 2.98]
    bm = M.boundary_metrics(pred, ref, tol_s=0.05)
    assert bm.f1 == pytest.approx(2 / 3)
    assert bm.mean_abs_ms == pytest.approx(25.0, abs=1e-6)


def test_timed_token_metrics():
    pred = segs([(1.01, 1.2, "が", False), (1.2, 1.5, "た", False), (2.3, 2.5, "す", False)])
    tm = M.timed_token_metrics(pred, REF, tol_s=0.05)
    # か@1.0 time-matched with が@1.01 (wrong token, high partial credit);
    # た matched exactly; す too far (2.3 vs 2.0)
    assert tm.n_time_matched == 2
    assert tm.precision == pytest.approx(1 / 3)
    assert tm.recall == pytest.approx(1 / 3)
    assert tm.partial_credit > 0.9


def test_noise_span_pr():
    ref = segs([(1.0, 1.5, NOISE, False), (3.0, 3.2, NOISE, False)])
    pred = segs([(1.1, 1.5, NOISE, False), (5.0, 5.2, NOISE, False)])
    out = M.noise_span_pr(pred, ref)
    assert out["precision"] == 0.5 and out["recall"] == 0.5


def test_clip_to_window():
    out = M.clip_to_window(REF, 1.1, 1.9)
    assert [s.token for s in out] == ["か", "た", SILENCE]
    assert out[0].start == 1.1 and out[-1].end == 1.9
