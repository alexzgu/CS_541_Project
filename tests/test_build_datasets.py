import numpy as np
import pandas as pd
import pytest

from kashi.data import build
from kashi.data.datasets import _frame, song_frame_labels
from kashi.subtitles import Segment
from kashi.tokens import SILENCE


def _raw(rows):
    return pd.DataFrame(rows, columns=["start", "end", "line", "unformatted", "token"])


def test_clean_song_hemisphere_and_silence():
    # top hemisphere (line=0) = kana readings, bottom (line=-1) = kanji, overlapping
    raw = _raw([
        [1.0, 1.5, 0, "かたな", "か"],
        [1.0, 1.5, -1, "刀", "刀"],
        [1.5, 2.0, 0, "かたな", "た"],
        [1.5, 2.0, -1, "刀", "刀"],
        [3.0, 3.5, 0, "かたな", "な"],
    ])
    out = build.clean_song(raw, ignore=[])
    kept = out[~out["exclude"]]["token"].tolist()
    # kanji hemisphere dropped, silences inserted at gaps
    assert kept[0] == SILENCE            # 0 -> 1.0
    assert "か" in kept and "た" in kept and "な" in kept
    assert "刀" not in out["token"].tolist()
    sil_rows = out[out["token"] == SILENCE]
    assert len(sil_rows) >= 3            # leading, 2.0-3.0 gap, trailing


def test_clean_song_marks_exclusions_and_token_set():
    raw = _raw([
        [0.5, 1.0, 0, "かた", "か"],
        [1.0, 1.5, 0, "かた", "XYZ"],    # not in the 110-token set
        [2.0, 2.5, 0, "かた", "た"],
    ])
    out = build.clean_song(raw, ignore=[build.TimeRange(2.0, 2.6)])
    xyz = out[out["token"] == "XYZ"]
    assert xyz["exclude"].all()
    ta = out[out["token"] == "た"]
    assert ta["exclude"].all()           # covered by an ignore range
    ka = out[out["token"] == "か"]
    assert not ka["exclude"].any()


def test_clean_song_long_vowel_merge():
    raw = _raw([
        [0.0, 0.4, 0, "かー", "か"],
        [0.4, 0.8, 0, "かー", "ー"],     # long-vowel mark merges into previous row
        [0.8, 1.2, 0, "かーた", "た"],
    ])
    out = build.clean_song(raw, ignore=[])
    ka = out[out["token"] == "か"].iloc[0]
    assert ka["end"] == pytest.approx(0.8)


def test_trim_length_pulls_in_trailing_silence():
    sr = 16000
    voiced = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sr // 2) / sr)
    seg = np.concatenate([voiced, np.zeros(sr // 2)])  # 0.5 s tone + 0.5 s silence
    trimmed = build._trim_length(seg.astype(np.float32), top_db=20)
    assert trimmed < len(seg)
    assert trimmed >= len(voiced) * 0.8


def test_frame_rounding_robust_to_float_ms():
    assert _frame(12.72, 20) == 636      # 12.72*1000 = 12719.999... must not floor to 635
    assert _frame(0.0, 20) == 0
    assert _frame(1.0, 20) == 50


def test_song_frame_labels_masks_excluded():
    segs = [
        Segment(0.0, 1.0, SILENCE),
        Segment(1.0, 1.2, "か"),
        Segment(1.2, 1.4, "た", exclude=True),
        Segment(1.4, 2.0, SILENCE),
    ]
    breaks, valid = song_frame_labels(segs, 20, 100)
    assert breaks[50] and breaks[60] and breaks[70]
    assert not valid[60:70].any()        # excluded row masked
    assert valid[:50].all()
