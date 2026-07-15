"""Cross-cover agreement filter: n-gram support logic (S15)."""

from kashi.train.covers import crop_support, grams_of, support_index


def test_grams():
    assert grams_of([1, 2, 3, 4]) == {(1, 2, 3), (2, 3, 4)}
    assert grams_of([1, 2]) == set()


def test_shared_lyrics_supported():
    # three covers sing the same line; one has a private intro
    line = [5, 6, 7, 8, 9, 10]
    seqs = {"a": [99, 98, 97] + line, "b": line, "c": line + [50, 51]}
    idx = support_index(seqs)
    assert crop_support(line, "a", idx) == 1.0          # both others reproduce it
    assert crop_support([99, 98, 97], "a", idx) == 0.0   # intro: no support
    assert crop_support([50, 51, 9], "c", idx) == 0.0    # outro fragment


def test_support_excludes_self():
    seqs = {"a": [1, 2, 3, 4], "b": [7, 8, 9]}
    idx = support_index(seqs)
    # only cover "a" has (1,2,3) — its own occurrence must not count
    assert crop_support([1, 2, 3, 4], "a", idx) == 0.0


def test_partial_support():
    line = [5, 6, 7, 8]
    seqs = {"a": line + [1, 2, 3], "b": line, "c": line}
    idx = support_index(seqs)
    # crop = supported line (2 grams) + private tail (3 grams): 2/5
    assert abs(crop_support(line + [1, 2, 3], "a", idx) - 2 / 5) < 1e-9
