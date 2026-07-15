"""Tier-1 normalization: sokuon merges and chunk splits (dataset v2, S13)."""

from kashi.data.normalize import decompose, normalize_rows


def _row(s, e, tok, excl="True"):
    return {"start": str(s), "end": str(e), "token": tok, "exclude": excl}


def test_decompose():
    assert decompose("かっ") == ["か", "っ"]
    assert decompose("って") == ["っ", "て"]
    assert decompose("ない") == ["な", "い"]
    assert decompose("ちゃん") == ["ちゃ", "ん"]   # youon stays one token
    assert decompose("the") is None
    assert decompose("<gap>") is None


def test_trailing_sokuon_becomes_host_row():
    rows = [_row(0.0, 0.2, "な", "False"), _row(0.2, 0.5, "かっ"), _row(0.5, 0.7, "た", "False")]
    out, st = normalize_rows(rows)
    assert [r["token"] for r in out] == ["な", "か", "た"]
    ka = out[1]
    assert ka["start"] == "0.2" and ka["end"] == "0.5"   # closure time stays inside か
    assert ka["exclude"] == "False"
    assert st["recovered_rows"] == 1


def test_bare_sokuon_extends_previous():
    rows = [_row(33.0, 33.144, "ず", "False"), _row(33.167, 33.367, "っ"), _row(33.367, 33.855, "と", "False")]
    out, st = normalize_rows(rows)
    assert [r["token"] for r in out] == ["ず", "と"]
    assert float(out[0]["end"]) == 33.367                # ず absorbed the closure
    assert st["merged_sokuon"] == 1 and st["recovered_rows"] == 1


def test_bare_sokuon_without_host_is_kept():
    rows = [_row(0.0, 1.0, "<silence>", "False"), _row(2.0, 2.2, "っ")]
    out, _ = normalize_rows(rows)
    assert [r["token"] for r in out] == ["<silence>", "っ"]
    assert out[1]["exclude"] == "True"                   # untouched, still excluded


def test_leading_sokuon_chunk_splits_share():
    rows = [_row(1.0, 1.3, "だ", "False"), _row(1.3, 1.9, "って")]
    out, st = normalize_rows(rows)
    assert [r["token"] for r in out] == ["だ", "て"]
    assert float(out[0]["end"]) == 1.6                   # だ gained っ's half of 0.6s
    assert out[1]["start"] == "1.6" and out[1]["end"] == "1.9"
    assert st["split_chunks"] == 0 and st["merged_sokuon"] == 1


def test_chunk_split_even():
    out, st = normalize_rows([_row(2.0, 2.4, "ない")])
    assert [(r["token"], r["start"], r["end"]) for r in out] == [
        ("な", "2.0", "2.2"), ("い", "2.2", "2.4")]
    assert all(r["exclude"] == "False" for r in out)
    assert st["split_chunks"] == 1 and st["recovered_rows"] == 2


def test_unmapped_passthrough():
    rows = [_row(0.0, 0.5, "the"), _row(0.5, 0.9, "<gap>")]
    out, st = normalize_rows(rows)
    assert [r["token"] for r in out] == ["the", "<gap>"]
    assert all(r["exclude"] == "True" for r in out)
    assert st["unmapped"] == 2
