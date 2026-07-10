from kashi import tokens


def test_inventory_matches_legacy_order():
    import importlib.util
    from pathlib import Path

    legacy_path = Path(__file__).resolve().parents[1] / "models/predict_syllables/syllables.py"
    if not legacy_path.exists():
        import pytest

        pytest.skip("legacy syllables.py not present")
    src = legacy_path.read_text()
    # extract the literal list without importing (legacy module imports torch at top)
    body = src.split("syllables = [", 1)[1].split("]", 1)[0]
    legacy = [s.strip().strip("'\"") for s in body.split(",") if s.strip()]
    assert legacy == tokens.TOKENS
    assert len(tokens.TOKENS) == 110
    assert tokens.TOKENS[-1] == tokens.SILENCE


def test_kana_helpers():
    assert tokens.keep_kana("硬すぎたのかな、ビンが") == "すぎたのかなが"
    assert tokens.keep_kana("硬すぎたビン", include_katakana=True) == "すぎたビン"
    assert tokens.katakana_to_hiragana("ビン") == "びん"
    assert tokens.keep_alnum("か!?3a") == "か3a"


def test_decompose_and_romaji():
    assert tokens.decompose("か") == ("k", False, "a")
    assert tokens.decompose("が") == ("g", False, "a")
    assert tokens.decompose("しゃ") == ("sh", True, "a")
    assert tokens.decompose("ん") == ("N", False, "")
    assert tokens.romaji("きゃ") == "kya"
    assert tokens.romaji("しゅ") == "shu"
    assert tokens.romaji("を") == "wo"
    assert tokens.ROMAJI_TO_KANA["ka"] == "か"
    assert tokens.ROMAJI_TO_KANA["ji"] == "じ"  # first occurrence wins (ぢ aliases)


def test_every_token_decomposes():
    for t in tokens.TOKENS[:-1]:
        cons, glide, vowel = tokens.decompose(t)
        assert vowel in "aiueo" or (cons == "N" and vowel == "")
