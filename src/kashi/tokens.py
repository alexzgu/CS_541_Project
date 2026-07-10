"""The 110-token target inventory (109 hiragana morae + <silence>).

IMPORTANT: TOKENS preserves the exact order of the legacy
models/predict_syllables/syllables.py — trained classifier checkpoints index
their output classes by this order.

<noise> is NOT a classifier class; it exists only as a dataset annotation
(breathing and similar non-lyric sounds found by `kashi realign`).
"""

from __future__ import annotations

SILENCE = "<silence>"
NOISE = "<noise>"

TOKENS: list[str] = [
    "あ", "い", "う", "え", "お",
    "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の",
    "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も",
    "や", "ゆ", "よ",
    "ら", "り", "る", "れ", "ろ",
    "わ", "を", "ん",
    "が", "ぎ", "ぐ", "げ", "ご",
    "ざ", "じ", "ず", "ぜ", "ぞ",
    "だ", "ぢ", "づ", "で", "ど",
    "ば", "び", "ぶ", "べ", "ぼ",
    "ぱ", "ぴ", "ぷ", "ぺ", "ぽ",
    "きゃ", "きゅ", "きょ",
    "しゃ", "しゅ", "しょ",
    "ちゃ", "ちゅ", "ちょ",
    "にゃ", "にゅ", "にょ",
    "ひゃ", "ひゅ", "ひょ",
    "みゃ", "みゅ", "みょ",
    "りゃ", "りゅ", "りょ",
    "ぎゃ", "ぎゅ", "ぎょ",
    "じゃ", "じゅ", "じょ",
    "びゃ", "びゅ", "びょ",
    "ぴゃ", "ぴゅ", "ぴょ",
    "でぃ", "ふぁ", "ふぃ", "ふぇ", "ふぉ",
    SILENCE,
]

TOKEN_INDEX: dict[str, int] = {t: i for i, t in enumerate(TOKENS)}
NUM_TOKENS = len(TOKENS)  # 110
SILENCE_ID = TOKEN_INDEX[SILENCE]


def token_id(token: str) -> int:
    return TOKEN_INDEX[token]


# ---------------------------------------------------------------------------
# Kana character helpers (port of data_processing/utils/character_filtering.py)
# ---------------------------------------------------------------------------

def is_hiragana(char: str) -> bool:
    return "ぁ" <= char <= "ゔ"


def is_katakana(char: str) -> bool:
    return "ァ" <= char <= "ヴ"


def is_long_vowel_mark(char: str) -> bool:
    return char == "ー"  # ー


def keep_kana(token: str, include_katakana: bool = False) -> str:
    """Strip everything that is not hiragana (optionally katakana) or 'ー'."""
    if include_katakana:
        return "".join(
            c for c in token if is_hiragana(c) or is_katakana(c) or is_long_vowel_mark(c)
        )
    return "".join(c for c in token if is_hiragana(c) or is_long_vowel_mark(c))


def keep_alnum(token: str) -> str:
    return "".join(c for c in token if c.isalnum())


def katakana_to_hiragana(token: str) -> str:
    """Codepoint shift, matching the legacy chr(ord(c) - 96) behaviour."""
    return "".join(chr(ord(c) - 96) if is_katakana(c) else c for c in str(token))


# ---------------------------------------------------------------------------
# Romaji decomposition of the inventory (used by kashi.phonetics and the
# romaji->kana mapping for the roumanji dataset import, P2b).
# ---------------------------------------------------------------------------

_VOWELS = "aiueo"

_GOJUON: dict[str, tuple[str, str]] = {}  # kana -> (consonant, vowel)


def _row(cons: str, kana: str, vowels: str = _VOWELS) -> None:
    for k, v in zip(kana, vowels):
        _GOJUON[k] = (cons, v)


_row("", "あいうえお")
_row("k", "かきくけこ")
_row("s", "さすせそ", "aueo")
_GOJUON["し"] = ("sh", "i")
_row("t", "たてと", "aeo")
_GOJUON["ち"] = ("ch", "i")
_GOJUON["つ"] = ("ts", "u")
_row("n", "なにぬねの")
_row("h", "はひへほ", "aieo")
_GOJUON["ふ"] = ("f", "u")
_row("m", "まみむめも")
_row("y", "やゆよ", "auo")
_row("r", "らりるれろ")
_GOJUON["わ"] = ("w", "a")
_GOJUON["を"] = ("w", "o")
_GOJUON["ん"] = ("N", "")
_row("g", "がぎぐげご")
_row("z", "ざずぜぞ", "aueo")
_GOJUON["じ"] = ("j", "i")
_row("d", "だでど", "aeo")
_GOJUON["ぢ"] = ("j", "i")   # modern Japanese: ぢ = じ
_GOJUON["づ"] = ("z", "u")   # modern Japanese: づ = ず
_row("b", "ばびぶべぼ")
_row("p", "ぱぴぷぺぽ")

_YOON_VOWEL = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
# Consonant of the yōon combo, keyed by its i-column kana.
_YOON_CONS = {
    "き": "k", "し": "sh", "ち": "ch", "に": "n", "ひ": "h",
    "み": "m", "り": "r", "ぎ": "g", "じ": "j", "び": "b", "ぴ": "p",
}
_EXTRA = {
    "でぃ": ("d", False, "i"),
    "ふぁ": ("f", False, "a"),
    "ふぃ": ("f", False, "i"),
    "ふぇ": ("f", False, "e"),
    "ふぉ": ("f", False, "o"),
}


def decompose(token: str) -> tuple[str, bool, str]:
    """token -> (consonant, palatal_glide, vowel).

    '' consonant means vowel-only mora; 'N' is the moraic nasal (vowel '').
    Raises KeyError for tokens outside the inventory (incl. <silence>).
    """
    if token in _EXTRA:
        return _EXTRA[token]
    if len(token) == 2 and token[1] in _YOON_VOWEL:
        return (_YOON_CONS[token[0]], True, _YOON_VOWEL[token[1]])
    cons, vowel = _GOJUON[token]
    return (cons, False, vowel)


def romaji(token: str) -> str:
    """Hepburn-ish romaji of an inventory token (for logs/tests)."""
    if token == SILENCE:
        return "sil"
    c, glide, v = decompose(token)
    if c == "N":
        return "n"
    return c + ("y" if glide and c not in ("sh", "ch", "j") else "") + v


# Reverse map romaji -> kana token, for the roumanji dataset import (P2b).
ROMAJI_TO_KANA: dict[str, str] = {}
for _t in TOKENS[:-1]:
    ROMAJI_TO_KANA.setdefault(romaji(_t), _t)
