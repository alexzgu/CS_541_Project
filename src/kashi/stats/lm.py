"""Token bigram over label sequences (spec §6). Add-k smoothed; a prior over
token strings learned offline — not a transcript; decoding stays textless.
Off by default (decoder.segmental.lambda_lm = 0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data import manifest
from ..subtitles import read_csv
from ..tokens import NOISE, TOKEN_INDEX, TOKENS


def fit_bigram(cfg, version: str | None = None, add_k: float = 0.1) -> Path:
    version = version or cfg["data.version"]
    V = len(TOKENS)
    counts = np.zeros((V, V))
    for song_id in manifest.labeled_ids(cfg, version):
        prev = None
        for seg in read_csv(manifest.subtitles_dir(cfg, version) / f"{song_id}.csv"):
            if seg.exclude or seg.token == NOISE or seg.token not in TOKEN_INDEX:
                prev = None
                continue
            cur = TOKEN_INDEX[seg.token]
            if prev is not None:
                counts[prev, cur] += 1
            prev = cur
    probs = (counts + add_k) / (counts.sum(1, keepdims=True) + add_k * V)
    path = cfg.artifacts_dir / "lm_bigram.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, log_bigram=np.log(probs))
    print(f"[fit lm] {int(counts.sum())} transitions -> {path}")
    return path


def log_bigram(cfg) -> np.ndarray | None:
    path = cfg.artifacts_dir / "lm_bigram.npz"
    return np.load(path)["log_bigram"] if path.is_file() else None


# ---------------------------------------------------------------------------
# Text-trained bigram (task #14): same artifact format, fit on kana-ized
# Japanese TEXT instead of the 93 songs' labels — still transcript-free at
# decode time (a prior over token strings, not this song's lyrics).
# ---------------------------------------------------------------------------

_KATA_TO_HIRA = {chr(k): chr(k - 0x60) for k in range(0x30A1, 0x30F7)}


def kana_token_stream(text: str) -> list[int]:
    """Hiragana/katakana text -> our 110-token ids. Youon pairs (きゃ) are one
    token; っ contributes no token (v2 policy: it belongs to its host's time);
    ー repeats the previous token's vowel; anything else breaks the sequence
    (encoded as -1 so callers don't count transitions across it)."""
    from ..tokens import decompose

    out: list[int] = []
    text = "".join(_KATA_TO_HIRA.get(ch, ch) for ch in text)
    i = 0
    while i < len(text):
        two = text[i:i + 2]
        if len(two) == 2 and two in TOKEN_INDEX:
            out.append(TOKEN_INDEX[two])
            i += 2
            continue
        ch = text[i]
        i += 1
        if ch == "っ":
            continue
        if ch == "ー" and out and out[-1] != -1:
            _, _, v = decompose(TOKENS[out[-1]])
            hira_v = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}.get(v)
            if hira_v:
                out.append(TOKEN_INDEX[hira_v])
            continue
        if ch in TOKEN_INDEX:
            out.append(TOKEN_INDEX[ch])
        elif not out or out[-1] != -1:
            out.append(-1)
    return out


def fit_text_bigram(cfg, text_file: str | Path, add_k: float = 0.1,
                    out_name: str = "lm_bigram_text.npz") -> Path:
    """Fit the bigram on a text corpus (one sentence per line or TSV with the
    sentence last). Mixed script is read with pykakasi; kana-only lines skip it."""
    import pykakasi

    kks = pykakasi.kakasi()
    V = len(TOKENS)
    counts = np.zeros((V, V))
    n_sent = 0
    with open(text_file, encoding="utf-8") as f:
        for line in f:
            sent = line.rstrip("\n").split("\t")[-1]
            if not sent:
                continue
            hira = "".join(item["hira"] for item in kks.convert(sent))
            prev = -1
            for tok in kana_token_stream(hira):
                if prev >= 0 and tok >= 0:
                    counts[prev, tok] += 1
                prev = tok
            n_sent += 1
            if n_sent % 50000 == 0:
                print(f"[fit text-lm] {n_sent} sentences, {int(counts.sum())} transitions", flush=True)
    probs = (counts + add_k) / (counts.sum(1, keepdims=True) + add_k * V)
    path = cfg.artifacts_dir / out_name
    np.savez_compressed(path, log_bigram=np.log(probs))
    print(f"[fit text-lm] {n_sent} sentences, {int(counts.sum())} transitions -> {path}")
    return path
