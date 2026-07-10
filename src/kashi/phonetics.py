"""Phonetic (articulatory) kernel over the 110-token inventory.

One kernel, three uses (docs/pipeline_specification.md §5.3):
  1. partial-credit classification loss  -> soft_targets()
  2. soft contrastive negatives          -> kernel_matrix()
  3. graded evaluation metric            -> partial_credit()

Each syllable decomposes as (consonant, glide, vowel); similarity is a convex
combination of consonant similarity (voicing/place/manner agreement) and vowel
similarity (height/backness distance), so ka~ga (one voicing bit) and wo~o
(homophone) score high while ka~n scores low.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .tokens import NUM_TOKENS, SILENCE, TOKENS, decompose

# Consonant articulatory features: consonant -> (voiced, place, manner)
_FEATURES: dict[str, tuple[int, str, str]] = {
    "k": (0, "velar", "plosive"),
    "g": (1, "velar", "plosive"),
    "s": (0, "alveolar", "fricative"),
    "z": (1, "alveolar", "fricative"),
    "sh": (0, "palatal", "fricative"),
    "j": (1, "palatal", "affricate"),
    "t": (0, "alveolar", "plosive"),
    "d": (1, "alveolar", "plosive"),
    "ch": (0, "palatal", "affricate"),
    "ts": (0, "alveolar", "affricate"),
    "n": (1, "alveolar", "nasal"),
    "N": (1, "alveolar", "nasal"),
    "h": (0, "glottal", "fricative"),
    "f": (0, "labial", "fricative"),
    "b": (1, "labial", "plosive"),
    "p": (0, "labial", "plosive"),
    "m": (1, "labial", "nasal"),
    "y": (1, "palatal", "approximant"),
    "r": (1, "alveolar", "approximant"),
    "w": (1, "labial", "approximant"),
}

# Vowel space (height, backness) in [0,1]^2.
_VOWEL_POS = {"a": (0.0, 0.5), "i": (1.0, 0.0), "u": (1.0, 1.0), "e": (0.5, 0.0), "o": (0.5, 1.0)}

# Weights (spec §5.3): consonant sub-weights and top-level mixture.
_LV, _LP, _LM = 0.2, 0.4, 0.4          # voice, place, manner
_MC, _MV, _MG = 0.45, 0.45, 0.10       # consonant, vowel, glide

# Modern-Japanese homophone pairs get near-identity similarity.
_HOMOPHONES = {frozenset(p) for p in [("を", "お"), ("ぢ", "じ"), ("づ", "ず")]}
_HOMOPHONE_SIM = 0.95


def _consonant_sim(c1: str, c2: str) -> float:
    if c1 == c2:
        return 1.0
    if c1 == "" or c2 == "":
        return 0.15  # vowel-only vs consonant onset
    v1, p1, m1 = _FEATURES[c1]
    v2, p2, m2 = _FEATURES[c2]
    return _LV * (v1 == v2) + _LP * (p1 == p2) + _LM * (m1 == m2)


def _vowel_sim(v1: str, v2: str) -> float:
    if v1 == v2:
        return 1.0
    if v1 == "" or v2 == "":
        return 0.0  # moraic nasal vs real vowel
    (h1, b1), (h2, b2) = _VOWEL_POS[v1], _VOWEL_POS[v2]
    return 1.0 - 0.5 * (abs(h1 - h2) + abs(b1 - b2))


def token_similarity(u: str, v: str) -> float:
    """Phonetic similarity in [0, 1]; 1 iff u == v."""
    if u == v:
        return 1.0
    if u == SILENCE or v == SILENCE:
        return 0.0
    if frozenset((u, v)) in _HOMOPHONES:
        return _HOMOPHONE_SIM
    c1, g1, v1 = decompose(u)
    c2, g2, v2 = decompose(v)
    sim = _MC * _consonant_sim(c1, c2) + _MV * _vowel_sim(v1, v2) + _MG * (g1 == g2)
    # Identical decomposition but distinct orthography (safety net).
    return min(sim, _HOMOPHONE_SIM)


@lru_cache(maxsize=1)
def kernel_matrix() -> np.ndarray:
    """Symmetric PSD [110, 110] kernel with unit diagonal, ordered like TOKENS."""
    K = np.empty((NUM_TOKENS, NUM_TOKENS))
    for i, u in enumerate(TOKENS):
        for j, v in enumerate(TOKENS):
            if j < i:
                K[i, j] = K[j, i]
            else:
                K[i, j] = token_similarity(u, v)
    # PSD projection: shift negative spectrum, renormalise to unit diagonal.
    lam_min = float(np.linalg.eigvalsh(K)[0])
    if lam_min < 0:
        K = K + (-lam_min + 1e-9) * np.eye(NUM_TOKENS)
        d = np.sqrt(np.diag(K))
        K = K / np.outer(d, d)
    return K


def soft_targets(alpha: float = 0.1, power: int = 4) -> np.ndarray:
    """Rows are smoothed one-hot targets (spec §5.3): (1-alpha) on the true
    class, alpha spread over phonetic neighbours (kernel**power, renormalised
    off-diagonal). power sharpens so only genuinely confusable classes get mass."""
    K = kernel_matrix() ** power
    np.fill_diagonal(K := K.copy(), 0.0)
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    Q = alpha * K / row_sums
    Q[np.arange(NUM_TOKENS), np.arange(NUM_TOKENS)] = 1.0 - alpha
    # Silence keeps a hard target (its off-diagonal row is all zeros).
    sil = TOKENS.index(SILENCE)
    Q[sil] = 0.0
    Q[sil, sil] = 1.0
    return Q


def partial_credit(pred: list[str], true: list[str]) -> float:
    """Mean kernel similarity between predicted and true tokens ('graded accuracy')."""
    if not true:
        return float("nan")
    return float(np.mean([token_similarity(p, t) for p, t in zip(pred, true)]))
