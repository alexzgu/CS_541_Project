import numpy as np

from kashi import phonetics
from kashi.tokens import NUM_TOKENS, SILENCE, TOKENS


def test_worked_values_from_spec():
    k = phonetics.token_similarity
    assert abs(k("か", "が") - 0.91) < 1e-9      # voicing bit only
    assert abs(k("か", "た") - 0.82) < 1e-9      # place differs
    assert abs(k("か", "き") - 0.66) < 0.02      # vowel moves
    assert k("か", "ん") <= 0.15                  # nothing in common
    assert k("を", "お") == 0.95                  # homophone
    assert k("ぢ", "じ") == 0.95                  # homophone (identical decomposition)
    assert k("か", "か") == 1.0
    assert k(SILENCE, "か") == 0.0
    assert k(SILENCE, SILENCE) == 1.0
    # ordering sanity: ka~ga > ka~ta > ka~ki > ka~n
    assert k("か", "が") > k("か", "た") > k("か", "き") > k("か", "ん")


def test_kernel_matrix_properties():
    K = phonetics.kernel_matrix()
    assert K.shape == (NUM_TOKENS, NUM_TOKENS)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K)[0] >= -1e-8  # PSD


def test_soft_targets_rows():
    Q = phonetics.soft_targets(alpha=0.1, power=4)
    assert np.allclose(Q.sum(axis=1), 1.0)
    ka = TOKENS.index("か")
    ga = TOKENS.index("が")
    n = TOKENS.index("ん")
    assert abs(Q[ka, ka] - 0.9) < 1e-9
    assert Q[ka, ga] > Q[ka, n]
    sil = TOKENS.index(SILENCE)
    assert Q[sil, sil] == 1.0


def test_partial_credit():
    assert phonetics.partial_credit(["か"], ["か"]) == 1.0
    assert phonetics.partial_credit(["か"], ["が"]) > 0.9
