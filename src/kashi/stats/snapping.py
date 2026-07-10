"""Monotone boundary snapping (spec §8.2) — event-set matching, NOT forced
alignment: no token identities, no transcript, moves bounded by delta_max."""

from __future__ import annotations

import numpy as np


def snap_events(
    events_s: list[float],
    candidates: list[dict],          # {time_s, prob, std_ms}
    delta_max_s: float = 0.1,
    c_miss: float = 1.2,
    eta: float = 0.5,
) -> list[dict]:
    """Order-preserving partial matching of labeled event times to candidate
    boundary times. cost(j,m) = |Δt|/Δmax − η·prob; unmatched events cost c_miss.

    Returns per event: {time_s (snapped or original), moved_ms, matched,
    candidate_prob, std_ms}.
    """
    J, M = len(events_s), len(candidates)
    ct = np.array([c["time_s"] for c in candidates])
    cp = np.array([c.get("prob", 1.0) for c in candidates])
    INF = 1e18

    def cost(j: int, m: int) -> float:
        d = abs(events_s[j] - ct[m])
        if d > delta_max_s:
            return INF
        return d / delta_max_s - eta * cp[m]

    # D[j][m]: min cost of matching first j events using first m candidates
    D = np.full((J + 1, M + 1), INF)
    D[0, :] = 0.0
    for j in range(1, J + 1):
        D[j, 0] = D[j - 1, 0] + c_miss
        for m in range(1, M + 1):
            c = cost(j - 1, m - 1)
            D[j, m] = min(
                D[j - 1, m - 1] + c if c < c_miss else INF,  # match (only if better than miss)
                D[j - 1, m] + c_miss,                        # event unmatched
                D[j, m - 1],                                 # candidate skipped
            )
    # backtrack
    out: list[dict | None] = [None] * J
    j, m = J, M
    while j > 0:
        c = cost(j - 1, m - 1) if m > 0 else INF
        if m > 0 and c < c_miss and D[j, m] == D[j - 1, m - 1] + c:
            out[j - 1] = {
                "time_s": float(ct[m - 1]),
                "moved_ms": float((ct[m - 1] - events_s[j - 1]) * 1000),
                "matched": True,
                "candidate_prob": float(cp[m - 1]),
                "std_ms": float(candidates[m - 1].get("std_ms") or 0.0),
            }
            j -= 1
            m -= 1
        elif D[j, m] == D[j - 1, m] + c_miss:
            out[j - 1] = {"time_s": float(events_s[j - 1]), "moved_ms": 0.0,
                          "matched": False, "candidate_prob": 0.0, "std_ms": 0.0}
            j -= 1
        else:
            m -= 1
    return out  # type: ignore[return-value]
