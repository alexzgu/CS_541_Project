"""Evaluation metrics (spec §10). All operate on lists of kashi.subtitles.Segment.

Conventions: "lyric" rows = not silence, not <noise>, not excluded. Boundary
sets = internal row ends over non-excluded rows. Gold windows restrict both
prediction and reference to the verified interval before scoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..phonetics import token_similarity
from ..subtitles import Segment
from ..tokens import NOISE, SILENCE


def lyric_tokens(segs: list[Segment]) -> list[str]:
    return [s.token for s in segs if not s.exclude and s.token not in (SILENCE, NOISE, "")]


def boundary_times(segs: list[Segment]) -> list[float]:
    """Lyric-edge boundary set: start and end of every lyric row (silence,
    <noise> and excluded rows contribute nothing), deduplicated within 1 ms —
    robust to differing silence/noise row structure between label versions."""
    edges: list[float] = []
    for s in segs:
        if s.exclude or s.token in (SILENCE, NOISE, ""):
            continue
        for t in (s.start, s.end):
            if not edges or t - edges[-1] > 1e-3:
                edges.append(t)
    return edges


def clip_to_window(segs: list[Segment], start: float, end: float) -> list[Segment]:
    out = []
    for s in segs:
        if s.end <= start or s.start >= end:
            continue
        out.append(Segment(max(s.start, start), min(s.end, end), s.token,
                           exclude=s.exclude, confidence=s.confidence, meta=dict(s.meta)))
    return out


# ---------------------------------------------------------------------------

@dataclass
class BoundaryMetrics:
    precision: float
    recall: float
    f1: float
    mean_abs_ms: float
    n_pred: int
    n_ref: int


def boundary_metrics(pred: list[float], ref: list[float], tol_s: float) -> BoundaryMetrics:
    """Greedy monotone 1-1 matching of boundary times within ±tol_s."""
    pred, ref = sorted(pred), sorted(ref)
    i = j = hits = 0
    errs: list[float] = []
    while i < len(pred) and j < len(ref):
        d = pred[i] - ref[j]
        if abs(d) <= tol_s:
            hits += 1
            errs.append(abs(d))
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    p = hits / len(pred) if pred else 0.0
    r = hits / len(ref) if ref else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return BoundaryMetrics(p, r, f1, float(np.mean(errs) * 1000) if errs else float("nan"),
                           len(pred), len(ref))


def levenshtein(a: list[str], b: list[str], same=None) -> int:
    """Edit distance; `same(x, y)` overrides equality (e.g. homophone-aware)."""
    if same is None:
        same = lambda x, y: x == y  # noqa: E731
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (not same(x, y)))
        prev = cur
    return prev[-1]


def ser(pred: list[Segment], ref: list[Segment]) -> tuple[float, int, int]:
    """(SER, edit_distance, ref_len) on lyric token sequences."""
    p, r = lyric_tokens(pred), lyric_tokens(ref)
    d = levenshtein(p, r)
    return (d / len(r) if r else float("nan")), d, len(r)


@dataclass
class TimedTokenMetrics:
    precision: float
    recall: float
    f1: float
    partial_credit: float   # kernel similarity over time-matched pairs
    n_time_matched: int


def timed_token_metrics(pred: list[Segment], ref: list[Segment], tol_s: float = 0.05) -> TimedTokenMetrics:
    """Monotone matching by start time within ±tol_s; a match is correct iff
    tokens are equal. Partial credit = mean kernel similarity over all
    time-matched pairs (token-equal or not)."""
    P = [s for s in pred if not s.exclude and s.token not in (SILENCE, NOISE, "")]
    R = [s for s in ref if not s.exclude and s.token not in (SILENCE, NOISE, "")]
    i = j = correct = 0
    sims: list[float] = []
    while i < len(P) and j < len(R):
        d = P[i].start - R[j].start
        if abs(d) <= tol_s:
            try:
                sims.append(token_similarity(P[i].token, R[j].token))
            except KeyError:
                sims.append(0.0)
            correct += int(P[i].token == R[j].token)
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    p = correct / len(P) if P else 0.0
    r = correct / len(R) if R else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return TimedTokenMetrics(p, r, f1, float(np.mean(sims)) if sims else float("nan"), len(sims))


def noise_span_pr(pred: list[Segment], ref: list[Segment], min_iou: float = 0.3) -> dict:
    """Span-level precision/recall for <noise> with IoU >= min_iou matching."""
    P = [(s.start, s.end) for s in pred if s.token == NOISE]
    R = [(s.start, s.end) for s in ref if s.token == NOISE]

    def iou(a, b):
        inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union > 0 else 0.0

    used: set[int] = set()
    hits = 0
    for a in P:
        best, best_j = 0.0, -1
        for j, b in enumerate(R):
            if j in used:
                continue
            v = iou(a, b)
            if v > best:
                best, best_j = v, j
        if best >= min_iou:
            hits += 1
            used.add(best_j)
    return {
        "precision": hits / len(P) if P else float("nan"),
        "recall": hits / len(R) if R else float("nan"),
        "n_pred": len(P),
        "n_ref": len(R),
    }


# ---------------------------------------------------------------------------

def song_report(pred: list[Segment], ref: list[Segment],
                tolerances_ms: list[int] = (20, 50)) -> dict:
    out: dict = {}
    s, d, n = ser(pred, ref)
    out["ser"] = s
    out["edit_distance"] = d
    out["ref_tokens"] = n
    for tol in tolerances_ms:
        bm = boundary_metrics(boundary_times(pred), boundary_times(ref), tol / 1000)
        out[f"boundary@{tol}ms"] = asdict(bm)
    out["timed_token"] = asdict(timed_token_metrics(pred, ref))
    out["noise"] = noise_span_pr(pred, ref)
    return out


def pool_reports(reports: dict[int, dict]) -> dict:
    """Pooled headline numbers across songs."""
    if not reports:
        return {}
    total_d = sum(r["edit_distance"] for r in reports.values())
    total_n = sum(r["ref_tokens"] for r in reports.values())
    pooled = {
        "songs": len(reports),
        "ser": total_d / total_n if total_n else float("nan"),
        "ref_tokens": total_n,
    }
    for key in ("boundary@20ms", "boundary@50ms"):
        if key in next(iter(reports.values())):
            pooled[key + "_f1"] = float(np.mean([r[key]["f1"] for r in reports.values()]))
            pooled[key + "_mean_abs_ms"] = float(
                np.nanmean([r[key]["mean_abs_ms"] for r in reports.values()])
            )
    pooled["timed_token_f1"] = float(np.mean([r["timed_token"]["f1"] for r in reports.values()]))
    pooled["partial_credit"] = float(
        np.nanmean([r["timed_token"]["partial_credit"] for r in reports.values()])
    )
    return pooled
