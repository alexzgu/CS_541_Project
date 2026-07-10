"""Dataset cleaning without forced alignment (spec §8).

Per song: unsupervised boundary candidates (HMM ∪ onsets ∪ voicing deltas) ->
monotone snapping of labeled row edges (≤ delta_max, order-preserving, token
sequence never consulted for timing) -> <noise>/missed-vocal detection inside
non-lyric regions -> classifier-agreement flags -> QA quarantine gates.
Outputs a new label version (append-only) + realign_report.csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import audio as audio_mod
from .components.boundaries import (
    hmm_boundaries,
    merge_candidates,
    onset_boundaries,
    voicing_boundaries,
)
from .data import manifest
from .data.store import FeatureStore, encoder_cache_id
from .stats.snapping import snap_events
from .stats.tda import voicing_track
from .subtitles import Segment, read_csv, write_csv
from .tokens import NOISE, SILENCE


def _song_candidates(cfg, song_id: int, feats, wave, sr, voicing) -> list[dict]:
    sources = []
    wanted = cfg["boundaries.sources"]
    if "hmm" in wanted:
        key = f"{encoder_cache_id(cfg)}_{cfg.frame_ms}ms_{song_id}"
        sources.append(hmm_boundaries(cfg, feats, cache_key=key))
    if "onset" in wanted:
        sources.append(onset_boundaries(cfg, wave, sr))
    if "voicing" in wanted:
        sources.append(voicing_boundaries(cfg, voicing))
    return merge_candidates(sources, merge_within_s=cfg.frame_ms / 1000.0)


def _snap_rows(cfg, rows: list[Segment], candidates: list[dict]) -> tuple[list[Segment], dict]:
    """Snap the unique edge times of non-excluded rows; rows sharing an edge
    move together. Degenerate rows (start >= end after snap) revert + flag."""
    delta_max = float(cfg["realign.delta_max_ms"]) / 1000.0
    events: list[float] = []
    for r in rows:
        if r.exclude:
            continue
        for t in (round(r.start, 3), round(r.end, 3)):
            if not events or t > events[-1]:
                events.append(t)
            elif t < events[-1]:  # out-of-order edge (overlap remnants): keep unique sorted
                if t not in events:
                    events.append(t)
                    events.sort()
    snapped = snap_events(events, candidates, delta_max_s=delta_max,
                          c_miss=float(cfg["realign.c_miss"]), eta=float(cfg["realign.eta"]))
    time_map = {e: s for e, s in zip(events, snapped)}

    out: list[Segment] = []
    moved, flagged, matched = [], 0, 0
    for r in rows:
        if r.exclude:
            out.append(r)
            continue
        s_info = time_map.get(round(r.start, 3))
        e_info = time_map.get(round(r.end, 3))
        new_s = s_info["time_s"] if s_info else r.start
        new_e = e_info["time_s"] if e_info else r.end
        flag = ""
        if new_s >= new_e:  # degenerate after snapping: revert
            new_s, new_e, flag = r.start, r.end, "degenerate"
        m_ms = (new_s - r.start) * 1000
        row_matched = bool(s_info and s_info["matched"]) or bool(e_info and e_info["matched"])
        if not row_matched and not r.is_silence:
            flag = flag or "unmatched"
        conf = max(s_info["candidate_prob"] if s_info else 0.0,
                   e_info["candidate_prob"] if e_info else 0.0)
        std = max(s_info["std_ms"] if s_info else 0.0, e_info["std_ms"] if e_info else 0.0)
        meta = {"moved_ms": round(m_ms, 1), "boundary_std_ms": round(std, 1)}
        if flag:
            meta["flag"] = flag
            flagged += 1
        if not r.is_silence:
            moved.append(abs(m_ms))
            matched += row_matched
        out.append(Segment(new_s, new_e, r.token, exclude=r.exclude,
                           confidence=round(conf, 3) if conf else None, meta=meta))
    stats = {
        "rows": len(rows),
        "lyric_rows": len(moved),
        "mean_abs_shift_ms": float(np.mean(moved)) if moved else 0.0,
        "p90_abs_shift_ms": float(np.percentile(moved, 90)) if moved else 0.0,
        "matched_frac": matched / max(1, len(moved)),
        "flagged_rows": flagged,
    }
    return out, stats


def flatness_track(wave: np.ndarray, sr: int, frame_ms: int) -> np.ndarray:
    """Per-frame spectral flatness in [0,1]: breath/hiss is broadband (high),
    voiced singing is harmonic (low). 46 ms windows on the frame grid."""
    hop = int(sr * frame_ms / 1000)
    win = int(sr * 0.046)
    T = len(wave) // hop
    out = np.zeros(T, dtype=np.float32)
    hann = np.hanning(win)
    for t in range(T):
        c = t * hop + hop // 2
        seg = wave[max(0, c - win // 2): c + win // 2]
        if len(seg) < win // 2:
            continue
        P = np.abs(np.fft.rfft(seg * hann[: len(seg)])) ** 2 + 1e-12
        P = P[2:]  # drop DC region
        out[t] = float(np.exp(np.mean(np.log(P))) / np.mean(P))
    return out


def _noise_pass(cfg, rows: list[Segment], rms_db: np.ndarray, voicing: np.ndarray,
                flatness: np.ndarray | None = None) -> tuple[list[Segment], dict]:
    """Inside non-lyric regions: energetic∧aperiodic runs -> <noise> rows;
    energetic∧periodic runs -> missed-vocal report entries."""
    frame_s = cfg.frame_ms / 1000.0
    T = len(rms_db)
    covered = np.zeros(T, dtype=bool)
    for r in rows:
        if not r.is_silence:  # excluded rows are known content, not "missed"
            covered[int(r.start / frame_s): max(0, int(np.ceil(r.end / frame_s)))] = True
    v_thr = float(cfg["realign.voicing_thresh"])
    min_fr = max(1, int(cfg["realign.noise_min_ms"]) // cfg.frame_ms)
    voicing = voicing[:T]

    if flatness is not None and cfg.get("realign.noise_method", "rms_voicing") == "flatness":
        # breath = audible + broadband + aperiodic; gate well below the lyric level
        quiet_gate = float(rms_db.max()) - float(cfg.get("realign.flatness_rms_window_db", 55.0))
        f_thr = float(cfg.get("realign.flatness_thresh", 0.3))
        energetic = (rms_db >= quiet_gate) & (flatness[:T] >= f_thr) & ~covered
    else:
        thr_db = max(float(cfg["realign.noise_rms_db"]), float(rms_db.max()) - 40.0)
        energetic = (rms_db >= thr_db) & ~covered
    noise_rows: list[Segment] = []
    missed: list[tuple[float, float]] = []
    t = 0
    while t < T:
        if not energetic[t]:
            t += 1
            continue
        u = t
        while u < T and energetic[u]:
            u += 1
        if u - t >= min_fr:
            span_voiced = float(np.mean(voicing[t:u] >= v_thr))
            if span_voiced < 0.5:
                noise_rows.append(Segment(t * frame_s, u * frame_s, NOISE,
                                          meta={"moved_ms": 0.0, "boundary_std_ms": 0.0}))
            else:
                missed.append((t * frame_s, u * frame_s))
        t = u
    merged = sorted(rows + noise_rows, key=lambda r: (r.start, r.end))
    return merged, {"noise_spans": len(noise_rows), "missed_vocal_spans": len(missed),
                    "missed_vocal": [(round(a, 2), round(b, 2)) for a, b in missed[:20]]}


def realign_dataset(cfg, song_ids: list[int] | None = None,
                    out_version: str | None = None) -> Path:
    out_version = out_version or cfg["realign.out_version"]
    out_dir = manifest.subtitles_dir(cfg, out_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(cfg)
    ids = song_ids or manifest.labeled_ids(cfg, cfg["data.version"])
    report_rows = []
    for song_id in ids:
        paths = manifest.song_paths(cfg, song_id)
        src = manifest.subtitles_dir(cfg, cfg["data.version"]) / f"{song_id}.csv"
        if not src.is_file() or not store.has(str(song_id)) or not paths.vocals.is_file():
            print(f"[realign] song {song_id}: missing inputs, skipped")
            continue
        rows = read_csv(src)
        feats = store.load(str(song_id))
        wave = audio_mod.load_audio(paths.vocals, sr=cfg.sample_rate)
        rms_db = audio_mod.log_rms_db(wave, cfg.sample_rate, cfg.frame_ms)[: len(feats)]
        voicing = voicing_track(
            wave, cfg.sample_rate, cfg.frame_ms,
            method=cfg["features.voicing"],
            window_ms=float(cfg["features.window_ms"]),
            fmin=float(cfg["features.fmin_hz"]), fmax=float(cfg["features.fmax_hz"]),
        )
        candidates = _song_candidates(cfg, song_id, feats, wave, cfg.sample_rate, voicing)
        snapped, stats = _snap_rows(cfg, rows, candidates)
        flat = None
        if cfg.get("realign.noise_method", "rms_voicing") == "flatness":
            flat = flatness_track(wave, cfg.sample_rate, cfg.frame_ms)
        final, noise_stats = _noise_pass(cfg, snapped, rms_db, voicing, flatness=flat)
        write_csv(final, out_dir / f"{song_id}.csv")
        stats.update(noise_stats)
        stats["song_id"] = song_id
        stats["candidates"] = len(candidates)
        stats["quarantine"] = bool(
            cfg["qa.quarantine"] and (
                stats["mean_abs_shift_ms"] > float(cfg["qa.max_mean_shift_ms"])
                or stats["flagged_rows"] / max(1, stats["rows"]) > float(cfg["qa.max_flagged_frac"])
                or stats["matched_frac"] < float(cfg["qa.min_candidate_recall"])
            )
        )
        report_rows.append(stats)
        print(f"[realign] song {song_id}: mean|shift|={stats['mean_abs_shift_ms']:.0f}ms "
              f"p90={stats['p90_abs_shift_ms']:.0f}ms matched={stats['matched_frac']:.2f} "
              f"noise={stats['noise_spans']} missed-vocal={stats['missed_vocal_spans']}"
              f"{' QUARANTINE' if stats['quarantine'] else ''}")
    report = out_dir.parent / "realign_report.csv"
    if report_rows:
        keys = [k for k in report_rows[0] if k != "missed_vocal"]
        with open(report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(report_rows)
        (out_dir.parent / "realign_report.json").write_text(
            json.dumps(report_rows, indent=1, default=float))
    print(f"[realign] {len(report_rows)} songs -> {out_dir} (report: {report})")
    return report


# ---------------------------------------------------------------------------
# Gold-side measurement (P2 acceptance)
# ---------------------------------------------------------------------------

def report_vs_gold(cfg, version: str | None = None) -> dict:
    from .eval import gold as gold_mod
    from .eval import metrics as M

    version = version or cfg["realign.out_version"]
    out = {}
    for song_id in gold_mod.gold_ids(cfg):
        gold_rows = read_csv(gold_mod.gold_dir(cfg) / f"{song_id}.csv")
        entry = {}
        for name, ver in (("v1", cfg["data.version"]), ("v2", version)):
            f = manifest.subtitles_dir(cfg, ver) / f"{song_id}.csv"
            if not f.is_file():
                continue
            rows = read_csv(f)
            bm = M.boundary_metrics(M.boundary_times(rows), M.boundary_times(gold_rows), 0.05)
            entry[name] = {"boundary_f1_50ms": bm.f1, "mean_abs_ms": bm.mean_abs_ms,
                           "recall": bm.recall, "precision": bm.precision}
            if name == "v2":
                entry["noise"] = M.noise_span_pr(rows, gold_rows)
        # candidate recall vs gold boundaries
        key = f"{encoder_cache_id(cfg)}_{cfg.frame_ms}ms_{song_id}"
        cache = cfg.artifacts_dir / "boundaries" / f"{key}.json"
        if cache.is_file():
            cand = [c["time_s"] for c in json.loads(cache.read_text())]
            gb = M.boundary_times(gold_rows)
            hits = sum(1 for g in gb if any(abs(g - c) <= 0.1 for c in cand))
            entry["hmm_candidate_recall_100ms"] = hits / max(1, len(gb))
        out[song_id] = entry
    if out:
        import numpy as _np

        agg = {}
        for name in ("v1", "v2"):
            vals = [v[name] for v in out.values() if name in v]
            if vals:
                agg[name] = {k: float(_np.mean([x[k] for x in vals])) for k in vals[0]}
        out["pooled"] = agg
    return out
