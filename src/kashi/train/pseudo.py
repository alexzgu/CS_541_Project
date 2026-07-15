"""P5.1 pseudo-labeling / self-training (spec §9.4).

Two flavors, matching the two emission modes:

* CTC era (production, `harvest_ctc` / `kashi loop ctc-harvest`): spike-decode
  the unlabeled pool with the champion CTC model into pseudo-TRANSCRIPT crops
  (audio slice + token sequence + confidence stats). The crops feed straight
  into the ctc_bootstrap notebook's training set (PSEUDO_DIR flag) so the CTC
  model continue-trains on labeled + pseudo data. Confidence filtering happens
  at train time — the harvest keeps every sane crop and records stats.

* frame era (`loop` below, kept for emissions="frame" ablations): decode pool
  vocals with the segmental decoder, keep high-confidence segments as weak
  FRAME labels, retrain the frame classifier.

Textless throughout — the decoder never sees a transcript. Pool songs whose
YouTube id matches a frozen-test song are excluded (leak check).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .. import audio as audio_mod
from ..components.base import FrameAux
from ..data.store import FeatureStore, encoder_cache_id
from ..registry import create
from ..tokens import SILENCE_ID, TOKEN_INDEX
from . import common


def harvest(cfg, vocals_dir: str | Path | None = None, min_conf: float = 0.9,
            limit: int | None = None) -> dict:
    """Decode pool vocals -> weak frame labels under artifacts/pseudo/.
    Features are cached under the raw encoder id keyed by the vocals file."""
    vocals_dir = Path(vocals_dir or cfg.data_dir / "unlabeled" / "htdemucs")
    files, _ = _leak_filtered(cfg, sorted(vocals_dir.glob("*/vocals.mp3")))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no separated vocals under {vocals_dir}")
    store = FeatureStore(cfg, encoder_id=encoder_cache_id(cfg, include_projection=False))
    out_dir = cfg.artifacts_dir / "pseudo"
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = create(cfg, "encoder")
    decoder = create(cfg, "decoder")
    frame_ms = cfg.frame_ms
    n_frames = n_weak = 0
    for i, f in enumerate(files, 1):
        key = "pool_" + audio_mod.content_key(f)
        label_file = out_dir / f"{key}.npy"
        if label_file.exists():
            continue
        try:
            wave = audio_mod.load_audio(f, sr=cfg.sample_rate)
            if store.has(key):
                feats = store.load(key)
            else:
                feats = encoder.encode(wave, cfg.sample_rate)
                store.save(key, feats)
            aux = FrameAux(rms_db=audio_mod.log_rms_db(wave, cfg.sample_rate, frame_ms)[: len(feats)])
            segs = decoder.decode(feats, aux)
        except Exception as e:
            print(f"[pseudo] {f.parent.name}: FAILED ({e})")
            continue
        y = np.full(len(feats), -1, dtype=np.int64)
        for s in segs:
            if (s.confidence or 0) >= min_conf and s.token in TOKEN_INDEX:
                a = int(round(s.start * 1000)) // frame_ms
                b = int(round(s.end * 1000)) // frame_ms
                y[a: min(len(y), b)] = TOKEN_INDEX[s.token]
        np.save(label_file, y)
        n_frames += len(y)
        n_weak += int((y >= 0).sum())
        if i % 25 == 0:
            print(f"[pseudo] {i}/{len(files)} songs, weak-frame coverage "
                  f"{n_weak}/{n_frames} ({n_weak/max(1,n_frames):.1%})", flush=True)
    report = {"songs": len(files), "weak_frames": n_weak, "min_conf": min_conf}
    (out_dir / "harvest.json").write_text(json.dumps(report))
    print(f"[pseudo] harvest done: {n_weak:,} weak frames @conf>={min_conf}")
    return report


def load_weak(cfg) -> tuple[np.ndarray, np.ndarray]:
    """(X, Y) of all harvested weak frames (features from the raw cache)."""
    store = FeatureStore(cfg, encoder_id=encoder_cache_id(cfg, include_projection=False))
    out_dir = cfg.artifacts_dir / "pseudo"
    X, Y = [], []
    for label_file in sorted(out_dir.glob("pool_*.npy")):
        key = label_file.stem
        if not store.has(key):
            continue
        y = np.load(label_file)
        keep = y >= 0
        if not keep.any():
            continue
        X.append(store.load(key)[: len(y)][keep])
        Y.append(y[keep])
    if not X:
        return np.zeros((0, 768), np.float32), np.zeros(0, np.int64)
    return np.concatenate(X), np.concatenate(Y)


def loop(cfg, rounds: int = 1, min_conf: float = 0.9, weak_weight: float = 0.3) -> None:
    from ..eval.baselines import evaluate_pipeline
    from . import frame as frame_mod

    incumbent = None
    for r in range(1, rounds + 1):
        print(f"[loop] round {r}/{rounds}: harvest")
        harvest(cfg, min_conf=min_conf)
        Xw, Yw = load_weak(cfg)
        print(f"[loop] training frame model with {len(Yw):,} weak frames (w={weak_weight})")
        ckpt = frame_mod.train(cfg, name=f"pseudo-r{r}", weak=(Xw, Yw, weak_weight))
        # evaluate the NEW checkpoint (config may pin an incumbent)
        cfg.as_dict()["decoder"]["segmental"]["frame_checkpoint"] = str(ckpt)
        rep = evaluate_pipeline(cfg, split="test")
        p = rep["pooled"]
        print(f"[loop] round {r}: SER {p['ser']:.3f} timedF1 {p['timed_token_f1']:.3f}")
        common.append_leaderboard(cfg, {
            "run": f"p51_pseudo_r{r}", "ser": round(p["ser"], 4),
            "timed_token_f1": round(p["timed_token_f1"], 4),
            "boundary_f1_50ms": round(p["boundary@50ms_f1"], 4), "accuracy": "",
            "note": f"pseudo-label round {r}: {len(Yw):,} weak frames, w={weak_weight}, ckpt {ckpt}",
        })
        out = cfg.runs_dir / "ablations" / f"p51_pseudo_r{r}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=float))
        if incumbent is not None and p["ser"] >= incumbent:
            print("[loop] no SER improvement — stopping")
            break
        incumbent = p["ser"]


# ----------------------------------------------------------------------
# CTC-era pseudo-labeling (P5.1 redesign): pool songs -> transcript crops
# ----------------------------------------------------------------------

def spikes_to_crops(path: np.ndarray, probs: np.ndarray, frame_s: float,
                    blank_id: int, crop_s: float = 15.0, gap_blank_s: float = 0.4,
                    min_tokens: int = 4, min_rate: float = 1.0,
                    max_rate: float = 12.0) -> list[dict]:
    """Group CTC spikes into <=crop_s pseudo-transcript crops, cut at blank
    runs >= gap_blank_s (the notebook cuts labeled crops at silence rows the
    same way). Pure function over the greedy path — unit-testable.

    Returns [{t0, t1, tokens, conf_mean, conf_min, dur_s}]; rate/size sanity
    filters drop hallucination-dense or near-empty crops."""
    n = len(path)
    gap_f = max(1, int(round(gap_blank_s / frame_s)))
    spikes: list[tuple[int, int, float]] = []          # (frame, class, prob)
    prev = -1
    for t, c in enumerate(path):
        if c != prev and c != blank_id:
            spikes.append((t, int(c), float(probs[t])))
        prev = int(c)

    # cut before spike k when a long blank run separates it from spike k-1
    cut_before = set()
    for k in range(1, len(spikes)):
        a, b = spikes[k - 1][0], spikes[k][0]
        run = path[a:b] == blank_id
        if run.any():
            # longest blank run strictly between the two spikes
            best = cur = 0
            for v in run:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            if best >= gap_f:
                cut_before.add(k)

    crops: list[dict] = []

    def flush(group: list[tuple[int, int, float]], end_f: int) -> None:
        if len(group) < min_tokens:
            return
        t0 = max(0.0, group[0][0] * frame_s - 0.1)
        t1 = min(n * frame_s, max(end_f * frame_s, group[-1][0] * frame_s + 0.2))
        dur = t1 - t0
        rate = len(group) / max(dur, 1e-6)
        if dur < 0.5 or not (min_rate <= rate <= max_rate):
            return
        conf = np.array([p for _, _, p in group])
        crops.append({
            "t0": round(t0, 3), "t1": round(t1, 3), "dur_s": round(dur, 3),
            "tokens": [c for _, c, _ in group],
            "conf_mean": round(float(conf.mean()), 4),
            "conf_min": round(float(conf.min()), 4),
        })

    group: list[tuple[int, int, float]] = []
    for k, sp in enumerate(spikes):
        if group and (k in cut_before or (sp[0] - group[0][0]) * frame_s > crop_s):
            # end at the start of the separating blank run (+0.1 s pad) when
            # cut at a gap; just before the next spike when cut by overflow
            if k in cut_before:
                a = group[-1][0]
                while a < sp[0] and path[a] != blank_id:
                    a += 1
                flush(group, a + int(0.1 / frame_s))
            else:
                flush(group, sp[0] - 1)
            group = []
        group.append(sp)
    if group:
        flush(group, min(n, group[-1][0] + int(0.5 / frame_s)))
    return crops


def _test_ytids(cfg) -> set[str]:
    """YouTube ids of the frozen-test songs (leak check for the pool)."""
    from ..data.manifest import PAPER_TEST_IDS

    idx = Path(cfg["paths.dataset_repo"]) / "data" / "indexed" / "index.tsv"
    if not idx.is_file():
        raise SystemExit(
            f"leak check needs {idx} (song id -> YouTube id); refusing to "
            "harvest without it — a pool song could duplicate a test song")
    want = {str(i) for i in PAPER_TEST_IDS}
    with open(idx, newline="") as f:
        return {r["ID"] for r in csv.DictReader(f, delimiter="\t") if r["Index"] in want}


def _leak_filtered(cfg, files: list[Path]) -> tuple[list[Path], list[str]]:
    """Drop pool songs whose [YouTube id] matches a frozen-test song."""
    leak_ids = _test_ytids(cfg)
    kept, skipped = [], []
    for f in files:
        m = re.search(r"\[([A-Za-z0-9_-]{11})\]", f.parent.name)
        if m and m.group(1) in leak_ids:
            skipped.append(f.parent.name)
            print(f"[pseudo] LEAK EXCLUDED (frozen-test song): {f.parent.name}")
        else:
            kept.append(f)
    return kept, skipped


def harvest_ctc(cfg, vocals_dir: str | Path | None = None,
                out_dir: str | Path | None = None, limit: int | None = None) -> dict:
    """Spike-decode pool vocals with the champion CTC model into pseudo-
    transcript crops: <out>/crops/<key>_<k>.npy (fp16 16 kHz audio) plus one
    manifest.jsonl row per crop. Resume-safe via <out>/done.txt."""
    vocals_dir = Path(vocals_dir or cfg.data_dir / "unlabeled" / "htdemucs")
    out_dir = Path(out_dir or cfg.artifacts_dir / "pseudo_ctc")
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    files = sorted(vocals_dir.glob("*/vocals.mp3"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no separated vocals under {vocals_dir}")

    kept_files, skipped_leak = _leak_filtered(cfg, files)

    decoder = create(cfg, "decoder")
    if getattr(decoder, "emissions", "") != "ctc":
        raise SystemExit("harvest_ctc needs decoder.segmental.emissions = 'ctc'")
    frame_s = cfg.frame_ms / 1000.0
    sr = cfg.sample_rate

    done_file = out_dir / "done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    manifest = out_dir / "manifest.jsonl"
    n_crops = tot_dur = 0.0
    confs: list[float] = []
    for i, f in enumerate(kept_files, 1):
        key = "pool_" + audio_mod.content_key(f)
        if key in done:
            continue
        try:
            wave = audio_mod.load_audio(f, sr=sr)
            T = max(1, int(len(wave) / sr / frame_s))
            logp = decoder._ctc_log_probs(wave, sr, T)
            path = logp.argmax(-1)
            probs = np.exp(logp[np.arange(len(path)), path])
            crops = spikes_to_crops(path, probs, frame_s, blank_id=SILENCE_ID)
        except Exception as e:  # noqa: BLE001 — one bad mp3 must not kill the run
            print(f"[pseudo-ctc] {f.parent.name}: FAILED ({e})")
            with open(done_file, "a") as df:
                df.write(key + "\n")
            continue
        with open(manifest, "a") as mf:
            for k, c in enumerate(crops):
                rel = f"crops/{key}_{k}.npy"
                np.save(out_dir / rel,
                        wave[int(c["t0"] * sr): int(c["t1"] * sr)].astype(np.float16))
                mf.write(json.dumps({"file": rel, "song": f.parent.name, **c}) + "\n")
                n_crops += 1
                tot_dur += c["dur_s"]
                confs.append(c["conf_mean"])
        with open(done_file, "a") as df:
            df.write(key + "\n")
        if i % 10 == 0 or i == len(kept_files):
            print(f"[pseudo-ctc] {i}/{len(kept_files)} songs, {int(n_crops)} crops, "
                  f"{tot_dur/3600:.2f} h", flush=True)

    qs = np.percentile(confs, [10, 25, 50, 75, 90]).round(3).tolist() if confs else []
    report = {"songs": len(kept_files), "excluded_leak": skipped_leak,
              "crops": int(n_crops), "hours": round(tot_dur / 3600, 2),
              "conf_mean_percentiles_10_25_50_75_90": qs}
    (out_dir / "harvest_ctc.json").write_text(json.dumps(report, indent=1))
    print(f"[pseudo-ctc] done: {int(n_crops)} crops / {tot_dur/3600:.2f} h "
          f"under {out_dir}; conf_mean pctl(10..90)={qs}")
    return report
