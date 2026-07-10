"""P5.1 pseudo-labeling / self-training (spec §9.4).

One round: decode separated pool vocals with the segmental decoder, keep
segments with confidence >= theta as weak frame labels, retrain the frame
classifier on labeled + weak data (weak weight w), evaluate on the frozen
test split. Adopt only if it beats the incumbent (caller judges). Textless
throughout — the decoder never sees a transcript.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import audio as audio_mod
from ..components.base import FrameAux
from ..data.store import FeatureStore, encoder_cache_id
from ..registry import create
from ..tokens import TOKEN_INDEX
from . import common


def harvest(cfg, vocals_dir: str | Path | None = None, min_conf: float = 0.9,
            limit: int | None = None) -> dict:
    """Decode pool vocals -> weak frame labels under artifacts/pseudo/.
    Features are cached under the raw encoder id keyed by the vocals file."""
    vocals_dir = Path(vocals_dir or cfg.data_dir / "unlabeled" / "htdemucs")
    files = sorted(vocals_dir.glob("*/vocals.mp3"))
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
