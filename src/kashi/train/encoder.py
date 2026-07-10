"""Contrastive training of the projection head g_w (spec §9.1-§9.2, P5.2).

mode="temporal" (unsupervised): anchors are random cached frames, positives
are frames 1-3 steps later — same phonetic content, different local noise.
Any audio that has been through `kashi encode` contributes, including the
unlabeled pool. Features are mmap'd, so RAM stays flat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.store import FeatureStore
from ..nn.contrastive import ProjectionHead, info_nce
from . import common


def _open_features(store: FeatureStore, max_len: int = 200_000) -> list[np.ndarray]:
    arrays = []
    for key in store.keys():
        path = store.dir / f"{key}.npy"
        if not path.exists():
            continue  # legacy .pt entries are skipped (mmap needs npy)
        arr = np.load(path, mmap_mode="r")
        if len(arr) > 10:
            arrays.append(arr)
    return arrays


def train(cfg, name: str | None = None, steps: int | None = None) -> Path:
    common.set_seed(int(cfg["train.seed"]))
    dev = common.device()
    run = common.run_dir(cfg, "encoder", name)
    store = FeatureStore(cfg)
    arrays = _open_features(store)
    if not arrays:
        raise SystemExit("feature cache is empty — run `kashi encode` first")
    n_frames = sum(len(a) for a in arrays)
    print(f"[encoder] temporal InfoNCE over {len(arrays)} songs / {n_frames:,} frames")

    steps = steps or int(cfg["train.encoder.steps"])
    batch = int(cfg["train.encoder.batch_size"])
    tau = float(cfg["train.encoder.temperature"])
    dim = arrays[0].shape[1]
    head = ProjectionHead(in_dim=dim).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    rng = np.random.default_rng(int(cfg["train.seed"]))

    weights = np.array([len(a) for a in arrays], dtype=float)
    weights /= weights.sum()
    for step in range(1, steps + 1):
        songs = rng.choice(len(arrays), size=batch, p=weights)
        anchors = np.empty((batch, dim), dtype=np.float32)
        positives = np.empty((batch, dim), dtype=np.float32)
        for i, s in enumerate(songs):
            a = arrays[s]
            delta = int(rng.integers(1, 4))
            t = int(rng.integers(0, len(a) - delta))
            anchors[i] = a[t]
            positives[i] = a[t + delta]
        za = head(torch.from_numpy(anchors).to(dev))
        zp = head(torch.from_numpy(positives).to(dev))
        loss = info_nce(za, zp, temperature=tau)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"[encoder] step {step}/{steps} loss={float(loss):.4f}")

    out = cfg.artifacts_dir / "encoder" / "temporal_head.pt"
    head.cpu().save(out)
    common.write_eval(run, {"steps": steps, "songs": len(arrays), "frames": n_frames})
    print(f"[encoder] head -> {out}")
    return out
