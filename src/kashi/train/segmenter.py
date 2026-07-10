"""Train / evaluate Model 1 (transformer break scorer).

Model selection uses tolerance-window boundary F1 (±tolerance frames), not
exact-frame F1 — with ±50 ms label jitter, exact-frame F1 mostly measures the
jitter. loss = "latent_offset" (spec eq. (3)) or "soft_bce" (the report's
padded labels, edge-guarded).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data import manifest
from ..data.datasets import BreakDataset
from ..data.store import FeatureStore
from ..nn.segmenter import (
    TransformerSegmenter,
    boundary_f1,
    latent_offset_loss,
    pick_boundaries,
    soft_bce_loss,
    soft_break_labels,
)
from . import common


@torch.inference_mode()
def evaluate(cfg, checkpoint: str | Path, split: str = "test",
             version: str | None = None) -> dict:
    dev = common.device()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TransformerSegmenter(**payload["hparams"]).to(dev).eval()
    model.load_state_dict(payload["state_dict"])

    train_ids, test_ids = manifest.split_ids(cfg, version)
    ids = test_ids if split == "test" else train_ids
    ds = BreakDataset(cfg, ids, FeatureStore(cfg), version=version)
    if len(ds) == 0:
        raise SystemExit("no evaluation songs; populate the feature cache first")

    tol = int(cfg["train.segmenter.tolerance_frames"])
    thr = float(cfg["segmenter.transformer.threshold"])
    nms = int(cfg["segmenter.transformer.nms_frames"])
    scores = []
    for feats, breaks, valid in ds:
        logits = model.frame_logits(feats.numpy())
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred = pick_boundaries(probs, threshold=thr, nms=nms)
        true = np.flatnonzero(breaks.numpy()).tolist()
        scores.append(boundary_f1(pred, true, tol_frames=tol, frame_ms=cfg.frame_ms))
    return {
        "split": split,
        "songs": len(scores),
        "precision": float(np.mean([s.precision for s in scores])),
        "recall": float(np.mean([s.recall for s in scores])),
        "boundary_f1": float(np.mean([s.f1 for s in scores])),
        "mean_abs_ms": float(np.nanmean([s.mean_abs_ms for s in scores])),
        "tolerance_frames": tol,
        "checkpoint": str(checkpoint),
    }


def train(cfg, version: str | None = None, init: str | None = None,
          name: str | None = None, input_dim: int | None = None) -> Path:
    common.set_seed(int(cfg["train.seed"]))
    dev = common.device()
    run = common.run_dir(cfg, "segmenter", name)

    store = FeatureStore(cfg)
    train_ids, test_ids = manifest.split_ids(cfg, version)
    ds = BreakDataset(cfg, train_ids, store, version=version)
    if len(ds) == 0:
        raise SystemExit("no training songs; run `kashi encode` first")
    if input_dim is None:
        input_dim = ds[0][0].shape[-1]

    hparams = dict(
        input_dim=int(input_dim),
        attn_window=int(cfg["segmenter.transformer.attn_window"]),
    )
    model = TransformerSegmenter(**hparams).to(dev)
    if init:
        payload = torch.load(init, map_location="cpu", weights_only=True)
        model.load_state_dict(payload.get("state_dict", payload))

    loss_kind = cfg["train.segmenter.loss"]
    delta = int(cfg["train.segmenter.delta"])
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train.segmenter.lr"]))
    epochs = int(cfg["train.segmenter.epochs"])

    best_f1, best_path = -1.0, run / "best.pt"
    chunk = 2048  # frames; attention window << chunk, so chunking is lossless-ish
    order = np.arange(len(ds))
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(order)
        total = n_steps = 0
        for i in order:
            feats, breaks, valid = ds[int(i)]
            offset = int(np.random.randint(0, min(chunk, max(1, len(feats)))))
            for s in range(offset % chunk - chunk, len(feats), chunk):
                lo, hi = max(0, s), min(len(feats), s + chunk)
                if hi - lo < 32:
                    continue
                f_c = feats[lo:hi]
                b_c = breaks[lo:hi]
                v_c = valid[lo:hi]
                logits = model(f_c[None].float().to(dev))[0]
                valid_t = v_c.to(dev)
                if loss_kind == "latent_offset":
                    loss = latent_offset_loss(
                        logits, np.flatnonzero(b_c.numpy()), valid_t, delta=delta
                    )
                else:
                    targets = torch.from_numpy(soft_break_labels(b_c.numpy())).to(dev)
                    n_pos = max(1, int(b_c.sum()))
                    pos_weight = (len(b_c) - n_pos) / n_pos
                    loss = soft_bce_loss(logits, targets, valid_t, pos_weight=pos_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss)
                n_steps += 1
        ckpt = common.save_checkpoint(run / f"epoch{epoch}.pt", model, hparams)
        metrics = evaluate(cfg, ckpt, split="test", version=version)
        print(f"[segmenter] epoch {epoch}/{epochs} loss={total/max(1, n_steps):.4f} "
              f"F1@±{metrics['tolerance_frames']}fr={metrics['boundary_f1']:.4f}")
        if metrics["boundary_f1"] > best_f1:
            best_f1 = metrics["boundary_f1"]
            common.save_checkpoint(best_path, model, hparams)
    common.write_eval(run, {"best_boundary_f1": best_f1})
    print(f"[segmenter] best boundary F1 {best_f1:.4f} -> {best_path}")
    return best_path
