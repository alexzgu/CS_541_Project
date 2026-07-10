"""Train Model 2f (per-frame syllable posterior) — spec §5.2.

Frame targets come from the subtitle rows (token per covered frame; silence
for uncovered gaps — the trim step removed only sub-threshold audio; excluded
and <noise> frames are masked out of the loss)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data import manifest
from ..data.datasets import _frame
from ..data.store import FeatureStore
from ..nn.classifier import FrameClassifier, PhoneticCrossEntropy
from ..subtitles import read_csv
from ..tokens import NOISE, SILENCE_ID, TOKEN_INDEX
from . import common


def frame_targets(cfg, song_id: int, T: int, version: str | None = None) -> np.ndarray:
    """[T] class ids; -1 = masked (excluded/noise/out-of-inventory rows)."""
    y = np.full(T, SILENCE_ID, dtype=np.int64)
    rows = read_csv(manifest.subtitles_dir(cfg, version) / f"{song_id}.csv")
    for r in rows:
        s, e = _frame(r.start, cfg.frame_ms), min(T, _frame(r.end, cfg.frame_ms))
        if s >= e:
            continue
        if r.exclude or r.token == NOISE or r.token not in TOKEN_INDEX:
            y[s:e] = -1
        else:
            y[s:e] = TOKEN_INDEX[r.token]
    return y


def _load_split(cfg, ids, store, version, targets_dir: Path | None = None):
    """targets_dir: use forced-alignment frame labels (colab fa_labels/<id>.npy,
    S9 bootstrap) instead of timestamp-derived targets where available."""
    X, Y = [], []
    for song_id in ids:
        if not store.has(str(song_id)):
            continue
        feats = store.load(str(song_id))
        fa = (targets_dir / f"{song_id}.npy") if targets_dir else None
        if fa is not None and fa.is_file():
            y_fa = np.load(fa).astype(np.int64)
            n = min(len(feats), len(y_fa))
            y = np.full(len(feats), -1, dtype=np.int64)
            y[:n] = y_fa[:n]
            # keep exclusion masking from the label rows
            y_rows = frame_targets(cfg, song_id, len(feats), version)
            y[y_rows < 0] = -1
        else:
            y = frame_targets(cfg, song_id, len(feats), version)
        keep = y >= 0
        X.append(feats[keep])
        Y.append(y[keep])
    return np.concatenate(X), np.concatenate(Y)


def _maybe_project(cfg, dev, *Xs):
    """Apply the configured contrastive head (if any) to raw cached features,
    so training matches what the encoder emits at inference."""
    head_path = cfg.get("encoder.wav2vec2.projection_head", "")
    if not head_path:
        return Xs
    from ..nn.contrastive import ProjectionHead

    head = ProjectionHead.load(head_path).to(dev).eval()
    out = []
    with torch.inference_mode():
        for X in Xs:
            parts = [
                head(torch.from_numpy(X[s : s + 65536]).to(dev)).cpu().numpy()
                for s in range(0, len(X), 65536)
            ]
            out.append(np.concatenate(parts))
    print(f"[frame] projected features via {head_path} -> dim {out[0].shape[1]}")
    return tuple(out)


def train(cfg, version: str | None = None, name: str | None = None,
          epochs: int = 4, lr: float = 1e-3, batch: int = 4096,
          weak: tuple[np.ndarray, np.ndarray, float] | None = None,
          targets: str | None = None) -> Path:
    common.set_seed(int(cfg["train.seed"]))
    dev = common.device()
    run = common.run_dir(cfg, "frame", name)
    # read the RAW feature cache; the projection head (if any) is applied here
    from ..data.store import encoder_cache_id

    store = FeatureStore(cfg, encoder_id=encoder_cache_id(cfg, include_projection=False))
    train_ids, test_ids = manifest.split_ids(cfg, version)
    tdir = Path(targets) if targets else None
    Xtr, Ytr = _load_split(cfg, train_ids, store, version, targets_dir=tdir)
    Xte, Yte = _load_split(cfg, test_ids, store, version)  # test targets stay label-derived
    Xtr, Xte = _maybe_project(cfg, dev, Xtr, Xte)
    print(f"[frame] train {len(Ytr):,} frames / test {len(Yte):,} (dim {Xtr.shape[1]})")

    model = FrameClassifier(input_size=Xtr.shape[1]).to(dev)
    crit = PhoneticCrossEntropy(
        alpha=float(cfg["train.classifier.smooth_alpha"]),
        power=int(cfg["train.classifier.kernel_power"]),
    ).to(dev)
    # class-balance: down-weight silence frames to the mean lyric-class mass
    w = np.ones(len(Ytr), dtype=np.float32)
    sil_frac = float((Ytr == SILENCE_ID).mean())
    if sil_frac > 0.3:
        w[Ytr == SILENCE_ID] = 0.3 / sil_frac
    if weak is not None and len(weak[1]):
        Xw, Yw, w_weak = weak
        (Xw,) = _maybe_project(cfg, dev, Xw)
        ww = np.full(len(Yw), w_weak, dtype=np.float32)
        sil_w = float((Yw == SILENCE_ID).mean())
        if sil_w > 0.3:
            ww[Yw == SILENCE_ID] *= 0.3 / sil_w
        Xtr = np.concatenate([Xtr, Xw])
        Ytr = np.concatenate([Ytr, Yw])
        w = np.concatenate([w, ww])
        print(f"[frame] + {len(Yw):,} weak frames at weight {w_weak}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(Ytr)
    best_acc, best = -1.0, run / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        total = 0.0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            x = torch.from_numpy(Xtr[idx]).float().to(dev)
            y = torch.from_numpy(Ytr[idx]).to(dev)
            logits = model(x)
            logp = torch.log_softmax(logits, dim=-1)
            q = crit.targets[y]
            loss = (-(q * logp).sum(-1) * torch.from_numpy(w[idx]).to(dev)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        # frame accuracy on test
        model.eval()
        correct = 0
        with torch.inference_mode():
            for s in range(0, len(Yte), 65536):
                x = torch.from_numpy(Xte[s : s + 65536]).float().to(dev)
                correct += int((model(x).argmax(-1).cpu().numpy() == Yte[s : s + 65536]).sum())
        acc = correct / len(Yte)
        print(f"[frame] epoch {epoch}/{epochs} loss={total/max(1, n//batch):.4f} test_frame_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            common.save_checkpoint(best, model, model.hparams)
    common.write_eval(run, {"best_test_frame_acc": best_acc})
    print(f"[frame] best test frame acc {best_acc:.4f} -> {best}")
    return best
