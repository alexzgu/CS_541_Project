"""P1 baselines (ROADMAP §4) and the end-to-end pipeline evaluation.

(a) legacy Model 2 on ground-truth segments (validates the port),
(b) energy + legacy LSTM end-to-end (first honest end-to-end number),
(c) paper Model 1 retrained VERBATIM from the recovered notebook — including
    its three bugs — to reproduce paper F1 ~= 0.41 under the paper's metric,
(d) the bug-fixed Model 1 on identical data/labels — quantifies the bug cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data import manifest
from ..data.datasets import BreakDataset
from ..data.store import FeatureStore
from ..subtitles import read_csv
from ..train import common
from . import gold as gold_mod
from . import metrics as M


def _record(cfg, name: str, payload: dict) -> Path:
    out_dir = cfg.runs_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    common.append_leaderboard(cfg, {
        "run": f"baseline_{name}",
        "ser": payload.get("pooled", {}).get("ser", payload.get("ser", "")),
        "timed_token_f1": payload.get("pooled", {}).get("timed_token_f1", ""),
        "boundary_f1_50ms": payload.get("pooled", {}).get("boundary@50ms_f1",
                                                          payload.get("boundary_f1_50ms", "")),
        "accuracy": payload.get("accuracy", ""),
        "note": payload.get("note", ""),
    })
    print(f"[baseline {name}] -> {path}")
    return path


# ---------------------------------------------------------------------------
# Pipeline evaluation (used by baseline b and `kashi eval pipeline`)
# ---------------------------------------------------------------------------

def evaluate_pipeline(cfg, split: str = "test", version: str | None = None,
                      gold_only: bool = False, song_ids: list[int] | None = None) -> dict:
    from ..pipeline import transcribe

    if song_ids is None:
        train_ids, test_ids = manifest.split_ids(cfg, version)
        song_ids = test_ids if split == "test" else train_ids
    windows = gold_mod.read_windows(cfg)
    reports: dict[int, dict] = {}
    for song_id in song_ids:
        paths = manifest.song_paths(cfg, song_id)
        if not paths.vocals.is_file():
            print(f"[eval pipeline] song {song_id}: no vocals, skipped")
            continue
        gold_file = gold_mod.gold_dir(cfg) / f"{song_id}.csv"
        if gold_only:
            if not gold_file.is_file():
                continue
            ref = read_csv(gold_file)
            ref_kind = "gold"
        else:
            ref = read_csv(manifest.subtitles_dir(cfg, version) / f"{song_id}.csv")
            ref_kind = version or cfg["data.version"]
        result = transcribe(cfg, paths.vocals, out_dir=cfg.runs_dir / "eval" / str(song_id),
                            formats=["csv"], separate=False)
        pred = result.segments
        if gold_only:
            wins = [w for w in windows if w["song_id"] == song_id]
            rep_parts = []
            for w in wins:
                p = M.clip_to_window(pred, w["start"], w["end"])
                r = M.clip_to_window(ref, w["start"], w["end"])
                rep_parts.append(M.song_report(p, r, cfg["eval.tolerances_ms"]))
            if not rep_parts:
                continue
            # merge window parts by summing edit distances / pooling
            rep = rep_parts[0] if len(rep_parts) == 1 else M.pool_reports(
                {i: r for i, r in enumerate(rep_parts)}
            )
        else:
            rep = M.song_report(pred, ref, cfg["eval.tolerances_ms"])
        rep["reference"] = ref_kind
        reports[song_id] = rep
        s = rep.get("ser", float("nan"))
        print(f"[eval pipeline] song {song_id}: SER={s:.3f} vs {ref_kind}")
    pooled = M.pool_reports(reports)
    return {"split": split, "gold_only": gold_only, "per_song": reports, "pooled": pooled,
            "pipeline": {k: cfg[f"pipeline.{k}"] for k in
                         ("mode", "separator", "encoder", "segmenter", "classifier")}}


# ---------------------------------------------------------------------------
# (c) verbatim paper Model 1 — bugs included, for the reproduction baseline
# ---------------------------------------------------------------------------

class _VerbatimModel1(torch.nn.Module):
    """Exact recovered architecture (Transformer_Wave2Vec_AAAAA.ipynb):
    no batch_first (=> attention over a length-1 sequence), no positional
    encoding, Linear(768,1) + sigmoid."""

    def __init__(self, input_dim: int = 768, n_heads: int = 8, num_layers: int = 2,
                 hidden_dim: int = 512):
        super().__init__()
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim,
                activation="relu",
            ),
            num_layers=num_layers,
        )
        self.output_layer = torch.nn.Linear(input_dim, 1)

    def forward(self, x):  # x: [1, T, 768] interpreted as seq=1, batch=T — the bug
        encoded = self.transformer_encoder(x)
        return torch.sigmoid(self.output_layer(encoded.squeeze(1)).squeeze(-1))


def _expand_ones_verbatim(arr: np.ndarray) -> np.ndarray:
    """Legacy label expansion INCLUDING the negative-index wraparound bug."""
    expanded = np.zeros_like(arr).astype(float)
    length = len(arr)
    for i in range(length):
        if arr[i] == 1:
            expanded[i - 2] = min(1, expanded[i - 2] + 0.2)
            expanded[i - 1] = min(1, expanded[i - 1] + 0.35)
            expanded[i] = 1
            if i + 1 < length:
                expanded[i + 1] = min(1, expanded[i + 1] + 1)
            if i + 2 < length:
                expanded[i + 2] = min(1, expanded[i + 2] + 0.8)
            if i + 3 < length:
                expanded[i + 3] = min(1, expanded[i + 3] + 0.25)
    return expanded


def _paper_metric(outputs: np.ndarray, targets: np.ndarray, thr: float = 0.45) -> dict:
    """The paper's frame-level metric: threshold both the sigmoid outputs and
    the EXPANDED soft targets at `thr`, then P/R/F1 over all frames."""
    o = (outputs >= thr).astype(int)
    t = (targets >= thr).astype(int)
    tp = int(((o == 1) & (t == 1)).sum())
    fp = int(((o == 1) & (t == 0)).sum())
    fn = int(((o == 0) & (t == 1)).sum())
    tn = int(((o == 0) & (t == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
            "confusion": [[tp, fp], [fn, tn]]}


def baseline_c(cfg, epochs: int = 10) -> dict:
    common.set_seed(int(cfg["train.seed"]))
    dev = common.device()
    store = FeatureStore(cfg)
    train_ids, test_ids = manifest.split_ids(cfg)
    train_ds = BreakDataset(cfg, train_ids, store)
    test_ds = BreakDataset(cfg, test_ids, store)

    model = _VerbatimModel1().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = torch.nn.BCELoss()
    order = np.arange(len(train_ds))
    # NB: the verbatim architecture treats every frame independently (the bug),
    # so frame-chunking is mathematically identical to full-song batches and
    # keeps the 6 GB GPU happy.
    chunk = 4096
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(order)
        total = n_steps = 0
        for i in order:
            feats, breaks, _ = train_ds[int(i)]
            targets_full = torch.from_numpy(
                _expand_ones_verbatim(breaks.numpy().astype(int))
            ).float()
            for s in range(0, len(feats), chunk):
                f_c = feats[s : s + chunk]
                if len(f_c) < 8:
                    continue
                probs = model(f_c[None].float().to(dev))
                loss = bce(probs, targets_full[None, s : s + chunk].to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss)
                n_steps += 1
        print(f"[baseline c] epoch {epoch}/{epochs} loss={total/max(1, n_steps):.4f}")

    # paper metric + tolerance boundary-F1 on the test split
    from ..nn.segmenter import boundary_f1, pick_boundaries

    model.eval()
    all_o, all_t = [], []
    bscores = []
    with torch.inference_mode():
        for feats, breaks, _ in test_ds:
            probs = np.concatenate([
                model(feats[s : s + chunk][None].float().to(dev))[0].cpu().numpy()
                for s in range(0, len(feats), chunk) if len(feats[s : s + chunk]) > 0
            ])
            targets = _expand_ones_verbatim(breaks.numpy().astype(int))
            all_o.append(probs)
            all_t.append(targets)
            pred_b = pick_boundaries(probs, threshold=0.45, nms=3)
            true_b = np.flatnonzero(breaks.numpy()).tolist()
            bscores.append(boundary_f1(pred_b, true_b, tol_frames=3, frame_ms=cfg.frame_ms))
    paper = _paper_metric(np.concatenate(all_o), np.concatenate(all_t))
    payload = {
        "note": "paper Model 1 retrained VERBATIM (bugs included); paper reported F1 0.4078",
        "paper_metric": paper,
        "boundary_f1_50ms": float(np.mean([s.f1 for s in bscores])),
        "boundary_mean_abs_ms": float(np.nanmean([s.mean_abs_ms for s in bscores])),
        "epochs": epochs,
    }
    _record(cfg, "c_model1_verbatim", payload)
    return payload


def baseline_d(cfg) -> dict:
    from ..train import segmenter as seg_train

    best = seg_train.train(cfg, name="baseline-d")
    rep = seg_train.evaluate(cfg, best, split="test")

    # also the paper metric for apples-to-paper comparison
    from ..nn.segmenter import TransformerSegmenter, soft_break_labels

    dev = common.device()
    payload_ckpt = torch.load(best, map_location="cpu", weights_only=True)
    model = TransformerSegmenter(**payload_ckpt["hparams"]).to(dev).eval()
    model.load_state_dict(payload_ckpt["state_dict"])
    store = FeatureStore(cfg)
    _, test_ids = manifest.split_ids(cfg)
    all_o, all_t = [], []
    with torch.inference_mode():
        for feats, breaks, _ in BreakDataset(cfg, test_ids, store):
            logits = model.frame_logits(feats.numpy())
            all_o.append(1 / (1 + np.exp(-logits)))
            all_t.append(soft_break_labels(breaks.numpy()))
    paper = _paper_metric(np.concatenate(all_o), np.concatenate(all_t))
    payload = {
        "note": "bug-fixed Model 1 (batch_first + positions + edge-guarded labels), same data",
        "paper_metric": paper,
        "boundary_f1_50ms": rep["boundary_f1"],
        "boundary_mean_abs_ms": rep["mean_abs_ms"],
        "checkpoint": str(best),
        "loss": cfg["train.segmenter.loss"],
    }
    _record(cfg, "d_model1_fixed", payload)
    return payload


def baseline_a(cfg) -> dict:
    from ..train.classifier import evaluate

    rep = evaluate(cfg, split="test", legacy_index=True)
    rep["note"] = "legacy Model 2 on ground-truth segments; paper reported acc 0.536"
    _record(cfg, "a_classifier_legacy", rep)
    return rep


def baseline_b(cfg) -> dict:
    rep = evaluate_pipeline(cfg, split="test")
    rep["note"] = "day-1 end-to-end: energy segmenter + legacy LSTM (first honest number)"
    _record(cfg, "b_endtoend_energy_lstm", rep)
    return rep
