"""Train / evaluate Model 2 (LSTM syllable classifier).

loss = "phonetic" uses the partial-credit smoothing from kashi.phonetics;
evaluation reports micro accuracy and the kernel-weighted ('partial credit')
score. `--init <path>` warm-starts from a legacy checkpoint. Evaluation with
--legacy-index reproduces the paper's segment slicing exactly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import manifest
from ..data.datasets import LegacyIndexDataset, SegmentDataset, collate_segments
from ..data.store import FeatureStore
from ..nn.classifier import LSTMClassifier, PhoneticCrossEntropy
from ..phonetics import kernel_matrix
from ..tokens import NUM_TOKENS
from . import common


def _make_model(cfg, input_size: int | None = None) -> LSTMClassifier:
    return LSTMClassifier(
        input_size=input_size or int(cfg["classifier.lstm.input_size"]),
        hidden_size=int(cfg["classifier.lstm.hidden_size"]),
        num_layers=int(cfg["classifier.lstm.num_layers"]),
        num_classes=NUM_TOKENS,
        dropout=float(cfg["classifier.lstm.dropout"]),
    )


def _load_state(path: str | Path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state


@torch.inference_mode()
def evaluate(cfg, checkpoint: str | Path | None = None, split: str = "test",
             version: str | None = None, legacy_index: bool = False) -> dict:
    dev = common.device()
    model = _make_model(cfg).to(dev).eval()
    ckpt = checkpoint or cfg.root / cfg["classifier.lstm.checkpoint"]
    model.load_state_dict(_load_state(ckpt))

    train_ids, test_ids = manifest.split_ids(cfg, version)
    ids = test_ids if split == "test" else train_ids
    store = FeatureStore(cfg)
    if legacy_index:
        ds = LegacyIndexDataset(cfg, ids, store)
    else:
        ds = SegmentDataset(cfg, ids, store, version=version)
    if len(ds) == 0:
        raise SystemExit(
            f"no evaluation segments (split={split}); is the feature cache populated? "
            f"(kashi encode --from-legacy)"
        )
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_segments)

    K = torch.from_numpy(kernel_matrix()).float()
    correct = total = 0
    pc_sum = 0.0
    for padded, labels, lengths in loader:
        logits = model(padded.to(dev), lengths)
        pred = logits.argmax(dim=-1).cpu()
        correct += int((pred == labels).sum())
        pc_sum += float(K[pred, labels].sum())
        total += len(labels)
    return {
        "split": split,
        "segments": total,
        "accuracy": correct / total,
        "partial_credit": pc_sum / total,
        "checkpoint": str(ckpt),
        "legacy_index": legacy_index,
    }


def train(cfg, version: str | None = None, init: str | None = None,
          name: str | None = None, input_size: int | None = None) -> Path:
    common.set_seed(int(cfg["train.seed"]))
    dev = common.device()
    run = common.run_dir(cfg, "classifier", name)

    model = _make_model(cfg, input_size).to(dev)
    if init:
        model.load_state_dict(_load_state(init))

    train_ids, test_ids = manifest.split_ids(cfg, version)
    store = FeatureStore(cfg)
    train_ds = SegmentDataset(cfg, train_ids, store, version=version)
    if len(train_ds) == 0:
        raise SystemExit("no training segments; run `kashi encode` first")
    loader = DataLoader(
        train_ds, batch_size=int(cfg["train.classifier.batch_size"]),
        shuffle=True, collate_fn=collate_segments,
    )

    if cfg["train.classifier.loss"] == "phonetic":
        criterion = PhoneticCrossEntropy(
            alpha=float(cfg["train.classifier.smooth_alpha"]),
            power=int(cfg["train.classifier.kernel_power"]),
        ).to(dev)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train.classifier.lr"]),
        weight_decay=float(cfg["train.classifier.weight_decay"]),
    )

    best_acc, best_path = -1.0, run / "best.pt"
    epochs = int(cfg["train.classifier.epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for padded, labels, lengths in loader:
            opt.zero_grad()
            logits = model(padded.to(dev), lengths)
            loss = criterion(logits, labels.to(dev))
            loss.backward()
            opt.step()
            total_loss += float(loss)
        common.save_checkpoint(run / f"epoch{epoch}.pt", model, model_hparams(cfg, input_size))
        metrics = evaluate(cfg, run / f"epoch{epoch}.pt", split="test", version=version)
        print(f"[classifier] epoch {epoch}/{epochs} loss={total_loss/len(loader):.4f} "
              f"test_acc={metrics['accuracy']:.4f} pc={metrics['partial_credit']:.4f}")
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            common.save_checkpoint(best_path, model, model_hparams(cfg, input_size))
    common.write_eval(run, {"best_test_accuracy": best_acc})
    print(f"[classifier] best test acc {best_acc:.4f} -> {best_path}")
    return best_path


def model_hparams(cfg, input_size: int | None = None) -> dict:
    return dict(
        input_size=input_size or int(cfg["classifier.lstm.input_size"]),
        hidden_size=int(cfg["classifier.lstm.hidden_size"]),
        num_layers=int(cfg["classifier.lstm.num_layers"]),
        dropout=float(cfg["classifier.lstm.dropout"]),
    )
