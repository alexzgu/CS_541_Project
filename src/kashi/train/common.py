"""Shared training utilities: seeding, device, run directories with config snapshots."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir(cfg, kind: str, name: str | None = None) -> Path:
    stamp = name or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = cfg.runs_dir / f"{kind}-{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.toml").write_text(cfg.dump_toml())
    return d


def save_checkpoint(path: Path, model: torch.nn.Module, hparams: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hparams": hparams or {}, "state_dict": model.state_dict()}, path)
    return path


def write_eval(run: Path, payload: dict) -> Path:
    out = run / "eval.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    return out


def append_leaderboard(cfg, row: dict) -> None:
    import csv

    path = cfg.runs_dir / "leaderboard.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
