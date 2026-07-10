"""Feature cache: artifacts/features/<encoder-id>/<frame_ms>ms/<key>.{npy,pt}.

The legacy wav2vec2 tensors (models/tensors/songs_20ms/*.pt, ~11 GB) are
adopted in place via `kashi encode --from-legacy` (symlinks, no copy).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def encoder_cache_id(cfg, encoder_name: str | None = None, include_projection: bool = True) -> str:
    name = encoder_name or cfg["pipeline.encoder"]
    if name == "wav2vec2":
        detail = _sanitize(cfg["encoder.wav2vec2.checkpoint"])
        head = cfg.get("encoder.wav2vec2.projection_head", "")
        if head and include_projection:
            detail += "+proj-" + _sanitize(Path(head).stem)
    elif name == "mel":
        detail = f"mel{cfg['encoder.mel.n_mels']}"
    else:
        detail = name
    return f"{name}_{detail}"


class FeatureStore:
    def __init__(self, cfg, encoder_id: str | None = None, frame_ms: int | None = None):
        self.frame_ms = frame_ms or cfg.frame_ms
        self.encoder_id = encoder_id or encoder_cache_id(cfg)
        self.dir = cfg.artifacts_dir / "features" / self.encoder_id / f"{self.frame_ms}ms"

    def path(self, key: str) -> Path:
        npy = self.dir / f"{key}.npy"
        if npy.exists():
            return npy
        pt = self.dir / f"{key}.pt"
        if pt.exists():
            return pt
        return npy

    def has(self, key: str) -> bool:
        return (self.dir / f"{key}.npy").exists() or (self.dir / f"{key}.pt").exists()

    def save(self, key: str, feats: np.ndarray) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{key}.npy"
        np.save(path, np.asarray(feats, dtype=np.float32))
        return path

    def load(self, key: str) -> np.ndarray:
        path = self.path(key)
        if not path.exists():
            raise FileNotFoundError(f"no cached features for {key!r} under {self.dir}")
        if path.suffix == ".pt":
            import torch

            t = torch.load(path, map_location="cpu", weights_only=True)
            return t.numpy().astype(np.float32)
        return np.load(path).astype(np.float32)

    def keys(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted({p.stem for p in self.dir.iterdir() if p.suffix in (".npy", ".pt")})

    def adopt_legacy(self, legacy_dir: Path) -> int:
        """Symlink legacy per-song .pt tensors into this cache. Returns count."""
        legacy_dir = Path(legacy_dir)
        if not legacy_dir.is_dir():
            raise FileNotFoundError(legacy_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for src in sorted(legacy_dir.glob("*.pt")):
            dst = self.dir / src.name
            if dst.exists() or dst.is_symlink():
                continue
            dst.symlink_to(src.resolve())
            n += 1
        return n
