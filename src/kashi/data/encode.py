"""Populate the feature cache for dataset songs or arbitrary unlabeled audio."""

from __future__ import annotations

from pathlib import Path

from .. import audio as audio_mod
from ..registry import create
from . import manifest
from .store import FeatureStore

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")


def encode_targets(
    cfg,
    songs: list[int] | None = None,
    unlabeled: str | Path | None = None,
    force: bool = False,
) -> int:
    store = FeatureStore(cfg)
    targets: list[Path] = []
    if unlabeled:
        targets = sorted(
            p for p in Path(unlabeled).iterdir() if p.suffix.lower() in AUDIO_SUFFIXES
        )
    else:
        ids = songs if songs is not None else manifest.song_ids(cfg)
        for i in ids:
            p = manifest.song_paths(cfg, int(i))
            path = p.vocals if p.vocals.is_file() else p.raw_audio
            if path.is_file():
                targets.append(path)
            else:
                print(f"missing audio: song {i}")
    encoder = create(cfg, "encoder")
    n = 0
    for path in targets:
        key = audio_mod.content_key(path)
        if store.has(key) and not force:
            continue
        wave = audio_mod.load_audio(path, sr=cfg.sample_rate)
        feats = encoder.encode(wave, cfg.sample_rate)
        store.save(key, feats)
        print(f"[encode] {path.name} -> {key} {feats.shape}")
        n += 1
    return n


def adopt_legacy(cfg, src: str | Path | None = None) -> int:
    """Adopt the legacy tensor cache by symlink.

    VERIFIED 2026-07-09: despite the directory name (songs_20ms), the tensors
    in this repo are the 10 ms resample-trick variant (frame count x 10 ms ==
    audio duration on every checked song). They are therefore adopted into the
    10 ms cache; the 20 ms cache is populated by `kashi encode` (re-encoding
    reproduces the legacy semantics: full-song normalisation + one forward).
    """
    frame_ms = int(cfg.get("paths.legacy_tensors_frame_ms", 10))
    store = FeatureStore(cfg, frame_ms=frame_ms)
    src = Path(src) if src else cfg.path("legacy_tensors")
    n = store.adopt_legacy(src)
    print(f"adopted {n} legacy tensors from {src} into {store.dir} "
          f"(NOTE: legacy tensors are {frame_ms} ms-resolution)")
    return n
