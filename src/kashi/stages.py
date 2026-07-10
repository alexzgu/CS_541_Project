"""DAG stage definitions (`kashi run <stage>`). Grows with each phase."""

from __future__ import annotations

from .dag import stage
from .data.store import FeatureStore


def _legacy_dir(cfg):
    return [cfg.path("legacy_tensors")]


def _cache_dir(cfg):
    return [FeatureStore(cfg).dir]


@stage(
    "adopt-legacy",
    inputs=_legacy_dir,
    outputs=_cache_dir,
    config_keys=["pipeline.encoder", "encoder.wav2vec2.checkpoint", "data.frame_ms"],
)
def adopt_legacy_stage(cfg) -> None:
    from .data.encode import adopt_legacy

    adopt_legacy(cfg)


@stage(
    "encode",
    inputs=lambda cfg: [cfg.data_dir / "clean" / "audio" / "vocals"],
    outputs=_cache_dir,
    config_keys=["pipeline.encoder", "encoder.wav2vec2.checkpoint", "encoder.mel.n_mels",
                 "data.frame_ms", "data.sample_rate"],
)
def encode_stage(cfg) -> None:
    from .data.encode import encode_targets

    encode_targets(cfg)


# `kashi run eval` and friends are wired as their phases land (P1+).
