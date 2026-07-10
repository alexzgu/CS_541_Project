"""Component interfaces. Two topologies (config [pipeline].mode):

  two_stage:  separator -> encoder -> segmenter -> classifier
  segmental:  separator -> encoder(+aux) -> decoder            (P3)

Each slot is filled by name through kashi.registry, so implementations are
swappable from config without touching pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Span:
    """A frame-index interval [start, end)."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class SeparationResult:
    vocals: Path
    instrumental: Path | None = None


@dataclass
class FrameAux:
    """Per-frame side information available to segmenters/decoders."""

    rms_db: np.ndarray | None = None      # [T] energy in dBFS
    voicing: np.ndarray | None = None     # [T] periodicity in [0,1]
    boundary_logits: np.ndarray | None = None  # [T] Model-1 logits
    extras: dict = field(default_factory=dict)


@dataclass
class Boundary:
    """A candidate boundary event (used by realign, P2)."""

    time_s: float
    confidence: float
    std_ms: float | None = None
    source: str = ""


@runtime_checkable
class Separator(Protocol):
    def separate(self, wav: Path, out_dir: Path) -> SeparationResult: ...


@runtime_checkable
class Encoder(Protocol):
    dim: int
    frame_ms: int

    def encode(self, wave: np.ndarray, sr: int) -> np.ndarray: ...


@runtime_checkable
class Segmenter(Protocol):
    def segment(self, feats: np.ndarray, aux: FrameAux | None = None) -> list[Span]: ...


@runtime_checkable
class Classifier(Protocol):
    def classify(
        self, feats: np.ndarray, spans: list[Span]
    ) -> list[tuple[str, float]]: ...


@runtime_checkable
class Decoder(Protocol):
    def decode(self, feats: np.ndarray, aux: FrameAux | None = None) -> list: ...
