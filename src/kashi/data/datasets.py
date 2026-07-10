"""Torch datasets over cached features + subtitle labels.

Break labels and segment labels are derived on the fly from the subtitle CSVs
(no pre-baked segment_breaks/segment_index files to drift out of sync). The
legacy segment_index.csv is still readable for exact baseline reproduction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..subtitles import Segment, read_csv
from ..tokens import NOISE, TOKEN_INDEX
from . import manifest
from .store import FeatureStore


def _frame(t_seconds: float, frame_ms: int) -> int:
    """Second -> frame index, robust to float ms (12.72 -> 12719.999...)."""
    return int(round(t_seconds * 1000)) // frame_ms


def song_frame_labels(
    segments: list[Segment], frame_ms: int, T: int
) -> tuple[np.ndarray, np.ndarray]:
    """(breaks[T] bool, valid[T] bool): break at each internal row end; frames
    inside excluded or <noise> rows are masked out of the loss."""
    breaks = np.zeros(T, dtype=bool)
    valid = np.ones(T, dtype=bool)
    for i, seg in enumerate(segments):
        if i < len(segments) - 1:
            b = _frame(seg.end, frame_ms)
            if 0 <= b < T:
                breaks[b] = True
        if seg.exclude or seg.token == NOISE:
            s, e = _frame(seg.start, frame_ms), _frame(seg.end, frame_ms)
            valid[max(0, s):max(0, min(T, e))] = False
    return breaks, valid


class BreakDataset(Dataset):
    """Per-song items for Model 1: (features [T,D], breaks[T], valid[T])."""

    def __init__(self, cfg, song_ids: list[int], store: FeatureStore | None = None,
                 version: str | None = None):
        self.cfg = cfg
        self.store = store or FeatureStore(cfg)
        self.version = version or cfg["data.version"]
        self.ids = [i for i in song_ids if self.store.has(str(i))]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        song_id = self.ids[idx]
        feats = self.store.load(str(song_id))
        segs = read_csv(manifest.subtitles_dir(self.cfg, self.version) / f"{song_id}.csv")
        breaks, valid = song_frame_labels(segs, self.cfg.frame_ms, len(feats))
        return (
            torch.from_numpy(feats),
            torch.from_numpy(breaks),
            torch.from_numpy(valid),
        )


class SegmentDataset(Dataset):
    """Per-segment items for Model 2: (features [n,D], class index).

    Segment frames follow the legacy convention: [floor(start/h), floor(end/h))
    with contiguous rows (a row's end == next row's start after label building).
    """

    def __init__(self, cfg, song_ids: list[int], store: FeatureStore | None = None,
                 version: str | None = None, include_excluded: bool = False):
        self.items: list[tuple[torch.Tensor, int]] = []
        store = store or FeatureStore(cfg)
        version = version or cfg["data.version"]
        frame_ms = cfg.frame_ms
        for song_id in song_ids:
            if not store.has(str(song_id)):
                continue
            feats = torch.from_numpy(store.load(str(song_id)))
            for seg in read_csv(manifest.subtitles_dir(cfg, version) / f"{song_id}.csv"):
                if (seg.exclude and not include_excluded) or seg.token not in TOKEN_INDEX:
                    continue
                s, e = _frame(seg.start, frame_ms), _frame(seg.end, frame_ms)
                piece = feats[max(0, s):max(0, e)]
                if len(piece) == 0:
                    continue
                self.items.append((piece, TOKEN_INDEX[seg.token]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class LegacyIndexDataset(Dataset):
    """Segments from the legacy clean/syllables/segment_index.csv
    (index,file,start,end,token[,pred]; times in int ms) — for exact
    reproduction of the paper's Model-2 evaluation."""

    def __init__(self, cfg, song_ids: list[int], store: FeatureStore | None = None,
                 index_path: str | Path | None = None):
        store = store or FeatureStore(cfg)
        index_path = Path(index_path or cfg.path("legacy_segment_index"))
        frame_ms = cfg.frame_ms
        df = pd.read_csv(index_path)
        wanted = set(song_ids)
        self.items: list[tuple[torch.Tensor, int]] = []
        for song_id, group in df.groupby("file"):
            if int(song_id) not in wanted or not store.has(str(int(song_id))):
                continue
            feats = torch.from_numpy(store.load(str(int(song_id))))
            for _, row in group.iterrows():
                s = int(row["start"]) // frame_ms
                e = int(row["end"]) // frame_ms
                piece = feats[s:e]
                if len(piece) == 0 or row["token"] not in TOKEN_INDEX:
                    continue
                self.items.append((piece, TOKEN_INDEX[row["token"]]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_segments(batch):
    """Legacy collate: sort by length desc (enforce_sorted=True), pad, return
    (padded [B,T,D], labels [B], lengths [B])."""
    from torch.nn.utils.rnn import pad_sequence

    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    order = torch.argsort(lengths, descending=True)
    sequences = [sequences[i] for i in order]
    labels = torch.tensor([labels[i] for i in order], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True), labels, lengths[order]
