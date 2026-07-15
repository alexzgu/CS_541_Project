"""End-to-end transcription: any audio/video file -> timed hiragana subtitles.

    input -> ffmpeg 16k mono wav -> separator -> encoder(+aux)
          -> [two_stage: segmenter -> classifier | segmental: decoder]
          -> silence fill -> SRT/VTT/ASS/CSV

Every stage is a registry component chosen by config, so swapping the vocal
separator (or anything else) never touches this file. Textless: no transcript
is consumed anywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio as audio_mod
from .components.base import FrameAux
from .registry import create
from .subtitles import Segment, write_outputs
from .tokens import SILENCE

ProgressCb = Callable[[str, float], None]


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    out_files: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def _fill_gaps(segments: list[Segment], total_s: float, min_sil: float = 0.02) -> list[Segment]:
    """Insert <silence> rows so the output covers [0, total_s] contiguously."""
    out: list[Segment] = []
    t = 0.0
    for seg in sorted(segments, key=lambda s: s.start):
        if seg.start - t >= min_sil:
            out.append(Segment(t, seg.start, SILENCE))
        out.append(seg)
        t = max(t, seg.end)
    if total_s - t >= min_sil:
        out.append(Segment(t, total_s, SILENCE))
    return out


def transcribe(
    cfg,
    input_path: str | Path,
    out_dir: str | Path | None = None,
    formats: list[str] | None = None,
    romaji: bool = False,
    separate: bool | None = None,
    progress: ProgressCb | None = None,
) -> TranscriptionResult:
    input_path = Path(input_path)
    out_dir = Path(out_dir) if out_dir else cfg.runs_dir / "transcribe" / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["srt", "vtt", "csv"]
    timings: dict[str, float] = {}

    def tick(stage: str, frac: float) -> None:
        if progress:
            progress(stage, frac)

    def timed(stage: str, frac: float, fn):
        tick(stage, frac)
        t0 = time.time()
        out = fn()
        timings[stage] = round(time.time() - t0, 2)
        return out

    sr = cfg.sample_rate
    wav16 = timed("extract-audio", 0.05,
                  lambda: audio_mod.ffmpeg_to_wav(input_path, out_dir / "input_16k.wav", sr=sr))

    do_separate = separate if separate is not None else cfg["pipeline.separator"] != "none"
    if do_separate:
        sep = create(cfg, "separator")
        vocals_path = timed("separate-vocals", 0.15,
                            lambda: sep.separate(wav16, out_dir / "separated").vocals)
    else:
        vocals_path = wav16

    wave = audio_mod.load_audio(vocals_path, sr=sr)
    total_s = len(wave) / sr

    encoder = create(cfg, "encoder")
    feats = timed("encode", 0.45, lambda: encoder.encode(wave, sr))
    aux = FrameAux(rms_db=audio_mod.log_rms_db(wave, sr, cfg.frame_ms)[: len(feats)],
                   extras={"wave": wave, "sr": sr})

    mode = cfg["pipeline.mode"]
    frame_s = cfg.frame_ms / 1000.0
    if mode == "two_stage":
        segmenter = create(cfg, "segmenter")
        spans = timed("segment", 0.7, lambda: segmenter.segment(feats, aux))
        classifier = create(cfg, "classifier")
        labeled = timed("classify", 0.85, lambda: classifier.classify(feats, spans))
        segments = [
            Segment(s.start * frame_s, s.end * frame_s, token, confidence=conf)
            for s, (token, conf) in zip(spans, labeled)
            if token
        ]
    elif mode == "segmental":
        decoder = create(cfg, "decoder")
        segments = timed("decode", 0.8, lambda: decoder.decode(feats, aux))
    else:
        raise SystemExit(f"unknown pipeline.mode {mode!r}")

    lyric = [s for s in segments if not s.is_silence]
    segments = _fill_gaps(lyric, total_s)

    out_files = timed(
        "write", 0.95,
        lambda: write_outputs(segments, out_dir, input_path.stem, formats, romaji_line=romaji,
                              display_lead_ms=float(cfg.get("subtitles.display_lead_ms", 0.0))),
    )
    tick("done", 1.0)
    return TranscriptionResult(segments=segments, out_files=out_files, timings=timings)
