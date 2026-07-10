"""Segmental decoders (segmental topology; lands in P3 — spec §6)."""

from __future__ import annotations

from ..registry import register


@register("decoder", "segmental")
class SegmentalDecoder:
    def __init__(self, cfg):
        raise SystemExit(
            "decoder 'segmental' lands in P3 (semi-Markov Viterbi over Model-2f "
            "frame posteriors + duration prior + Model-1 boundary evidence) — "
            "see ROADMAP.md and docs/pipeline_specification.md §6"
        )
