"""kashi — textless syllable-level transcription of Japanese songs.

Swappable pipeline components (separator, encoder, segmenter, classifier,
decoder) wired by a registry (`kashi.registry`) and a layered TOML config
(`kashi.config`). `kashi.pipeline` runs audio -> subtitles; `kashi.cli`
exposes everything as commands.

Design constraints (see docs/pipeline_specification.md):
- textless: no transcript at inference,
- no forced alignment anywhere (incl. dataset cleaning),
- explicit probabilistic structure over black-box behavior.
"""

__version__ = "0.2.0"
