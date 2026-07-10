"""Vocal/instrumental separators. Swap via [pipeline].separator = none | uvr | demucs.

- uvr wraps the pip `audio-separator` package: ANY of its checkpoints
  (roformer, MDX-Net, VR, ...) is selectable via separator.uvr.model_filename.
- demucs shells out to `python -m demucs` (its CLI is its stable API).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ..registry import register
from .base import SeparationResult


@register("separator", "none")
class NoSeparator:
    """Pass-through: input is treated as already-isolated vocals."""

    def __init__(self, cfg):
        pass

    def separate(self, wav: Path, out_dir: Path) -> SeparationResult:
        return SeparationResult(vocals=Path(wav))


@register("separator", "uvr")
class UVRSeparator:
    def __init__(self, cfg):
        self.model_filename = cfg["separator.uvr.model_filename"]

    def separate(self, wav: Path, out_dir: Path) -> SeparationResult:
        try:
            from audio_separator.separator import Separator as UVR
        except ImportError as e:
            raise SystemExit(
                "separator 'uvr' needs the [uvr] extra: pip install 'kashi[uvr]' "
                "(package audio-separator)"
            ) from e
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sep = UVR(output_dir=str(out_dir))
        sep.load_model(model_filename=self.model_filename)
        outputs = [Path(p) if Path(p).is_absolute() else out_dir / p for p in sep.separate(str(wav))]
        vocals = next((p for p in outputs if "vocal" in p.name.lower()), None)
        inst = next((p for p in outputs if "instrument" in p.name.lower()), None)
        if vocals is None:
            raise RuntimeError(f"uvr produced no vocals stem among {outputs}")
        return SeparationResult(vocals=vocals, instrumental=inst)


@register("separator", "demucs")
class DemucsSeparator:
    def __init__(self, cfg):
        self.model = cfg["separator.demucs.model"]

    def separate(self, wav: Path, out_dir: Path) -> SeparationResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-n", self.model,
            "-o", str(out_dir),
            str(wav),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise SystemExit(
                "separator 'demucs' needs the [demucs] extra: pip install 'kashi[demucs]'"
            ) from e
        stem_dir = out_dir / self.model / Path(wav).stem
        vocals = stem_dir / "vocals.wav"
        inst = stem_dir / "no_vocals.wav"
        if not vocals.is_file():
            raise RuntimeError(f"demucs produced no vocals at {vocals}")
        return SeparationResult(vocals=vocals, instrumental=inst if inst.is_file() else None)


def separate_files(cfg, inputs: list[Path], out_dir: Path, name: str | None = None) -> list[SeparationResult]:
    from ..registry import create

    sep = create(cfg, "separator", name)
    results = []
    for path in inputs:
        print(f"[separate] {path}")
        results.append(sep.separate(Path(path), out_dir))
    return results


__all__ = ["NoSeparator", "UVRSeparator", "DemucsSeparator", "separate_files", "shutil"]
