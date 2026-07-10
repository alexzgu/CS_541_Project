"""Colab offload (SIGNOFFS S10): pack inputs for Drive, integrate results back.

`kashi colab pack`   -> artifacts/colab_pack.zip (vocals + labels + tokens for
                        all labeled songs; upload to Drive as MyDrive/kashi/)
`kashi colab integrate <dir-or-zip>` -> installs returned artifacts under
                        artifacts/colab/ (ctc_model/, fa_labels/, fa_segments/,
                        metrics.json) and prints next steps.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from . import manifest


def pack(cfg) -> Path:
    out = cfg.artifacts_dir / "colab_pack.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    labeled = manifest.labeled_ids(cfg)
    from ..tokens import TOKENS

    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.writestr("tokens.txt", "\n".join(TOKENS))
        train, test = manifest.split_ids(cfg)
        z.writestr("split.txt", "train:" + ",".join(map(str, train)) +
                   "\ntest:" + ",".join(map(str, test)))
        for song_id in labeled:
            p = manifest.song_paths(cfg, song_id)
            if p.vocals.is_file():
                z.write(p.vocals, f"vocals/{song_id}.mp3")
            sub = manifest.subtitles_dir(cfg) / f"{song_id}.csv"
            if sub.is_file():
                z.write(sub, f"subtitles/{song_id}.csv")
    print(f"[colab] {len(labeled)} songs -> {out} "
          f"({out.stat().st_size/1e6:.0f} MB). Upload to Drive as MyDrive/kashi/colab_pack.zip, "
          f"then run colab/ctc_bootstrap.ipynb")
    return out


def integrate(cfg, src: str | Path) -> Path:
    src = Path(src)
    dest = cfg.artifacts_dir / "colab"
    dest.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            z.extractall(dest)
    else:
        for item in src.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    have = sorted(p.name for p in dest.iterdir())
    print(f"[colab] integrated into {dest}: {have}")
    print("next: `kashi train frame --targets artifacts/colab/fa_labels` (FA-bootstrapped"
          " frame targets) and/or point decoder emissions at the CTC model")
    return dest
