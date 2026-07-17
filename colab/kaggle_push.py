"""Push the canonical colab/ctc_bootstrap.ipynb to Kaggle with overrides.

usage:
  .venv/bin/python colab/kaggle_push.py '<json overrides>' [--no-watch] [--interval 90]

Example:
  .venv/bin/python colab/kaggle_push.py '{"EPOCHS": 1, "REUSE_DATA": false,
      "REUSE_MODEL": false, "REUSE_FA": false}'

The overrides are applied TEXTUALLY to the run-flags cell (cell 1) and the
`---- config ----` cell — the same knobs run_nb.py overrides locally — by
rewriting `KEY = <value>` lines in place (comment preserved). The rewritten
notebook + kernel-metadata.json land in artifacts/kaggle_push/, then
`kaggle kernels push` uploads AND STARTS a new version of KERNEL_ID.
Watching is delegated to colab/kaggle_watch.py (verbose polling).

The kernel keeps the id of the user's uploaded notebook so account-side
attachments (HF_TOKEN secret) stay associated; the run itself loads the
secret with a graceful fallback, so a lost attachment degrades to anonymous
HF downloads rather than failing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "colab" / "ctc_bootstrap.ipynb"
STAGE = ROOT / "artifacts" / "kaggle_push"
KERNEL_ID = "alexzgu/ctc-bootstrap"
DATASET = "alexzgu/kashi-pack-152"

METADATA = {
    "id": KERNEL_ID,
    "title": "ctc-bootstrap",
    "code_file": "ctc-bootstrap.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [DATASET, "alexzgu/kashi-ckpt"],  # pack + warm-start checkpoint (flat: root is the model dir)
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}


def apply_overrides(nb: dict, overrides: dict) -> list[str]:
    """Rewrite `KEY = value` lines in the flags/config cells; return applied keys."""
    applied = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "==== run flags" not in src and "---- config ----" not in src:
            continue
        for key, val in overrides.items():
            pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#\n]*?)(\s*)(#.*)?$",
                             re.MULTILINE)
            if pat.search(src):
                rep = repr(val)  # NOT json.dumps: booleans must stay Python (True/False)
                src = pat.sub(lambda m: f"{m.group(1)}{rep}{m.group(3)}{m.group(4) or ''}",
                              src, count=1)
                applied.append(key)
        cell["source"] = src.splitlines(keepends=True)
    return applied


def normalize_nb(nb: dict) -> None:
    """Kaggle executes with papermill, which hard-fails without a kernelspec
    (the web editor adds one on save; raw API pushes must bring their own).
    Cell ids are required from nbformat 4.5 and warned about below."""
    nb.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"].setdefault("language_info", {"name": "python"})
    if nb.get("nbformat", 4) >= 4 and nb.get("nbformat_minor", 0) < 5:
        nb["nbformat_minor"] = 5
    for i, cell in enumerate(nb["cells"]):
        cell.setdefault("id", f"cell-{i}")


def main() -> int:
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else {}
    nb = json.loads(CANONICAL.read_text())
    normalize_nb(nb)
    applied = apply_overrides(nb, overrides)
    missing = set(overrides) - set(applied)
    if missing:
        sys.exit(f"overrides not found in flags/config cells: {sorted(missing)}")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    (STAGE / "ctc-bootstrap.ipynb").write_text(json.dumps(nb, indent=1))
    (STAGE / "kernel-metadata.json").write_text(json.dumps(METADATA, indent=2))
    print(f"[push] staged {STAGE} (overrides applied: {applied or 'none'})")

    r = subprocess.run([str(ROOT / ".venv" / "bin" / "kaggle"),
                        "kernels", "push", "-p", str(STAGE)])
    if r.returncode != 0:
        return r.returncode
    if "--no-watch" in sys.argv:
        print(f"[push] pushed {KERNEL_ID}; watch later with: "
              f".venv/bin/python colab/kaggle_watch.py watch {KERNEL_ID}")
        return 0
    interval = "90"
    if "--interval" in sys.argv:
        interval = sys.argv[sys.argv.index("--interval") + 1]
    return subprocess.run([str(ROOT / ".venv" / "bin" / "python"),
                           str(ROOT / "colab" / "kaggle_watch.py"),
                           "watch", KERNEL_ID, "--interval", interval]).returncode


if __name__ == "__main__":
    sys.exit(main())
