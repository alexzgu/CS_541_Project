"""Run ctc_bootstrap.ipynb locally: `python colab/run_nb.py <KASHI_DRIVE> '<json overrides>'`.

Execs each code cell with `!`/`%` lines stripped; applies the JSON overrides
after the run-flags cell AND the `---- config ----` cell (so CHECKPOINT and
the PSEUDO_*/EPOCHS/REUSE_* knobs can all be overridden). Example:

    python colab/run_nb.py artifacts/ctc_myrun \
      '{"REUSE_MODEL": false, "EPOCHS": 4,
        "CHECKPOINT": "/abs/path/to/champion/ctc_model"}'

Per-run KASHI_DRIVE dirs conventionally hold `data` and `out/fa_*` symlinks
into artifacts/ctc_local (see ROADMAP P3/P5 status notes).
"""

import json
import os
import sys

os.environ["KASHI_DRIVE"] = sys.argv[1]
overrides = json.loads(sys.argv[2])
nb = json.load(open("colab/ctc_bootstrap.ipynb"))
g = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(l for l in c["source"] if not l.lstrip().startswith(("!", "%")))
    if not src.strip():
        continue
    print(f"===== cell {i} =====", flush=True)
    exec(compile(src, f"cell{i}", "exec"), g)
    if "==== run flags" in "".join(c["source"]) or "---- config ----" in "".join(c["source"]):
        g.update(overrides)
        print("overrides applied:", overrides, flush=True)
print("NOTEBOOK RUN COMPLETE", flush=True)
