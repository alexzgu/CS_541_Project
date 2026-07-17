"""Kaggle kernel harness: push a kernel and babysit the run with verbose polling.

usage:
  .venv/bin/python runs/kaggle_watch.py push  runs/kaggle/ctc_bootstrap [--interval 90] [--no-watch]
  .venv/bin/python runs/kaggle_watch.py watch alexzgu/ctc-bootstrap    [--interval 90] [--max-hours 12]
  .venv/bin/python runs/kaggle_watch.py fetch alexzgu/ctc-bootstrap    [--out artifacts/kaggle_out]

push  = `kaggle kernels push -p <dir>` (kernel-metadata.json inside), then watch.
watch = poll `kernels status` every --interval s; every poll is logged with a
        timestamp; state changes are called out; on COMPLETE/ERROR the last
        LOG_TAIL lines of the run log are printed and outputs are fetched.
fetch = pull outputs + full log right now into --out.

Exit codes: 0 COMPLETE, 1 ERROR/CANCELLED, 2 watch window exceeded.
All kaggle CLI stderr is passed through — nothing is swallowed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

KAGGLE = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "kaggle")
LOG_TAIL = 80

TERMINAL_OK = {"COMPLETE"}
TERMINAL_BAD = {"ERROR", "CANCELACKNOWLEDGED", "CANCELREQUESTED"}


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def sh(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([KAGGLE, *args], text=True,
                          capture_output=capture)


def slug_of(kernel_dir: Path) -> str:
    import json
    return json.loads((kernel_dir / "kernel-metadata.json").read_text())["id"]


def status_of(slug: str) -> str:
    r = sh("kernels", "status", slug)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r'"?([A-Za-z]+)"?\s*$', out.strip())
    tok = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    if tok:
        return tok.group(1)
    if m:
        return m.group(1).upper()
    return f"UNPARSED({out.strip()[:120]})"


def show_logs(slug: str) -> None:
    r = sh("kernels", "logs", slug)
    text = (r.stdout or "") + (r.stderr or "")
    lines = text.splitlines()
    print(f"[{ts()}] ---- run log (last {LOG_TAIL} of {len(lines)} lines) ----")
    for l in lines[-LOG_TAIL:]:
        print("   ", l)
    print(f"[{ts()}] ---- end log ----")


def fetch_outputs(slug: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] fetching outputs -> {out_dir}")
    r = sh("kernels", "output", slug, "-p", str(out_dir), capture=False)
    if r.returncode != 0:
        print(f"[{ts()}] WARNING: kernels output exited {r.returncode}")


def watch(slug: str, interval: int, max_hours: float, out_dir: Path) -> int:
    deadline = time.time() + max_hours * 3600
    last = None
    print(f"[{ts()}] watching {slug} (poll every {interval}s, max {max_hours}h)")
    while time.time() < deadline:
        st = status_of(slug)
        marker = "  <-- STATE CHANGE" if st != last else ""
        print(f"[{ts()}] status: {st}{marker}", flush=True)
        last = st
        if st in TERMINAL_OK:
            show_logs(slug)
            fetch_outputs(slug, out_dir)
            print(f"[{ts()}] COMPLETE")
            return 0
        if st in TERMINAL_BAD:
            show_logs(slug)
            print(f"[{ts()}] run ended badly: {st}")
            return 1
        time.sleep(interval)
    print(f"[{ts()}] watch window exceeded ({max_hours}h) — kernel may still be running")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "watch", "fetch"])
    ap.add_argument("target", help="kernel dir (push) or user/slug (watch, fetch)")
    ap.add_argument("--interval", type=int, default=90)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--out", default="artifacts/kaggle_out")
    ap.add_argument("--no-watch", action="store_true")
    a = ap.parse_args()

    if a.cmd == "push":
        kdir = Path(a.target)
        slug = slug_of(kdir)
        print(f"[{ts()}] pushing {kdir} -> {slug}")
        r = sh("kernels", "push", "-p", str(kdir), capture=False)
        if r.returncode != 0:
            print(f"[{ts()}] push FAILED ({r.returncode})")
            return 1
        if a.no_watch:
            return 0
        time.sleep(20)  # give the queue a moment before first poll
        return watch(slug, a.interval, a.max_hours, Path(a.out))
    if a.cmd == "watch":
        return watch(a.target, a.interval, a.max_hours, Path(a.out))
    fetch_outputs(a.target, Path(a.out))
    show_logs(a.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
