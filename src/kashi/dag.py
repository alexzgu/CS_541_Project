"""Minimal stage DAG with content-addressed skip (ROADMAP §3.9).

Each stage declares input paths, output paths, and the config subtree it
reads. fingerprint = sha256(input file stats + config subtree + code_version).
`kashi run <stage>` executes only stale stages in dependency order; training
stages are barrier-marked and skipped unless --allow-train.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CODE_VERSION = "kashi-0.2.0"


@dataclass
class Stage:
    name: str
    fn: Callable[..., object]
    inputs: Callable[[object], list[Path]]
    outputs: Callable[[object], list[Path]]
    config_keys: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    is_training: bool = False


_STAGES: dict[str, Stage] = {}


def stage(
    name: str,
    *,
    inputs,
    outputs,
    config_keys: list[str] | None = None,
    deps: list[str] | None = None,
    is_training: bool = False,
):
    def deco(fn):
        _STAGES[name] = Stage(
            name=name,
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            config_keys=config_keys or [],
            deps=deps or [],
            is_training=is_training,
        )
        return fn

    return deco


def stages() -> dict[str, Stage]:
    return dict(_STAGES)


def _stat_sig(paths: list[Path]) -> list[str]:
    sig = []
    for p in sorted(set(map(Path, paths))):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    st = f.stat()
                    sig.append(f"{f}|{st.st_size}|{st.st_mtime_ns}")
        elif p.is_file():
            st = p.stat()
            sig.append(f"{p}|{st.st_size}|{st.st_mtime_ns}")
        else:
            sig.append(f"{p}|missing")
    return sig


def fingerprint(st: Stage, cfg) -> str:
    payload = {
        "code": CODE_VERSION,
        "inputs": _stat_sig(st.inputs(cfg)),
        "config": {k: cfg.get(k) for k in st.config_keys},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _state_file(cfg) -> Path:
    return cfg.artifacts_dir / "dag_state.json"


def _load_state(cfg) -> dict:
    f = _state_file(cfg)
    if f.is_file():
        return json.loads(f.read_text())
    return {}


def _save_state(cfg, state: dict) -> None:
    f = _state_file(cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state, indent=1))


def _order(target: str) -> list[str]:
    seen: list[str] = []

    def visit(name: str, chain: tuple = ()):
        if name in chain:
            raise ValueError(f"stage cycle: {' -> '.join(chain + (name,))}")
        if name in seen:
            return
        if name not in _STAGES:
            raise KeyError(f"unknown stage {name!r}; available: {sorted(_STAGES)}")
        for d in _STAGES[name].deps:
            visit(d, chain + (name,))
        seen.append(name)

    visit(target)
    return seen


def run(cfg, target: str, force: bool = False, allow_train: bool = False) -> list[str]:
    """Execute stale stages up to `target`. Returns the list of stages that ran."""
    state = _load_state(cfg)
    ran: list[str] = []
    for name in _order(target):
        st = _STAGES[name]
        fp = fingerprint(st, cfg)
        outputs_exist = all(p.exists() for p in st.outputs(cfg))
        if not force and outputs_exist and state.get(name) == fp:
            continue
        if st.is_training and not allow_train:
            print(f"[dag] skipping training stage {name!r} (pass --allow-train to run)")
            continue
        print(f"[dag] running {name}")
        st.fn(cfg)
        state[name] = fingerprint(st, cfg)
        ran.append(name)
    _save_state(cfg, state)
    return ran
