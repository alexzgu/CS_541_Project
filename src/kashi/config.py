"""Layered TOML configuration.

Precedence (low to high): configs/default.toml < user file (--config) < --set
overrides. A Config is a thin dict wrapper with dotted-path access:
cfg["decoder.segmental.d_max"].
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(raw: str) -> Any:
    """Parse a --set value: try bool/int/float/list, fall back to string."""
    s = raw.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if "," in s:
        return [_parse_value(p) for p in s.split(",") if p.strip()]
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


class Config:
    def __init__(self, data: dict, root: Path = REPO_ROOT):
        self._data = data
        self.root = root

    @classmethod
    def load(
        cls,
        config_file: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> "Config":
        with open(DEFAULT_CONFIG, "rb") as f:
            data = tomllib.load(f)
        if config_file:
            with open(config_file, "rb") as f:
                data = _deep_merge(data, tomllib.load(f))
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"--set expects key=value, got {item!r}")
            key, _, raw = item.partition("=")
            node = data
            parts = key.strip().split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = _parse_value(raw)
        return cls(data)

    def __getitem__(self, dotted: str) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            node = node[part]
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return self[dotted]
        except (KeyError, TypeError):
            return default

    def section(self, dotted: str) -> dict:
        val = self.get(dotted, {})
        return dict(val) if isinstance(val, dict) else {}

    def path(self, key: str) -> Path:
        """Resolve a [paths] entry (absolute kept, relative anchored at repo root)."""
        p = Path(self[f"paths.{key}"])
        return p if p.is_absolute() else self.root / p

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def artifacts_dir(self) -> Path:
        return self.path("artifacts_dir")

    @property
    def runs_dir(self) -> Path:
        return self.path("runs_dir")

    @property
    def frame_ms(self) -> int:
        return int(self["data.frame_ms"])

    @property
    def sample_rate(self) -> int:
        return int(self["data.sample_rate"])

    def as_dict(self) -> dict:
        return self._data

    def dump_toml(self) -> str:
        """Serialise (for run-dir snapshots). Minimal TOML writer."""
        lines: list[str] = []

        def fmt(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, str):
                return f'"{v}"'
            if isinstance(v, list):
                return "[" + ", ".join(fmt(x) for x in v) + "]"
            return repr(v)

        def walk(d: dict, prefix: str) -> None:
            scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
            subs = {k: v for k, v in d.items() if isinstance(v, dict)}
            if scalars and prefix:
                lines.append(f"[{prefix}]")
            for k, v in scalars.items():
                lines.append(f"{k} = {fmt(v)}")
            for k, v in subs.items():
                walk(v, f"{prefix}.{k}" if prefix else k)

        walk(self._data, "")
        return "\n".join(lines) + "\n"
