"""Component registry: the model-swapping mechanism.

Each pipeline slot (separator/encoder/segmenter/classifier/decoder/boundary)
maps names to factories. Implementations self-register with @register;
create_from_config builds the component chosen in [pipeline] with its
[<kind>.<name>] section. Adding a new model = one class + one decorator,
then select it in config or with --set pipeline.<kind>=<name>.
"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {}


def register(kind: str, name: str):
    def deco(factory: Callable[..., Any]):
        _REGISTRY.setdefault(kind, {})[name] = factory
        return factory

    return deco


def names(kind: str) -> list[str]:
    from . import components  # noqa: F401  (importing registers built-ins)

    return sorted(_REGISTRY.get(kind, {}))


def create(cfg, kind: str, name: str | None = None) -> Any:
    """Instantiate component `name` of `kind` with its [<kind>.<name>] config section."""
    # Importing the package registers all built-ins.
    from . import components  # noqa: F401

    if name is None:
        name = cfg[f"pipeline.{kind}"]
    reg = _REGISTRY.get(kind, {})
    if name not in reg:
        raise KeyError(f"No {kind} named {name!r}. Available: {names(kind)}")
    return reg[name](cfg)


def create_from_config(cfg, kind: str) -> Any:
    return create(cfg, kind)
