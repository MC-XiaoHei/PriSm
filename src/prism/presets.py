from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, TypeVar, cast

T = TypeVar("T")

SECTIONS = ("run", "data", "model")

EMPTY: Mapping[str, Any] = {}
EMPTY_SECTIONS: Mapping[str, Mapping[str, Any]] = {}


def load(path: str | Path) -> Mapping[str, Mapping[str, Any]]:
    with Path(path).open("rb") as handle:
        preset = tomllib.load(handle)
    unknown = set(preset) - set(SECTIONS)
    if unknown:
        raise KeyError(f"unknown preset section {sorted(unknown)}")
    return preset


def _coerce(default: Any, value: Any) -> Any:
    if default is None or isinstance(value, type(default)):
        return value
    return type(default)(value)


def apply(cls: type[T], base: T, overrides: Mapping[str, Any]) -> T:
    known = {item.name for item in fields(cast(Any, cls))}
    unknown = set(overrides) - known
    if unknown:
        raise KeyError(f"{cls.__name__} has no field {sorted(unknown)}")
    values = {key: _coerce(getattr(base, key), value) for key, value in overrides.items()}
    return cast(T, replace(cast(Any, base), **values))
