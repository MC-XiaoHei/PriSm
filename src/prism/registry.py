from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Entry(Generic[T]):
    id: str
    obj: T


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Entry[T]] = {}
        self._by_obj: dict[T, str] = {}

    def register(self, ident: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if ident in self._entries:
                raise KeyError(f"{self.kind} already registered: {ident}")
            self._entries[ident] = Entry(ident, obj)
            self._by_obj[obj] = ident
            return obj

        return deco

    def get(self, ident: str) -> T:
        entry = self._entries.get(ident)
        if entry is None:
            known = ", ".join(self.ids()) or "none"
            raise KeyError(f"unknown {self.kind} {ident}; available: {known}")
        return entry.obj

    def id_of(self, obj: T) -> str:
        try:
            return self._by_obj[obj]
        except KeyError:
            raise KeyError(f"{obj!r} is not registered as a {self.kind}") from None

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def __contains__(self, ident: str) -> bool:
        return ident in self._entries

    def __iter__(self) -> Iterator[Entry[T]]:
        for ident in self.ids():
            yield self._entries[ident]
