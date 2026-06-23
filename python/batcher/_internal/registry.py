"""A single generic registry pattern, used for every extension point.

Sources, sinks, operators, optimization rules, and backends all register through
an instance of `Registry[T]`. Third-party packages can also contribute via
`importlib.metadata` entry points (wired in once the extension points stabilize),
so plugging in a new source or backend never requires forking the engine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from batcher._internal.errors import BatcherError

T = TypeVar("T")

__all__ = ["Registry"]


class Registry(Generic[T]):
    """A name → factory mapping with decorator registration.

    Used as a module-level singleton per extension point, e.g.
    ``SOURCES = Registry[Source]("source")``.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that registers `obj` under `name` and returns it unchanged."""

        def _decorator(obj: T) -> T:
            if name in self._items:
                raise BatcherError(f"{self._kind} {name!r} is already registered")
            self._items[name] = obj
            return obj

        return _decorator

    def add(self, name: str, obj: T) -> None:
        """Imperative registration (for non-decorator call sites)."""
        if name in self._items:
            raise BatcherError(f"{self._kind} {name!r} is already registered")
        self._items[name] = obj

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise BatcherError(f"unknown {self._kind} {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))
