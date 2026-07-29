"""A single generic registry pattern, used for every extension point.

Sources, sinks, operators, optimization rules, and backends all register through
an instance of `Registry[T]`. Third-party packages can also contribute via
`importlib.metadata` entry points (wired in once the extension points stabilize),
so plugging in a new source or backend never requires forking the engine.

Because every extension point funnels through here, this is also where a user's typo
in a format or backend name is caught — so `get` raises the canonical unknown-name
error (`_internal.errors.unknown_value`), with a suggestion and the registered names,
rather than a bare "not found".
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from batcher._internal.errors import BatcherError, unknown_value

T = TypeVar("T")

__all__ = ["Registry"]


class Registry(Generic[T]):
    """A name → factory mapping with decorator registration.

    Used as a module-level singleton per extension point, e.g.
    ``SOURCES = Registry[Source]("source")``.

    Entries normally register at import, as a side effect of the defining module being
    imported. A registry whose family is large and rarely-used can instead pass `on_miss`
    and register on first demand — see `complete`.
    """

    def __init__(
        self, kind: str, *, doc: str = "", on_miss: Callable[[], None] | None = None
    ) -> None:
        """Create an empty registry.

        Args:
            kind: What is registered, singular and lowercase (``"source"``). It is the
                noun every error from this registry is phrased around.
            doc: An optional documentation path, attached to unknown-name errors so a
                user who mistyped a name is pointed at the list of real ones.
            on_miss: An optional one-shot hook run before a lookup is declared a miss and
                before any call that promises a *complete* view of the registry. It exists
                so a family of entries can register itself the first time anyone asks for
                one, instead of at import. See `complete`.
        """
        self._kind = kind
        self._doc = doc
        self._items: dict[str, T] = {}
        self._on_miss = on_miss
        self._completed = on_miss is None

    def complete(self) -> None:
        """Run the deferred-registration hook, at most once.

        The extension points here are populated by importing the modules that register
        into them, and for a large family — every database, warehouse, and message broker
        Batcher can read — that import is most of what ``import batcher`` costs, paid by
        every process whether or not it ever names one of those formats. `on_miss` lets
        the family register on first demand instead; this is the "now I actually need
        them" trigger, called from every lookup that could otherwise answer from an
        incomplete registry.

        Idempotent, and self-disarming *before* the hook runs, so a hook that registers
        through this same registry cannot recurse.
        """
        hook = self._on_miss
        if self._completed or hook is None:
            return
        self._completed = True
        hook()

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that registers `obj` under `name` and returns it unchanged."""

        def _decorator(obj: T) -> T:
            self.add(name, obj)
            return obj

        return _decorator

    def add(self, name: str, obj: T) -> None:
        """Imperative registration (for non-decorator call sites).

        Args:
            name: The lookup name. Must be a non-empty string — a non-string name is
                unreachable through `get`, which takes the name a user typed, so
                accepting one registers an entry nothing can ever find.
            obj: The registered value.

        Raises:
            BatcherError: If `name` is not a non-empty string, or is already taken.
        """
        if not isinstance(name, str) or not name:
            raise BatcherError(
                f"Cannot register a {self._kind} under {name!r}.",
                hint="A registry name must be a non-empty string.",
            )
        if name in self._items:
            raise BatcherError(
                f"A {self._kind} named {name!r} is already registered.",
                hint=(
                    "Registration names are unique. Pick a different name, or check "
                    "whether the defining module is being imported twice."
                ),
            )
        self._items[name] = obj

    def get(self, name: str) -> T:
        """The registered value for `name`.

        Args:
            name: The lookup name, as the user spelled it.

        Returns:
            The registered value.

        Raises:
            BatcherError: If nothing is registered under `name`. The error names the
                closest registered match and lists what is registered.
        """
        try:
            return self._items[name]
        except (KeyError, TypeError):
            pass
        # A miss may only mean the family that owns this name has not registered yet.
        self.complete()
        try:
            return self._items[name]
        except (KeyError, TypeError):
            raise unknown_value(
                BatcherError,
                self._kind,
                name,
                self._items,
                hint=(
                    f"No {self._kind} is registered yet — the module that registers "
                    "it may not have been imported."
                    if not self._items
                    else ""
                ),
                doc=self._doc,
            ) from None

    def names(self) -> list[str]:
        """The registered names, sorted."""
        self.complete()
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        if name in self._items:
            return True
        self.complete()
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        self.complete()
        return iter(sorted(self._items))

    def __len__(self) -> int:
        self.complete()
        return len(self._items)

    def __repr__(self) -> str:
        """Name the extension point and what is registered in it.

        The default `object.__repr__` — an address — is useless at exactly the moment
        a registry is printed, which is while working out why a lookup missed.
        """
        return f"Registry({self._kind!r}, {len(self._items)} registered: {self.names()})"
