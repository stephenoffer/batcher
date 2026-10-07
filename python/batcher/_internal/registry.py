"""Keyed lookup tables: the generic extension-point registry and the identity memo.

Sources, sinks, operators, optimization rules, and backends all register through
an instance of `Registry[T]`. Registration happens when the registering module is
imported, so a third-party source or sink plugs in by subclassing the public
`batcher.io` bases and being imported before it is named; nothing is forked. There is
no automatic discovery: nothing reads `importlib.metadata` entry points, and a package
that is never imported never registers. `Registry(on_miss=...)` is the seam such
discovery would hang from.

Because every extension point funnels through here, this is also where a user's typo
in a format or backend name is caught — so `get` raises the canonical unknown-name
error (`_internal.errors.unknown_value`), with a suggestion and the registered names,
rather than a bare "not found".
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Hashable, Iterator
from typing import Generic, Literal, TypeVar

from batcher._internal.errors import BatcherError, unknown_value

T = TypeVar("T")
V = TypeVar("V")

__all__ = ["MISSING", "IdentityMemo", "KeyedMemo", "Registry"]


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
            # Not an error yet, and deliberately not traced: a miss here is ordinary
            # control flow on the way to the retry below, not a suppressed failure.
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

        This deliberately does **not** call `complete`, unlike every other method here
        that reports a whole-registry view. A repr is a debugging aid, and it is called
        by tooling that never asked for the deferred import: Sphinx's autodoc reprs
        every module attribute it documents. Completing here turns printing a registry
        into importing every lakehouse, warehouse, NoSQL, SQL and streaming connector,
        and turns any failure inside one of those imports into a failure of the print.
        That is not hypothetical — it is how `just docs` came to die on a broken stdlib
        `sqlite3`: autodoc reprd `SOURCES`, the repr imported the streaming family, and
        the traceback named Sphinx rather than the interpreter. So the repr reports what
        is registered *now*, and says when that is not yet the whole list.
        """
        pending = "" if self._completed else " so far, deferred families not loaded"
        return (
            f"Registry({self._kind!r}, {len(self._items)} registered{pending}: "
            f"{sorted(self._items)})"
        )


class _Missing(enum.Enum):
    MISSING = enum.auto()


#: What `IdentityMemo.get` returns on a miss, so a memoized `None` stays a hit.
MISSING = _Missing.MISSING


class IdentityMemo(Generic[V]):
    """A bounded memo keyed on an object's identity, for immutable objects too costly to hash.

    Plans, schemas and rule lists are immutable, so an answer computed from one never goes
    stale, but hashing them by value costs more than the answer. Each entry holds a reference
    to its key object: without it a freed object's address is reused and a stale answer would
    be served for an unrelated one. Full, the memo clears wholesale; a dropped entry costs one
    recomputation, never a wrong answer.

    Args:
        maxsize: Entries held before the memo clears.
    """

    __slots__ = ("_entries", "_maxsize")

    def __init__(self, maxsize: int) -> None:
        self._entries: dict[tuple[Hashable, ...], tuple[object, V]] = {}
        self._maxsize = maxsize

    def get(self, obj: object, *extra: Hashable) -> V | Literal[_Missing.MISSING]:
        """The answer memoized for `obj` (and `extra`), or `MISSING`.

        Args:
            obj: The object the answer was computed from.
            *extra: Further hashable inputs the answer depends on.

        Returns:
            The memoized answer, or `MISSING` when there is none.
        """
        hit = self._entries.get((id(obj), *extra))
        if hit is not None and hit[0] is obj:
            return hit[1]
        return MISSING

    def put(self, obj: object, value: V, *extra: Hashable) -> V:
        """Memoize `value` for `obj` (and `extra`).

        Args:
            obj: The object the answer was computed from.
            value: The answer.
            *extra: Further hashable inputs the answer depends on.

        Returns:
            `value`, so a call site can `return memo.put(...)`.
        """
        if len(self._entries) >= self._maxsize:
            self._entries.clear()
        self._entries[(id(obj), *extra)] = (obj, value)
        return value

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()


class KeyedMemo(Generic[V]):
    """A bounded memo keyed on a hashable *value*, for answers that are a pure function of it.

    The companion to `IdentityMemo` for a key that names content rather than an object, such
    as a plan's `content_key`: two plans built separately from the same query share it, so an
    answer computed for the first serves the second, which an identity memo never can. Full,
    the memo clears wholesale; a dropped entry costs one recomputation, never a wrong answer.

    Args:
        maxsize: Entries held before the memo clears.
    """

    __slots__ = ("_entries", "_maxsize")

    def __init__(self, maxsize: int) -> None:
        self._entries: dict[Hashable, V] = {}
        self._maxsize = maxsize

    def get(self, key: Hashable) -> V | Literal[_Missing.MISSING]:
        """The answer memoized under `key`, or `MISSING`.

        Args:
            key: The value the answer is a function of.

        Returns:
            The memoized answer, or `MISSING` when there is none.
        """
        return self._entries.get(key, MISSING)

    def put(self, key: Hashable, value: V) -> V:
        """Memoize `value` under `key`.

        Args:
            key: The value the answer is a function of.
            value: The answer.

        Returns:
            `value`, so a call site can `return memo.put(...)`.
        """
        if len(self._entries) >= self._maxsize:
            self._entries.clear()
        self._entries[key] = value
        return value

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()
