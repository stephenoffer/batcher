"""An identity-keyed memo that pins its keys, optionally bounded by clear-on-full.

The optimizer memoizes many pure functions of an immutable object — a plan node, an Arrow
schema, a rule list, a `SourceStatistics` — and the key that repeats is the object's
*identity*, not its value. `id()` alone is unsound as a key: CPython reuses a freed object's
address immediately, so a memo that does not hold its key alive can answer for an unrelated
object that landed at the same address. Each entry therefore stores the key object beside the
value, which both pins the id for the entry's lifetime and lets a lookup confirm `stored is
key` before trusting it.

A bounded memo clears wholesale when it fills rather than evicting one entry. A dropped entry
costs one recomputation, never a wrong answer, and wholesale clearing keeps the retention the
pinning causes from growing without limit.

Layer 0: `kyber`, `carbonite`, `core` and `governance` may not import one another, so a
shape each of them may need is written once here rather than by hand at every site.
"""

from __future__ import annotations

import enum
from typing import Final, Generic, TypeVar

__all__ = ["MISSING", "IdentityMemo", "Missing"]

K = TypeVar("K")
V = TypeVar("V")


class Missing(enum.Enum):
    """The type of `MISSING`, a lookup miss that stays distinct from a memoized `None`."""

    MISSING = enum.auto()


#: Returned by `IdentityMemo.get` on a miss. An enum member so `hit is not MISSING` narrows
#: the result to the value type for a type checker.
MISSING: Final = Missing.MISSING


class IdentityMemo(Generic[K, V]):
    """A memo keyed on object identity whose entries pin their keys against `id()` reuse.

    Deliberately minimal and lock-free: it sits on optimizer hot paths, where a hit costs one
    dict probe and one identity comparison and allocates nothing. Like the hand-written dicts
    it replaces it is not synchronized, which is sound for memoizing a pure function — a race
    at worst recomputes an entry.

    Examples:
        .. doctest::

            >>> from batcher._internal.memo import MISSING, IdentityMemo
            >>> memo: IdentityMemo[list[int], int] = IdentityMemo(max_entries=2)
            >>> key = [1, 2, 3]
            >>> memo.get(key) is MISSING
            True
            >>> memo.put(key, 6)
            >>> memo.get(key)
            6
            >>> memo.get([1, 2, 3]) is MISSING  # equal but not identical
            True

    Args:
        max_entries: Clear every entry before storing a new one once this many are held.
            `None` (the default) never clears, for a memo whose owner bounds its lifetime,
            such as one optimize run.
    """

    __slots__ = ("_entries", "_max_entries")

    def __init__(self, max_entries: int | None = None) -> None:
        self._entries: dict[int, tuple[K, V]] = {}
        self._max_entries = max_entries

    def get(self, key: K) -> V | Missing:
        """The value memoized for this exact object, or `MISSING`.

        Args:
            key: The object whose identity is looked up.

        Returns:
            The stored value when `key` itself was stored, otherwise `MISSING` — including
            when a different, since-freed object had the same `id()`.
        """
        hit = self._entries.get(id(key))
        if hit is not None and hit[0] is key:
            return hit[1]
        return MISSING

    def put(self, key: K, value: V) -> None:
        """Memoize `value` for `key`, clearing the memo first if it is full.

        Args:
            key: The object to key on. It is held alive until the entry is dropped.
            value: The value to return for `key`.
        """
        entries = self._entries
        if self._max_entries is not None and len(entries) >= self._max_entries:
            entries.clear()
        entries[id(key)] = (key, value)

    def __len__(self) -> int:
        """The number of entries held."""
        return len(self._entries)
