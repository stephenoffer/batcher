"""The `ds.meta.against(other)` accessor — what two relations' footers say about their join.

The shortcut that saves the most work in absolute terms. If one side's key range is ``[1, 10]``
and the other's is ``[900, 999]``, the inner join is empty — provably, from four numbers, with
neither side read. No build, no probe, no shuffle. A partitioned fact table joined against a
dimension filtered to a range it does not cover hits this case every time, and today it pays
for a full shuffle to discover what the footers already knew.

Only *emptiness* is proved. Overlapping ranges do not imply a match exists (two columns can
interleave without sharing a value), so an overlap falls back to running the join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.api.dataset.meta._facts import MetaBase, answer
from batcher.kyber.shortcuts import joins

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["PairMeta"]


class PairMeta(MetaBase):
    """Two-relation shortcuts, reached as ``ds.meta.against(other)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> left = bt.from_pydict({"k": [1, 2, 3]})
            >>> right = bt.from_pydict({"k": [900, 901]})
            >>> left.meta.against(right).join_is_empty("k")
            True
    """

    __slots__ = ("_other",)

    def __init__(self, ds: Dataset, other: Dataset) -> None:
        """Bind to a pair of datasets; prefer ``ds.meta.against(other)``."""
        super().__init__(ds)
        self._other = other

    def join_is_empty(self, on: str, right_on: str | None = None) -> bool:
        """Whether an inner equi-join on these keys yields no row at all.

        Proved for free when the two key ranges are disjoint — the case that lets a whole
        shuffle be skipped. Otherwise the join runs and its emptiness is observed.

        Args:
            on: The join key on this side.
            right_on: The join key on the other side; defaults to `on`.

        Returns:
            ``True`` if the join produces no rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"k": [1, 2]})
                >>> b = bt.from_pydict({"k": [50, 60]})
                >>> a.meta.against(b).join_is_empty("k")
                True
        """
        left_key, right_key = self._keys(on, right_on)
        decided = self._ask_pair(joins.join_is_empty, left_key, right_key)
        return answer(decided, lambda: self._join(left_key, right_key).is_empty())

    def overlaps(self, on: str, right_on: str | None = None) -> bool:
        """Whether the two relations share any key value — the complement of `join_is_empty`.

        Args:
            on: The join key on this side.
            right_on: The join key on the other side; defaults to `on`.

        Returns:
            ``True`` if at least one key value appears on both sides.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"k": [1, 2]})
                >>> b = bt.from_pydict({"k": [2, 3]})
                >>> a.meta.against(b).overlaps("k")
                True
        """
        return not self.join_is_empty(on, right_on)

    def key_overlap(self, on: str, right_on: str | None = None) -> tuple[Any, Any] | None:
        """The range of key values the two sides could share, as ``(low, high)``.

        The intersection of the two key bounds — the only window a matching value can lie in,
        and therefore the range worth partitioning or pruning both sides to. ``None`` when
        either side's bounds are not known exactly.

        Args:
            on: The join key on this side.
            right_on: The join key on the other side; defaults to `on`.

        Returns:
            The shared key range, or ``None`` if the bounds are not provable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"k": [1, 5]})
                >>> b = bt.from_pydict({"k": [3, 9]})
                >>> a.meta.against(b).key_overlap("k")
                (3, 5)
        """
        left_key, right_key = self._keys(on, right_on)
        return self._ask_pair(joins.key_overlap, left_key, right_key)

    def estimated_rows(self, on: str, right_on: str | None = None) -> float:
        """The estimated size of the join result — the number the optimizer orders joins by.

        Explicitly approximate, and free: it reads sketched distinct counts, never the data.
        Use it to decide whether a join is worth attempting at all, not to report a count.

        Args:
            on: The join key on this side.
            right_on: The join key on the other side; defaults to `on`.

        Returns:
            The estimated number of result rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"k": [1, 2]})
                >>> b = bt.from_pydict({"k": [1, 2]})
                >>> a.meta.against(b).estimated_rows("k") >= 0
                True
        """
        left_key, right_key = self._keys(on, right_on)
        estimate = self._ask_pair(joins.estimated_join_rows, left_key, right_key)
        return 0.0 if estimate is None else estimate

    def _keys(self, on: str, right_on: str | None) -> tuple[str, str]:
        """Validate the join keys on both sides, defaulting the right key to the left one."""
        left_key = self.require_column(on)
        right_key = MetaBase(self._other).require_column(right_on or on)
        return left_key, right_key

    def _ask_pair(self, shortcut: Any, left_key: str, right_key: str) -> Any:
        """Put a two-relation question to Kyber, or None when either side has no facts."""
        left = self.facts()
        right = MetaBase(self._other).facts()
        if left is None or right is None:
            return None
        return shortcut(left, right, left_key, right_key)

    def _join(self, left_key: str, right_key: str) -> Dataset:
        """The inner join whose emptiness the metadata path is trying to predict."""
        if left_key == right_key:
            return self._ds.join(self._other, on=left_key, how="inner")
        return self._ds.join(self._other, left_on=left_key, right_on=right_key, how="inner")
