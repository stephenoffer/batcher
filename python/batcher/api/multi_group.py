"""Multi-level grouped aggregation — `ROLLUP`, `CUBE` and `GROUPING SETS`.

The SQL front-end has had these since it could parse them; the DataFrame surface had
no spelling for them at all, so a subtotal report had to be written as a `union` of
hand-written `group_by`s (or in SQL). This module is that spelling:
`ds.rollup("region", "city").agg(...)`.

A multi-level GROUP BY is **not** a distinct execution strategy here — the same choice
the SQL translator makes, for the same reason. Each level is an ordinary `group_by`
over its active keys, with the inactive keys grouped by a *typed null*
(`nullif(col, col)`, a null of the column's own type, which also keeps one row per
level), and the levels are stacked with `union(distinct=False)`. So every level is a
plan the optimizer, the spill path and the distributed executor already understand,
and nothing in the aggregate path needs to know that levels exist.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import AggExpr, Expr, col, nullif
from batcher.plan.logical import Union

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["MultiLevelGroupBy", "cube_levels", "rollup_levels"]


def rollup_levels(keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    """The grouping levels of `ROLLUP(k₁, …, kₙ)`: every prefix, longest first.

    Args:
        keys: The rollup keys, most significant first.

    Returns:
        The prefixes of `keys` from the full list down to the empty grand total.
    """
    return [keys[:i] for i in range(len(keys), -1, -1)]


def cube_levels(keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    """The grouping levels of `CUBE(k₁, …, kₙ)`: every subset, largest first.

    Args:
        keys: The cube keys.

    Returns:
        Every subset of `keys`, ordered by decreasing size, each in the original
        key order so the output column order is stable.
    """
    return [
        tuple(c) for size in range(len(keys), -1, -1) for c in itertools.combinations(keys, size)
    ]


class MultiLevelGroupBy:
    """An in-progress multi-level aggregation, from `Dataset.rollup`/`cube`/`grouping_sets`.

    Not constructed directly. Like `GroupBy` it is a lazy builder with one finisher,
    `agg`, which returns a `Dataset` whose rows are every level's groups stacked in
    level order — the aggregated levels first, the grand total last.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"r": ["e", "e", "w"], "c": ["x", "y", "z"], "v": [1, 2, 4]})
            >>> out = ds.rollup("r", "c").agg(total=bt.col("v").sum())
            >>> out.sort("r", "c").to_pydict()["total"]
            [1, 2, 3, 4, 4, 7]
    """

    __slots__ = ("_ds", "_keys", "_levels")

    def __init__(self, ds: Dataset, keys: tuple[str, ...], levels: list[tuple[str, ...]]) -> None:
        """Hold the source dataset, the full key list, and the levels to aggregate."""
        self._ds = ds
        self._keys = keys
        self._levels = levels

    def __repr__(self) -> str:
        """A source-like rendering naming the keys and the level count."""
        return f"MultiLevelGroupBy(keys={list(self._keys)!r}, levels={len(self._levels)})"

    def agg(self, **named: AggExpr | Expr) -> Dataset:
        """Aggregate every grouping level and stack the results.

        Each level's inactive keys read as NULL, which is how SQL marks a subtotal row.
        Use ``grouping(...)`` semantics — testing a key for null — to tell a subtotal
        apart from a genuine null in the data.

        Args:
            **named: Output names bound to aggregate expressions, as for
                :meth:`GroupBy.agg`.

        Returns:
            A new `Dataset` with the key columns followed by the aggregates, one block
            of rows per level.

        Raises:
            PlanError: If no aggregates are given.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"r": ["e", "e", "w"], "v": [1, 2, 4]})
                >>> ds.rollup("r").agg(n=bt.col("v").sum()).sort("r").to_pydict()
                {'r': ['e', 'w', None], 'n': [3, 4, 7]}
        """
        if not named:
            raise PlanError(
                "rollup()/cube()/grouping_sets() need at least one aggregate, "
                "e.g. .agg(total=col('x').sum())"
            )
        # Every level is built from `self._ds`, so every level's plan indexes *this* source
        # list — the same objects in the same order. `Dataset.union` cannot know that: it
        # takes arbitrary datasets, so it renumbers each one's scans and concatenates the
        # source lists. Going through it here would bind the same relation once per level:
        # 5 levels over TPC-DS q22's three tables is 15 bindings of 3 relations. That is not
        # a cosmetic difference — it is what stops plan-level CSE recognizing the levels as
        # sharing a subtree, so each level re-reads and re-joins the whole input, and q22
        # runs 61x DuckDB. Building the `Union` over the one shared list keeps the levels
        # structurally identical below the aggregate, which is the precondition for sharing
        # the work.
        #
        # Deliberately not a change to `Dataset.union`, which must keep renumbering because
        # its inputs are unrelated in general. This is the one caller that can prove they
        # are not.
        frames = [self._level_frame(level, named) for level in self._levels]
        assert frames  # `_levels` is never empty: the grand total is always one
        plans = tuple(f._plan for f in frames)
        if len(plans) == 1:
            return frames[0]
        return self._ds._derive(Union(plans, False))

    def _level_frame(self, level: tuple[str, ...], named: dict[str, AggExpr | Expr]) -> Dataset:
        """One level: group by its active keys, null the rest, and order the columns.

        The inactive keys are grouped by ``nullif(col, col)`` rather than projected
        afterwards, which does two things at once: the null carries the column's own
        type (so every level's schema matches and the union is legal), and it is a
        constant key, so the level collapses to the groups of its active keys.
        """
        active = set(level)
        keyed = {k: col(k) if k in active else nullif(col(k), col(k)) for k in self._keys}
        grouped = self._ds.group_by(**keyed).agg(**named)
        return grouped.select(*self._keys, *named)
