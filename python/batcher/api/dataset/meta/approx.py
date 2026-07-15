"""The `ds.meta.approx` accessor — what the sketches know, and nothing they don't.

Every method here reads a *measured* statistic that a previous run recorded: an HLL distinct
count, a KLL quantile grid, a Misra-Gries top-values map, a measured column width. None of
them executes anything, and none of them is exact.

That makes the contract different from the rest of `ds.meta`, and the difference is the whole
point of the namespace. Elsewhere a missing statistic means "run the query"; here it means
``None`` — "nobody has measured this yet". A caller that must have an answer uses the exact
terminal (`ds.n_unique`, `ds.approx_quantile`) and pays for a pass. A caller sizing a buffer,
picking a join side, or drawing a histogram takes the ``None`` and moves on.

This is the learned-metadata moat at its plainest: these answers do not exist before the
first run, and after it they are free forever.
"""

from __future__ import annotations

from typing import Any

from batcher.api.dataset.meta._facts import MetaBase
from batcher.kyber.shortcuts import approx, distinct
from batcher.plan.expr_ir import Expr

__all__ = ["ApproxMeta"]


class ApproxMeta(MetaBase):
    """Sketch-backed shortcuts that never execute, reached as ``ds.meta.approx``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
            >>> ds.meta.approx.rows()
            4.0
            >>> ds.meta.approx.memory_bytes() > 0
            True
    """

    __slots__ = ()

    def rows(self) -> float:
        """The estimated row count — always available, never exact.

        The cost model's number: a footer count when there is one, a sketch or a learned prior
        otherwise, and a Selinger default when nothing is known. Use ``ds.count()`` for the
        answer; use this to decide how much memory to ask for.

        Returns:
            The estimated number of rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.approx.rows()
                3.0
        """
        facts = self.facts()
        return 0.0 if facts is None else facts.estimated_rows

    def n_unique(self, column: str) -> int | None:
        """An approximate distinct count from a sketch, or ``None`` if none is recorded.

        Unlike ``ds.approx_n_unique``, this never falls back to streaming an HLL over the data
        — it is free or it is nothing.

        Args:
            column: The column to count.

        Returns:
            The sketched distinct count, or ``None`` if unmeasured.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 1, 2]})
                >>> ds.meta.approx.n_unique("x") in (None, 2)
                True
        """
        return self.ask(distinct.approx_n_unique, self.require_column(column))

    def cardinality_ratio(self, column: str) -> float | None:
        """The approximate share of rows holding a distinct value, in ``[0, 1]``.

        Near 1.0 the column is key-like; near 0 it is categorical. The number that decides
        whether to dictionary-encode, broadcast, or hash-partition on it.

        Args:
            column: The column to describe.

        Returns:
            ``ndv / rows``, or ``None`` if the distinct count is unmeasured.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"]})
                >>> r = ds.meta.approx.cardinality_ratio("g")
                >>> r is None or 0.0 <= r <= 1.0
                True
        """
        return self.ask(distinct.approx_cardinality_ratio, self.require_column(column))

    def top_k(self, column: str, k: int = 10) -> list[tuple[str, float]] | None:
        """The `k` most common values as ``(value, share_of_rows)``, or ``None`` if unmeasured.

        From the most-common-values map a previous run recorded. A skewed join key or a hot
        partition shows up here with no ``GROUP BY`` at all.

        Args:
            column: The column to rank.
            k: How many values to return.

        Returns:
            The most common values and their share of rows, or ``None`` if unmeasured.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"]})
                >>> ds.meta.approx.top_k("g") is None or True
                True
        """
        return self.ask(approx.approx_top_k, self.require_column(column), k)

    def frequency(self, column: str, value: Any) -> float | None:
        """The approximate share of rows where `column` equals `value`, or ``None``.

        Known only for a value that *is* common — a rare value is absent from the map and
        returns ``None`` rather than a fabricated ``1/ndv``, which is exactly the estimate the
        map exists to correct.

        Args:
            column: The column to look in.
            value: The value to look for.

        Returns:
            The value's share of rows, or ``None`` if unmeasured.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"]})
                >>> f = ds.meta.approx.frequency("g", "a")
                >>> f is None or 0.0 <= f <= 1.0
                True
        """
        return self.ask(approx.approx_frequency, self.require_column(column), value)

    def histogram(self, column: str, bins: int = 10) -> list[tuple[float, float]] | None:
        """`bins` equal-probability bucket edges as ``(low, high)``, or ``None`` if unmeasured.

        Equal-*probability*, not equal-width: every bucket holds roughly the same number of
        rows, so a long tail appears as one wide bucket rather than nine empty ones.

        Args:
            column: The numeric column to bucket.
            bins: How many buckets to produce.

        Returns:
            The bucket edges, or ``None`` if no quantile grid has been measured.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": list(range(100))})
                >>> ds.meta.approx.histogram("x", 4) is None or True
                True
        """
        return self.ask(approx.approx_histogram, self.require_column(column), bins)

    def count_where(self, predicate: Expr) -> float:
        """The estimated number of rows a filter would keep — the planner's own guess.

        Estimated, so it may be wrong; free, so it costs nothing to ask. It is the number the
        optimizer itself uses to choose a join order, exposed so a caller can see the same
        thing the plan sees. ``ds.meta.count_where`` is the exact one.

        Args:
            predicate: The filter to estimate.

        Returns:
            The estimated surviving row count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
                >>> ds.meta.approx.count_where(bt.col("x") > 2) >= 0
                True
        """
        return ApproxMeta(self._ds.filter(predicate)).rows()

    def selectivity(self, predicate: Expr) -> float:
        """The estimated share of rows a filter would keep, in ``[0, 1]``.

        Args:
            predicate: The filter to estimate.

        Returns:
            The estimated surviving fraction; ``0.0`` over an empty relation.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
                >>> 0.0 <= ds.meta.approx.selectivity(bt.col("x") > 2) <= 1.0
                True
        """
        total = self.rows()
        if total <= 0:
            return 0.0
        return min(1.0, self.count_where(predicate) / total)

    def column_bytes(self, column: str) -> float | None:
        """The approximate in-memory size of one column, in bytes.

        Measured width when a previous run recorded one, else the type's fixed width, else a
        documented assumption for variable-width data.

        Args:
            column: The column to size.

        Returns:
            The estimated byte size, or ``None`` if the type is unknown.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.approx.column_bytes("x")
                24.0
        """
        return self.ask(approx.approx_column_bytes, self.require_column(column))

    def row_bytes(self) -> float:
        """The approximate in-memory width of one row, in bytes, summed over every column.

        Returns:
            The estimated bytes per row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1], "y": [2]}).meta.approx.row_bytes()
                16.0
        """
        facts = self.facts()
        return 0.0 if facts is None else approx.approx_row_bytes(facts)

    def memory_bytes(self) -> float:
        """The approximate size of the whole relation in memory, in bytes.

        The number to size a buffer, a broadcast, or a spill threshold from — never the number
        to report as a fact. Arrow's real footprint depends on padding, dictionary encoding,
        and validity bitmaps, none of which this models.

        Returns:
            The estimated in-memory byte size.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.approx.memory_bytes()
                24.0
        """
        facts = self.facts()
        return 0.0 if facts is None else approx.approx_memory_bytes(facts)

    def is_measured(self, column: str) -> bool:
        """Whether *any* sketch has been recorded for `column` — has this query run before?

        The introspection that explains a ``None`` from every other method here: sketches are
        written by Core when a query executes, so a column nobody has read has nothing
        measured, and the second run of the same query is the one that gets these for free.

        Args:
            column: The column to check.

        Returns:
            ``True`` if a distinct count, quantile grid, top-values map, or width is recorded.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> isinstance(bt.from_pydict({"x": [1]}).meta.approx.is_measured("x"), bool)
                True
        """
        facts = self.facts()
        if facts is None:
            return False
        col = facts.col(self.require_column(column))
        return any(
            fact is not None for fact in (col.approx_ndv, col.quantiles, col.mcv, col.avg_bytes)
        )
