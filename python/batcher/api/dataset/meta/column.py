"""The `ds.meta.col(...)` accessor — one column's facts, from the footer when it can be.

Every method here is a *shortcut*: it asks Kyber whether the answer is provable from
statistics and returns it if so, and otherwise runs the query that computes it. Which of the
two happened is deliberately invisible, because the answers are identical — that is the
contract the whole metadata layer is built to keep. What changes is the cost: on a Parquet
scan `is_key()` is a footer read, and on a filtered join it is a `COUNT(DISTINCT)`.

The predicate checks (`all_positive`, `contains`, …) live one step further in, on
`ds.meta.col("x").check` — see `checks`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.api.dataset.meta._facts import MetaBase, answer
from batcher.kyber.shortcuts import bounds, distinct, moments, nulls
from batcher.plan.expr_ir import Col

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.api.dataset.meta.checks import ColumnChecks

__all__ = ["ColumnMeta"]


class ColumnMeta(MetaBase):
    """Metadata shortcuts for a single column, reached as ``ds.meta.col("x")``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, 3], "g": ["a", "a", "b"]})
            >>> ds.meta.col("x").bounds()
            (1, 3)
            >>> ds.meta.col("x").is_key()
            True
    """

    __slots__ = ("_column",)

    def __init__(self, ds: Dataset, column: str) -> None:
        """Bind to one column of a dataset; prefer ``ds.meta.col(name)``."""
        super().__init__(ds)
        self._column = self.require_column(column)

    @property
    def check(self) -> ColumnChecks:
        """Predicate checks over this column's values (``all_positive``, ``contains``, …).

        Returns:
            The check accessor bound to this column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.col("x").check.all_positive()
                True
        """
        from batcher.api.dataset.meta.checks import ColumnChecks

        return ColumnChecks(self._ds, self._column)

    def bounds(self) -> tuple[Any, Any]:
        """The column's ``(min, max)`` pair — one footer read instead of two passes.

        Both extremes come from the same statistics, so asking for the pair costs exactly what
        asking for one of them does. Nulls are ignored; an empty or all-null column gives
        ``(None, None)``.

        Returns:
            The minimum and maximum, each ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [3, 1, 2]}).meta.col("x").bounds()
                (1, 3)
        """
        pair = self.ask(bounds.bounds, self._column)
        if pair is not None:
            return pair
        return (self._ds.min(self._column), self._ds.max(self._column))

    def range(self) -> Any:
        """The width of the column's range (``max - min``), for a numeric column.

        Returns:
            The range width, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [3, 1, 10]}).meta.col("x").range()
                9
        """
        return answer(self.ask(bounds.value_range, self._column), self._range_by_execution)

    def midpoint(self) -> float | None:
        """The centre of the column's range, ``(min + max) / 2``.

        The midpoint of the *range*, not the mean and not the median: a column of
        ``[0, 0, 0, 100]`` has a midpoint of 50. It is what you split a range partition on, or
        probe a sorted column at, without reading the data.

        Returns:
            The midpoint, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [0, 0, 0, 100]}).meta.col("x").midpoint()
                50.0
        """
        value = self.ask(bounds.midpoint, self._column)
        if value is not None:
            return value
        low, high = self.bounds()
        return None if low is None or high is None else (float(low) + float(high)) / 2.0

    def abs_max(self) -> float | None:
        """The largest absolute value the column holds, ``max(|min|, |max|)``.

        The number that decides whether an ``int64`` column would fit in an ``int32``, or
        whether a feature needs scaling — answered from the bounds, not from the values.

        Returns:
            The largest magnitude, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [-9, 2, 5]}).meta.col("x").abs_max()
                9.0
        """
        value = self.ask(bounds.abs_max, self._column)
        if value is not None:
            return value
        low, high = self.bounds()
        return None if low is None or high is None else max(abs(float(low)), abs(float(high)))

    def null_fraction(self) -> float:
        """The share of the column that is null, in ``[0, 1]`` — an empty column gives ``0.0``.

        Returns:
            The fraction of rows whose value is null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, None, 3, None]}).meta.col("x").null_fraction()
                0.5
        """
        value = self.ask(nulls.null_fraction, self._column)
        if value is not None:
            return value
        missing, total = self._ds._exec_null_total(self._column)
        return 0.0 if total == 0 else missing / total

    def no_nulls(self) -> bool:
        """Whether the column contains no null at all — the complement of ``has_nulls``.

        Returns:
            ``True`` if every value is non-null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.col("x").no_nulls()
                True
        """
        value = self.ask(nulls.no_nulls, self._column)
        return answer(value, lambda: not self._ds.has_nulls(self._column))

    def n_unique(self) -> int:
        """The exact number of distinct non-null values (SQL ``COUNT(DISTINCT)``).

        Answered from an **exact** distinct count when one is known — an immutable in-memory
        relation computes and caches one per column, so the second query that asks is free.
        Never from a sketch; ``ds.meta.approx.n_unique`` is the approximate one.

        Returns:
            The number of distinct non-null values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 1, 2, 3, 3]}).meta.col("x").n_unique()
                3
        """
        value = self.ask(distinct.n_unique, self._column, ndv=(self._column,))
        return answer(value, lambda: self._ds.n_unique(self._column))

    def is_unique(self) -> bool:
        """Whether every non-null value occurs exactly once.

        A column of all nulls is vacuously unique — no value repeats.

        Returns:
            ``True`` if no non-null value is repeated.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 2]}).meta.col("x").is_unique()
                False
        """
        value = self.ask(distinct.is_unique, self._column, ndv=(self._column,))
        return answer(value, lambda: self.n_unique() == self._nonnull_count())

    def has_duplicates(self) -> bool:
        """Whether some non-null value occurs more than once.

        Returns:
            ``True`` if any non-null value is repeated.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 2]}).meta.col("x").has_duplicates()
                True
        """
        value = self.ask(distinct.has_duplicates, self._column, ndv=(self._column,))
        return answer(value, lambda: not self.is_unique())

    def duplicate_count(self) -> int:
        """How many rows a ``DISTINCT`` on this column would remove.

        ``count(col) - count(distinct col)``: zero exactly when the column is unique.

        Returns:
            The number of non-null values that repeat one already seen.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 1, 1, 2]}).meta.col("x").duplicate_count()
                2
        """
        value = self.ask(distinct.duplicate_count, self._column, ndv=(self._column,))
        return answer(value, lambda: self._nonnull_count() - self.n_unique())

    def is_key(self) -> bool:
        """Whether the column is a primary key — unique *and* never null.

        The fact a join planner wants before it has read a byte: a key on the build side makes
        the join cardinality-preserving.

        Returns:
            ``True`` if every value is present and distinct.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"id": [1, 2, 3]}).meta.col("id").is_key()
                True
        """
        value = self.ask(distinct.is_key, self._column, ndv=(self._column,))
        return answer(value, lambda: self.is_unique() and not self._ds.has_nulls(self._column))

    def is_constant(self) -> bool:
        """Whether every non-null value of the column is the same.

        Proved from the bounds when they are known: a min and a max are values that *occur*,
        so ``min == max`` means one value occurs and no other. An all-null column is
        vacuously constant.

        Returns:
            ``True`` if the column holds at most one distinct value.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [7, 7, 7]}).meta.col("x").is_constant()
                True
        """
        value = self.ask(bounds.is_constant, self._column)
        return answer(value, lambda: self.n_unique() <= 1)

    def constant_value(self) -> Any:
        """The single value the column holds when it is constant, else ``None``.

        ``None`` means "not constant" *or* "constant and that value is null" — ask
        `is_constant` to tell them apart.

        Returns:
            The column's one value, or ``None``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [7, 7, 7]}).meta.col("x").constant_value()
                7
        """
        value = self.ask(bounds.constant_value, self._column)
        if value is not None:
            return value
        return self._ds.min(self._column) if self.is_constant() else None

    def is_low_cardinality(self, max_distinct: int = 128) -> bool:
        """Whether the column has at most `max_distinct` distinct values.

        The dictionary-encode / one-hot / does-this-``GROUP BY``-fit-in-cache question.

        Args:
            max_distinct: The threshold, inclusive.

        Returns:
            ``True`` if the distinct count is at or below the threshold.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"g": ["a", "b", "a"]}).meta.col("g").is_low_cardinality(2)
                True
        """
        value = self.ask(
            distinct.is_low_cardinality, self._column, max_distinct, ndv=(self._column,)
        )
        return answer(value, lambda: self.n_unique() <= max_distinct)

    def is_binary_valued(self) -> bool:
        """Whether the column holds at most two distinct values — a flag, a label, a mask.

        Returns:
            ``True`` if the distinct count is at most two.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"f": [0, 1, 1, 0]}).meta.col("f").is_binary_valued()
                True
        """
        return self.is_low_cardinality(2)

    def sum(self) -> Any:
        """The total of the column's non-null values (SQL ``SUM``).

        Answered from a recorded exact total when the source has one — an in-memory relation
        caches its own, so the second query is free. Otherwise one aggregate pass runs.

        Returns:
            The sum, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.col("x").sum()
                6
        """
        value = self.ask(moments.total, self._column, total=(self._column,))
        return answer(value, lambda: self._ds.sum(self._column))

    def mean(self) -> Any:
        """The average of the column's non-null values (SQL ``AVG``).

        Read from a recorded mean, or derived from a recorded total and the exact non-null
        count; otherwise one aggregate pass runs.

        Returns:
            The mean, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3, 4]}).meta.col("x").mean()
                2.5
        """
        value = self.ask(moments.average, self._column, mean=(self._column,), total=(self._column,))
        return answer(value, lambda: self._ds.mean(self._column))

    def summary(self) -> dict[str, Any]:
        """Everything known about the column, as one dictionary.

        The per-column ``describe``, assembled from the shortcuts above — so on a Parquet scan
        it is a footer read and on an arbitrary plan it is a handful of aggregates. Keys:
        ``dtype``, ``count`` (non-null), ``null_count``, ``min``, ``max``, ``n_unique``.

        Returns:
            The column's facts, keyed by name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 2]}).meta.col("x").summary()["n_unique"]
                2
        """
        low, high = self.bounds()
        return {
            "dtype": self._ds.meta.schema.dtype(self._column),
            "count": self._nonnull_count(),
            "null_count": self._ds.n_null(self._column),
            "min": low,
            "max": high,
            "n_unique": self.n_unique(),
        }

    def _nonnull_count(self) -> int:
        """The non-null count, from metadata when exact (it usually is) else one pass."""
        value = self.ask(nulls.non_null_count, self._column)
        return answer(value, lambda: int(self._ds._exec_scalar(Col(self._column).count())))

    def _range_by_execution(self) -> Any:
        """`max - min` over the executed bounds, or None when the column has none."""
        low, high = self.bounds()
        return None if low is None or high is None else high - low
