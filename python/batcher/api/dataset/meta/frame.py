"""The `ds.meta` accessor — the entry point to every metadata shortcut.

`ds.meta` is one promise, kept a hundred different ways: **ask the metadata first**. A
Parquet footer, an ORC stripe header, a lakehouse manifest, a warehouse catalog, and an
immutable in-memory relation all already know things a query would otherwise be run to
rediscover — how many rows there are, what a column's extremes are, how many values are
missing, whether a key is unique, whether a join can match at all. Every method here asks
Kyber whether the answer is provable from those statistics, returns it when it is, and runs
the query that computes it when it is not.

The two are indistinguishable by design. A shortcut is `Provenance.EXACT`-gated, so what it
returns *is* what executing would return — only the cost differs. That is what makes it safe
to reach for `ds.meta` by default rather than as an optimisation you have to justify.

The breadth lives on sub-accessors, not on this class: ``ds.meta.col("x")`` for one column,
``.check`` for its predicates, ``.schema`` for types, ``.nulls`` for missing data,
``.approx`` for what the sketches know, ``.storage`` for the bytes on disk, and
``.against(other)`` for a join.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from batcher.api.dataset.meta._facts import MetaBase, answer
from batcher.kyber.shortcuts import distinct, ordering, rows
from batcher.plan.stats import SortOrder

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.api.dataset.meta.approx import ApproxMeta
    from batcher.api.dataset.meta.column import ColumnMeta
    from batcher.api.dataset.meta.nulls import NullsMeta
    from batcher.api.dataset.meta.pair import PairMeta
    from batcher.api.dataset.meta.schema import SchemaMeta
    from batcher.api.dataset.meta.storage import StorageMeta
    from batcher.plan.expr_ir import Expr

__all__ = ["DatasetMeta"]


class DatasetMeta(MetaBase):
    """Metadata shortcuts for a dataset, reached as ``ds.meta``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, 3], "g": ["a", "a", "b"]})
            >>> ds.meta.shape()
            (3, 2)
            >>> ds.meta.none_match(bt.col("x") > 100)
            True
    """

    __slots__ = ()

    # --- sub-accessors ------------------------------------------------------------------

    def col(self, column: str) -> ColumnMeta:
        """One column's metadata shortcuts (bounds, uniqueness, nulls, and `.check`).

        Args:
            column: The column to describe.

        Returns:
            The column accessor.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.col("x").bounds()
                (1, 3)
        """
        from batcher.api.dataset.meta.column import ColumnMeta

        return ColumnMeta(self._ds, column)

    def against(self, other: Dataset) -> PairMeta:
        """Join shortcuts between this dataset and `other` — can the join match at all?

        Args:
            other: The dataset this one would be joined to.

        Returns:
            The pair accessor.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"k": [1, 2]})
                >>> b = bt.from_pydict({"k": [90, 99]})
                >>> a.meta.against(b).join_is_empty("k")
                True
        """
        from batcher.api.dataset.meta.pair import PairMeta

        return PairMeta(self._ds, other)

    @property
    def schema(self) -> SchemaMeta:
        """Type-level shortcuts — which columns are numeric, temporal, nested, and so on.

        Returns:
            The schema accessor.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1], "s": ["a"]}).meta.schema.numeric()
                ['x']
        """
        from batcher.api.dataset.meta.schema import SchemaMeta

        return SchemaMeta(self._ds)

    @property
    def nulls(self) -> NullsMeta:
        """Relation-wide missing-data shortcuts — every column's null count in one question.

        Returns:
            The nulls accessor.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, None]}).meta.nulls.counts()
                {'a': 1}
        """
        from batcher.api.dataset.meta.nulls import NullsMeta

        return NullsMeta(self._ds)

    @property
    def approx(self) -> ApproxMeta:
        """Sketch-backed shortcuts that never execute — free, approximate, or ``None``.

        Returns:
            The approximate accessor.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.approx.rows()
                3.0
        """
        from batcher.api.dataset.meta.approx import ApproxMeta

        return ApproxMeta(self._ds)

    @property
    def storage(self) -> StorageMeta:
        """Physical-layout shortcuts — files, bytes, row groups, partitioning.

        Returns:
            The storage accessor.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).meta.storage.row_count()
                2
        """
        from batcher.api.dataset.meta.storage import StorageMeta

        return StorageMeta(self._ds)

    # --- relation-level shortcuts --------------------------------------------------------

    def shape(self) -> tuple[int, int]:
        """The ``(rows, columns)`` shape of the result — free whenever the row count is.

        Returns:
            The number of rows and the number of columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]}).meta.shape()
                (3, 2)
        """
        known = self.ask(rows.shape)
        return answer(known, lambda: (self._ds.count(), len(self._ds.columns)))

    def count_where(self, predicate: Expr) -> int:
        """How many rows would survive `predicate` — without running the filter when it can.

        Several filtered counts are provable from statistics alone: ``col IS NULL`` is the
        recorded null count, ``col > <above the maximum>`` is zero, an always-true predicate
        is the row count. Anything partial runs the filter.

        Args:
            predicate: The filter to count.

        Returns:
            The number of surviving rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, None]})
                >>> ds.meta.count_where(bt.col("x").is_null())
                1
        """
        return self._ds.filter(predicate).count()

    def is_empty_where(self, predicate: Expr) -> bool:
        """Whether `predicate` would keep no row at all.

        The pruning question: a ``WHERE x > 1000`` over a column whose maximum is 42 is
        provably empty, and answering it from the footer is what lets the file go unread.

        Args:
            predicate: The filter to test.

        Returns:
            ``True`` if no row survives.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.meta.is_empty_where(bt.col("x") > 1000)
                True
        """
        return self._ds.filter(predicate).is_empty()

    def any_match(self, predicate: Expr) -> bool:
        """Whether at least one row satisfies `predicate`.

        Args:
            predicate: The condition to look for.

        Returns:
            ``True`` if some row matches.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.any_match(bt.col("x") > 2)
                True
        """
        return not self.is_empty_where(predicate)

    def none_match(self, predicate: Expr) -> bool:
        """Whether no row satisfies `predicate` — the spelling a skip decision reads best as.

        Args:
            predicate: The condition to rule out.

        Returns:
            ``True`` if no row matches.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.none_match(bt.col("x") > 100)
                True
        """
        return self.is_empty_where(predicate)

    def all_match(self, predicate: Expr) -> bool:
        """Whether every row satisfies `predicate`.

        A row where the predicate evaluates to NULL does **not** satisfy it — the same
        three-valued rule SQL's ``WHERE`` applies, so this agrees with ``count_where`` by
        construction. An empty relation satisfies everything, vacuously.

        Args:
            predicate: The condition every row must meet.

        Returns:
            ``True`` if every row matches.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).meta.all_match(bt.col("x") > 0)
                True
        """
        return self.count_where(predicate) == self._ds.count()

    def is_key(self, columns: str | Sequence[str]) -> bool:
        """Whether `columns` uniquely identify a row and never hold a null — a primary key.

        For a single column this is a footer read whenever an exact distinct count is known.
        A composite key runs one distinct count, because no format records a multi-column one.

        Args:
            columns: The column or columns forming the candidate key.

        Returns:
            ``True`` if the combination is unique and complete.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"id": [1, 2, 3]}).meta.is_key("id")
                True
        """
        names = self.require_columns((columns,) if isinstance(columns, str) else tuple(columns))
        if len(names) == 1:
            known = self.ask(distinct.is_key, names[0], ndv=names)
            if known is not None:
                return known
        return self._is_key_by_execution(names)

    def sorted_by(self) -> tuple[SortOrder, ...]:
        """The ordering the result is *known* to be in, direction included.

        Empty means "no recorded ordering", which is not the same as "unordered" — only a
        declared, order-preserved sort is tracked. A sort on this prefix is a no-op. Each
        key is a `SortOrder` naming the column, whether it descends, and where its nulls
        sit, so a descending order is reported as faithfully as an ascending one.

        Returns:
            The recorded sort prefix.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [3, 1, 2]}).meta.sorted_by()
                ()
                >>> bt.from_pydict({"x": [3, 1, 2]}).sort("x", descending=True).meta.sorted_by()
                (SortOrder(column='x', descending=True, nulls_first=False),)
        """
        known = self.ask(ordering.sorted_columns)
        return () if known is None else known

    def is_known_sorted_by(self, columns: str | Sequence[SortOrder | str]) -> bool:
        """Whether the result is already known to be sorted by `columns` — so a sort can be skipped.

        Deliberately one-sided: ``True`` proves the ordering holds, ``False`` only means it is
        not recorded (the data may well be sorted anyway). Never executes — there is no
        execution that could answer it, since a relation is a bag and its physical order is a
        property of the plan, not of the data.

        Args:
            columns: The ordering to test, as a prefix. A bare column name means ascending,
                nulls-last; pass a `SortOrder` to ask about a descending ordering.

        Returns:
            ``True`` if the recorded ordering starts with `columns`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [3, 1, 2]}).meta.is_known_sorted_by("x")
                False
                >>> bt.from_pydict({"x": [3, 1, 2]}).sort("x").meta.is_known_sorted_by("x")
                True
        """
        keys = (columns,) if isinstance(columns, str) else tuple(columns)
        self.require_columns(tuple(k if isinstance(k, str) else k.column for k in keys))
        return self.ask(ordering.is_sorted_by, keys) is True

    def explain(self) -> dict[str, Any]:
        """What the metadata actually knows — the answer to "why wasn't that free?".

        Never executes. Reports the exact row count (or ``None``), the estimate, the recorded
        ordering, and per column every facet that is provable without a scan. A shortcut that
        fell back to execution did so because the facet it needed is missing here; a filter
        anywhere in the plan is usually the reason, since it turns an exact extreme into a
        mere bound.

        Returns:
            The provable facts, as a nested dictionary.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> report = bt.from_pydict({"x": [1, 2, 3]}).meta.explain()
                >>> report["rows"]
                3
        """
        facts = self.facts()
        if facts is None:
            return {"rows": None, "estimated_rows": None, "sorted_by": (), "columns": {}}
        return {
            "rows": facts.rows,
            "estimated_rows": facts.estimated_rows,
            "sorted_by": facts.sorted_by,
            "columns": {name: _known_facets(facts.col(name)) for name in self._ds.columns},
        }

    def _is_key_by_execution(self, names: tuple[str, ...]) -> bool:
        """A key check the engine performs: every value present, every combination distinct."""
        if any(self._ds.has_nulls(name) for name in names):
            return False
        return self._ds.select(*names).distinct().count() == self._ds.count()


def _known_facets(col: Any) -> dict[str, Any]:
    """The facets of one column that are provable without a scan, omitting the rest."""
    facets = {
        "dtype": str(col.dtype) if col.dtype is not None else None,
        "min": col.min,
        "max": col.max,
        "null_count": col.null_count,
        "n_unique": col.ndv,
        "approx_n_unique": col.approx_ndv,
    }
    return {name: value for name, value in facets.items() if value is not None}
