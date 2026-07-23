"""The `ds.meta.nulls` accessor — the whole relation's missing-data shape, in one question.

A Parquet or ORC footer records a null count per column chunk, and a lakehouse manifest
records one per file. So "which columns have gaps, and how big are they" — the first thing
anybody asks of a new dataset — is a metadata read, not the full scan that a
``df.isnull().sum()`` compiles to.

When the footers cannot answer (a filtered relation, a computed column, a source with no
statistics), one aggregate pass computes every column's null count together. Never one pass
per column: the shape of the answer is per-column, but the cost of getting it should not be.
"""

from __future__ import annotations

from batcher.api.dataset.meta._facts import MetaBase, answer
from batcher.kyber.shortcuts import nulls
from batcher.plan.expr_ir import Col
from batcher.plan.expr_ir.constructors import count

__all__ = ["NullsMeta"]


class NullsMeta(MetaBase):
    """Relation-wide null shortcuts, reached as ``ds.meta.nulls``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, None], "b": [1, 2]})
            >>> ds.meta.nulls.counts()
            {'a': 1, 'b': 0}
            >>> ds.meta.nulls.complete_columns()
            ['b']
    """

    __slots__ = ()

    def counts(self) -> dict[str, int]:
        """Every column's null count, keyed by column name.

        Returns:
            The number of nulls in each output column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, None], "b": [1, 2]}).meta.nulls.counts()
                {'a': 1, 'b': 0}
        """
        known = self.ask(nulls.null_counts)
        columns = self._ds.columns
        if known is None or any(name not in known for name in columns):
            return self._counts_by_execution()
        return {name: int(known[name]) for name in columns}

    def fractions(self) -> dict[str, float]:
        """Every column's null count as a share of the rows, in ``[0, 1]``.

        An empty relation reports ``0.0`` for every column rather than dividing by zero.

        Returns:
            The fraction of rows that are null, per column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, None]}).meta.nulls.fractions()
                {'a': 0.5}
        """
        counts = self.counts()
        rows = self._ds.count()
        if rows == 0:
            return dict.fromkeys(counts, 0.0)
        return {name: n / rows for name, n in counts.items()}

    def total(self) -> int:
        """How many null values the relation holds in total, across every column.

        Returns:
            The sum of every column's null count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, None], "b": [None, None]}).meta.nulls.total()
                3
        """
        return sum(self.counts().values())

    def any(self) -> bool:
        """Whether the relation holds a null anywhere.

        Returns:
            ``True`` if at least one column has at least one null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, 2]}).meta.nulls.any()
                False
        """
        return any(n > 0 for n in self.counts().values())

    def is_complete(self) -> bool:
        """Whether the relation has no null at all — the data-contract question.

        Returns:
            ``True`` if every value of every column is present.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, 2]}).meta.nulls.is_complete()
                True
        """
        return answer(self.ask(nulls.is_complete), lambda: not self.any())

    def columns_with_nulls(self) -> list[str]:
        """The columns that hold at least one null, in schema order.

        Returns:
            The names of the columns with missing values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, None], "b": [1, 2]}).meta.nulls.columns_with_nulls()
                ['a']
        """
        counts = self.counts()
        return [name for name in self._ds.columns if counts.get(name, 0) > 0]

    def complete_columns(self) -> list[str]:
        """The columns with no null at all, in schema order.

        Returns:
            The names of the fully-populated columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, None], "b": [1, 2]}).meta.nulls.complete_columns()
                ['b']
        """
        counts = self.counts()
        return [name for name in self._ds.columns if counts.get(name, 0) == 0]

    def _counts_by_execution(self) -> dict[str, int]:
        """Every column's null count in **one** aggregate pass — never one pass per column."""
        columns = self._ds.columns
        aggregates = {f"__bc_nn_{i}__": Col(name).count() for i, name in enumerate(columns)}
        result = self._ds.agg(__bc_rows__=count(), **aggregates).to_pydict()
        rows = int(result["__bc_rows__"][0]) if result["__bc_rows__"] else 0
        counts: dict[str, int] = {}
        for i, name in enumerate(columns):
            values = result[f"__bc_nn_{i}__"]
            non_null = int(values[0]) if values and values[0] is not None else 0
            counts[name] = rows - non_null
        return counts
