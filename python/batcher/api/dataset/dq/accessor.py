"""`DatasetDQ` — the `ds.dq` accessor: accumulate constraints, then apply them.

The constraint values live in `constraints` and the result of counting them in `report`;
this module is the fluent builder and the four terminal actions
(`fail`/`drop`/`quarantine`/`validate`) that turn a chain into a dataset or an error.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import reduce
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import DataQualityError, PlanError
from batcher.api.dataset.dq.constraints import Constraint, RowConstraint, UniqueConstraint
from batcher.api.dataset.dq.report import ValidationReport
from batcher.plan.expr_ir import Col, Expr, count, lit, when

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["DatasetDQ"]


class DatasetDQ:
    """Accessor for data-quality expectations over a `Dataset` (``ds.dq``).

    Constraint methods accumulate (returning a new `DatasetDQ`); a terminal method
    (`fail`/`drop`/`quarantine`/`validate`) applies them.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"id": [1, 2, 3], "age": [40, -1, 25]})
            >>> ds.dq.not_null("id").in_range("age", 0, 120).drop().to_pydict()
            {'id': [1, 3], 'age': [40, 25]}
    """

    __slots__ = ("_constraints", "_ds")

    def __init__(self, ds: Dataset, constraints: tuple[Constraint, ...] = ()) -> None:
        """Bind the data-quality accessor to its `Dataset`; reached as `ds.dq`, not direct."""
        self._ds = ds
        self._constraints = constraints

    def _add(self, c: Constraint) -> DatasetDQ:
        return DatasetDQ(self._ds, (*self._constraints, c))

    # --- constraints -------------------------------------------------------
    def not_null(self, *cols: str) -> DatasetDQ:
        """Require every column in `cols` to be non-null.

        Args:
            cols: The columns that must not contain a null.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, None, 3]})
                >>> ds.dq.not_null("id").drop().to_pydict()
                {'id': [1, 3]}
        """
        if not cols:
            raise PlanError("not_null() requires at least one column, e.g. not_null('id')")
        valid = reduce(lambda a, b: a & b, (Col(c).is_not_null() for c in cols))
        return self._add(RowConstraint(f"not_null({', '.join(cols)})", valid))

    def unique(self, keys: str | list[str]) -> DatasetDQ:
        """Require the combination of `keys` to be unique across all rows.

        Args:
            keys: The key column or list of columns whose combination must be unique.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 1, 2]})
                >>> ds.dq.unique("id").drop().to_pydict()
                {'id': [2]}
        """
        key_list = [keys] if isinstance(keys, str) else list(keys)
        if not key_list:
            raise PlanError("unique() requires at least one key column, e.g. unique('id')")
        return self._add(UniqueConstraint(f"unique({', '.join(key_list)})", tuple(key_list)))

    def in_range(self, column: str, low: Any, high: Any) -> DatasetDQ:
        """Require `column` ∈ ``[low, high]`` (NULL passes; add `not_null` to forbid).

        Args:
            column: The column to bound.
            low: Inclusive lower bound.
            high: Inclusive upper bound.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).drop().to_pydict()
                {'x': [1, 2]}
        """
        if low > high:
            raise PlanError(
                f"in_range({column!r}): low ({low!r}) > high ({high!r}) — swap the arguments?"
            )
        c = Col(column)
        return self._add(
            RowConstraint(f"in_range({column}, {low}, {high})", c.is_null() | c.between(low, high))
        )

    def matches(self, column: str, pattern: str) -> DatasetDQ:
        """Require `column` to match the regex `pattern` (NULL passes).

        Args:
            column: The column to test.
            pattern: The regular expression each value must match.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"code": ["A1", "B2", "xx"]})
                >>> ds.dq.matches("code", r"^[A-Z][0-9]$").drop().to_pydict()
                {'code': ['A1', 'B2']}
        """
        c = Col(column)
        return self._add(
            RowConstraint(
                f"matches({column}, {pattern!r})", c.is_null() | c.str.regexp_matches(pattern)
            )
        )

    def accepted_values(self, column: str, values: Iterable[Any]) -> DatasetDQ:
        """Require `column` to be one of `values` (NULL passes).

        Args:
            column: The column to test.
            values: The permitted set of values.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"c": ["a", "b", "z"]})
                >>> ds.dq.accepted_values("c", ["a", "b"]).drop().to_pydict()
                {'c': ['a', 'b']}
        """
        c = Col(column)
        return self._add(
            RowConstraint(f"accepted_values({column})", c.is_null() | c.is_in(list(values)))
        )

    def check(self, predicate: Expr, *, name: str) -> DatasetDQ:
        """A custom constraint — any boolean `predicate` that is TRUE for a valid row.

        Args:
            predicate: A boolean expression that is TRUE for a valid row.
            name: Label for this constraint in the violation report.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.check(bt.col("x") > 0, name="positive").drop().to_pydict()
                {'x': [1, 2]}
        """
        # `total=False`: a user predicate may be NULL for a row (a comparison against a null
        # column), and a NULL validity is a violation. That is handled correctly by executing
        # (`when(valid).then(0).otherwise(1)` catches it) but *not* by the metadata probe, so
        # this constraint opts out of the shortcut rather than risk discharging a contract it
        # has not proved. See `api.dataset.meta.prove`.
        return self._add(RowConstraint(name, predicate, total=False))

    def foreign_key(
        self,
        columns: str | list[str],
        *,
        references: Dataset,
        ref_columns: str | list[str] | None = None,
    ) -> Dataset:
        """Return the **orphan** rows whose `columns` have no matching key in `references`.

        A referential-integrity check: an empty result means every key resolves;
        otherwise the orphans are ready to quarantine. Lowers to an anti-join — no new
        IR. ``ds.dq.foreign_key("customer_id", references=customers)`` returns rows
        referencing a customer that does not exist.

        A row whose key is NULL is **not** an orphan. A NULL foreign key means "no
        reference", not "a reference that is broken" — SQL's own ``FOREIGN KEY``
        constraint accepts it, dbt's ``relationships`` test excludes it, and it is the
        same convention the value constraints on this accessor already follow (NULL
        passes; forbid it explicitly with `not_null`). The bare anti-join says otherwise,
        because NULL matches nothing, so every unset optional key was reported as a
        referential-integrity failure.

        Args:
            columns: The foreign-key column(s) on this dataset.
            references: The dataset holding the referenced keys.
            ref_columns: The key column(s) in `references`; defaults to `columns`.

        Returns:
            A lazy `Dataset` of the orphan rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> orders = bt.from_pydict({"customer_id": [1, 2, 9]})
                >>> customers = bt.from_pydict({"customer_id": [1, 2]})
                >>> orders.dq.foreign_key("customer_id", references=customers).to_pydict()
                {'customer_id': [9]}

                >>> optional = bt.from_pydict({"customer_id": [1, None, 9]})
                >>> optional.dq.foreign_key("customer_id", references=customers).to_pydict()
                {'customer_id': [9]}
        """
        cols = [columns] if isinstance(columns, str) else list(columns)
        ref_cols = (
            cols
            if ref_columns is None
            else ([ref_columns] if isinstance(ref_columns, str) else list(ref_columns))
        )
        ref = references.select(*ref_cols).distinct()
        # Drop the unset keys before the anti-join rather than after: a composite key with
        # a NULL in any part is equally "no reference", and filtering first also keeps
        # those rows out of the join itself.
        declared = reduce(lambda a, b: a & b, (Col(c).is_not_null() for c in cols))
        checked = self._ds.filter(declared)
        return checked.join(ref, left_on=cols, right_on=ref_cols, how="anti")

    def validate(self) -> ValidationReport:
        """Execute the checks and return per-constraint violation counts (no raise).

        Returns:
            A `ValidationReport` of per-constraint violation counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> str(ds.dq.in_range("x", 0, 10).validate())
                'ValidationReport(violations: in_range(x, 0, 10)=1)'
        """
        if self._provably_clean():
            return ValidationReport({c.name: 0 for c in self._constraints})
        violations: dict[str, int] = {}
        rows = [c for c in self._constraints if isinstance(c, RowConstraint)]
        if rows:
            aggs = {
                f"v{i}": when(c.valid).then(lit(0)).otherwise(lit(1)).sum()
                for i, c in enumerate(rows)
            }
            res = self._ds.agg(**aggs).to_pydict()
            for i, c in enumerate(rows):
                violations[c.name] = int((res[f"v{i}"][0]) or 0)
        for u in self._constraints:
            if isinstance(u, UniqueConstraint):
                violations[u.name] = self._duplicate_row_count(u)
        return ValidationReport(violations)

    def _duplicate_row_count(self, unique: UniqueConstraint) -> int:
        """How many *rows* the uniqueness constraint rejects — not how many keys repeat.

        `ValidationReport.total_violations` is documented as the number of violating rows, and
        every row-wise constraint reports one. This counted the duplicated *groups* instead,
        so it under-reported by the size of each group and by an unbounded factor: over
        ``[1, 1, 1, 2]``, `drop` removes three rows and `quarantine` rejects three, while the
        report said `1`. A key repeated a thousand times still said `1`.

        Summing the group sizes matches what `drop`/`quarantine` do — they keep a row iff
        ``count() OVER (PARTITION BY keys) == 1`` — so the non-raising report and the
        non-raising split now agree, which is what anyone reading a monitoring dashboard next
        to a dead-letter sink is entitled to assume.

        Args:
            unique: The uniqueness constraint to measure.

        Returns:
            The number of rows whose key combination occurs more than once.
        """
        duplicated = (
            self._ds.group_by(*unique.keys)
            .agg(__dq_n=count())
            .filter(Col("__dq_n") > 1)
            .agg(__dq_rows=Col("__dq_n").sum())
            .to_pydict()["__dq_rows"]
        )
        # `sum` over no group is NULL, which is zero violating rows.
        return int(duplicated[0]) if duplicated and duplicated[0] is not None else 0

    def fail(self) -> Dataset:
        """Raise `DataQualityError` on any violation; else return the dataset unchanged.

        The data-contract gate at a pipeline boundary.

        Returns:
            The input `Dataset`, unchanged, when every constraint holds.

        Raises:
            DataQualityError: If any constraint is violated, with per-constraint counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.dq.in_range("x", 0, 10).fail().to_pydict()
                {'x': [1, 2, 3]}
        """
        report = self.validate()
        if not report.ok:
            raise DataQualityError(
                f"data-quality check failed: {report}. Use .drop() to keep only valid rows, "
                "or .quarantine() to route violating rows aside.",
                violations=report.violations,
            )
        return self._ds

    def drop(self) -> Dataset:
        """Return only the rows that satisfy every constraint.

        Returns:
            A lazy `Dataset` of the rows passing every constraint.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).drop().to_pydict()
                {'x': [1, 2]}
        """
        if self._provably_clean():
            return self._ds  # every row passes — the filter is the identity, so don't build one
        prepared, valid, helpers = self._prepared()
        kept = prepared.filter(when(valid).then(lit(True)).otherwise(lit(False)))
        return kept.drop(*helpers) if helpers else kept

    def quarantine(self) -> tuple[Dataset, Dataset]:
        """Return ``(clean, rejected)`` so bad rows route to a dead-letter sink.

        Splits the input into valid rows and violating rows instead of failing the run.

        Returns:
            The ``(clean, rejected)`` pair of datasets.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> clean, rejected = ds.dq.in_range("x", 0, 10).quarantine()
                >>> clean.to_pydict(), rejected.to_pydict()
                ({'x': [1, 2]}, {'x': [-3]})
        """
        if self._provably_clean():
            # Nothing violates the contract, so the split is the whole relation and nothing.
            # `limit(0)` is the canonical empty marker — it carries the right schema and the
            # optimizer prunes its input, so the dead-letter side costs nothing to produce.
            return self._ds, self._ds.limit(0)
        prepared, valid, helpers = self._prepared()
        keep = when(valid).then(lit(True)).otherwise(lit(False))
        reject = when(valid).then(lit(False)).otherwise(lit(True))
        clean = prepared.filter(keep)
        bad = prepared.filter(reject)
        if helpers:
            clean, bad = clean.drop(*helpers), bad.drop(*helpers)
        return clean, bad

    def _provably_clean(self) -> bool:
        """Whether metadata proves every constraint holds — see `api.dataset.meta.prove`."""
        from batcher.api.dataset.meta.prove import constraints_provably_hold

        return constraints_provably_hold(self._ds, self._constraints)

    def _prepared(self) -> tuple[Dataset, Expr, list[str]]:
        """Add window-count helpers for uniqueness and return ``(dataset, validity,
        helper_columns)`` where `validity` is TRUE for a row that passes everything."""
        ds = self._ds
        terms: list[Expr] = [c.valid for c in self._constraints if isinstance(c, RowConstraint)]
        helpers: list[str] = []
        uniques = [u for u in self._constraints if isinstance(u, UniqueConstraint)]
        if uniques:
            # A constant non-null column to COUNT over each key partition: COUNT(1)
            # OVER (PARTITION BY keys) is the per-key row count (==1 iff unique).
            ds = ds.with_columns(__dq_one=lit(1))
            helpers.append("__dq_one")
            for i, u in enumerate(uniques):
                h = f"__dq_uniq_{i}"
                ds = ds.window(
                    partition_by=list(u.keys), order_by=[], functions={h: ("count", "__dq_one")}
                )
                terms.append(Col(h) == 1)
                helpers.append(h)
        valid = reduce(lambda a, b: a & b, terms) if terms else lit(True)
        return ds, valid, helpers
