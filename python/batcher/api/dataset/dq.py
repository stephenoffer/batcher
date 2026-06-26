"""The `Dataset.dq` namespace — data-quality expectations with quarantine.

Breadth on `Dataset` lives on accessors (the Polars/v2 pattern), so quality checks
are reached as ``ds.dq.not_null("id").unique(["id"]).in_range("age", 0, 120)`` and
then a terminal action:

- ``.fail()`` — raise `DataQualityError` (with per-constraint counts) if any row
  violates; otherwise return the dataset unchanged. The data-contract gate.
- ``.drop()`` — return only the rows that satisfy every constraint.
- ``.quarantine()`` — return ``(clean, rejected)`` so bad rows route to a
  dead-letter sink instead of failing the pipeline.
- ``.validate()`` — a `ValidationReport` of per-constraint violation counts.

A constraint is just a boolean `Expr` that is TRUE for a valid row (plus the
group-level uniqueness check, which lowers to a window count). Everything lowers to
existing relational ops (FILTER, a keyless AGGREGATE for the report, ``count() OVER
(PARTITION BY keys)`` for uniqueness) — no new IR, and the valid/invalid split is a
provably total partition (validity is forced to a non-null boolean, so
``valid ⊎ invalid == input``). Value constraints (`in_range`/`matches`/
`accepted_values`) treat NULL as **valid** so they compose independently — forbid
nulls explicitly with `not_null` (the dbt/Great-Expectations convention).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import DataQualityError
from batcher.plan.expr_ir import Col, Expr, count, lit, when

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["DatasetDQ", "ValidationReport"]


@dataclass(frozen=True, slots=True)
class RowConstraint:
    """A row-level constraint: `valid` is TRUE exactly for rows that satisfy it."""

    name: str
    valid: Expr


@dataclass(frozen=True, slots=True)
class UniqueConstraint:
    """A uniqueness constraint over `keys` — a row is valid iff its key combination
    occurs once (lowered to ``count() OVER (PARTITION BY keys) == 1``)."""

    name: str
    keys: tuple[str, ...]


Constraint = RowConstraint | UniqueConstraint


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Per-constraint violation counts from `DatasetDQ.validate`."""

    violations: dict[str, int]

    @property
    def ok(self) -> bool:
        """True when no constraint has any violating row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.dq.in_range("x", 0, 10).validate().ok
                True
                >>> bad = bt.from_pydict({"x": [1, 2, -3]})
                >>> bad.dq.in_range("x", 0, 10).validate().ok
                False
        """
        return all(v == 0 for v in self.violations.values())

    @property
    def total_violations(self) -> int:
        """Total number of violating rows summed across every constraint."""
        return sum(self.violations.values())

    def __str__(self) -> str:
        """Render ``ValidationReport(ok)`` or the per-constraint violation counts."""
        if self.ok:
            return "ValidationReport(ok)"
        bad = ", ".join(f"{k}={v}" for k, v in self.violations.items() if v)
        return f"ValidationReport(violations: {bad})"


class DatasetDQ:
    """Accessor for data-quality expectations over a `Dataset` (``ds.dq``).

    Constraint methods accumulate (returning a new `DatasetDQ`); a terminal method
    (`fail`/`drop`/`quarantine`/`validate`) applies them.
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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, None, 3]})
                >>> ds.dq.not_null("id").drop().to_pydict()
                {'id': [1, 3]}
        """
        if not cols:
            raise ValueError("not_null() requires at least one column")
        valid = reduce(lambda a, b: a & b, (Col(c).is_not_null() for c in cols))
        return self._add(RowConstraint(f"not_null({', '.join(cols)})", valid))

    def unique(self, keys: str | list[str]) -> DatasetDQ:
        """Require the combination of `keys` to be unique across all rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 1, 2]})
                >>> ds.dq.unique("id").drop().to_pydict()
                {'id': [2]}
        """
        key_list = [keys] if isinstance(keys, str) else list(keys)
        if not key_list:
            raise ValueError("unique() requires at least one key column")
        return self._add(UniqueConstraint(f"unique({', '.join(key_list)})", tuple(key_list)))

    def in_range(self, column: str, low: Any, high: Any) -> DatasetDQ:
        """Require `column` ∈ ``[low, high]`` (NULL passes; add `not_null` to forbid).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).drop().to_pydict()
                {'x': [1, 2]}
        """
        c = Col(column)
        return self._add(
            RowConstraint(f"in_range({column}, {low}, {high})", c.is_null() | c.between(low, high))
        )

    def matches(self, column: str, pattern: str) -> DatasetDQ:
        """Require `column` to match the regex `pattern` (NULL passes).

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.check(bt.col("x") > 0, name="positive").drop().to_pydict()
                {'x': [1, 2]}
        """
        return self._add(RowConstraint(name, predicate))

    def foreign_key(
        self,
        columns: str | list[str],
        *,
        references: Dataset,
        ref_columns: str | list[str] | None = None,
    ) -> Dataset:
        """Return the **orphan** rows whose `columns` have no matching key in
        `references` (referential-integrity check). An empty result means every key
        resolves; otherwise the orphans are ready to quarantine. Lowers to an
        anti-join — no new IR.

        ``ds.dq.foreign_key("customer_id", references=customers)`` → rows referencing
        a customer that does not exist.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> orders = bt.from_pydict({"customer_id": [1, 2, 9]})
                >>> customers = bt.from_pydict({"customer_id": [1, 2]})
                >>> orders.dq.foreign_key("customer_id", references=customers).to_pydict()
                {'customer_id': [9]}
        """
        cols = [columns] if isinstance(columns, str) else list(columns)
        ref_cols = (
            cols
            if ref_columns is None
            else ([ref_columns] if isinstance(ref_columns, str) else list(ref_columns))
        )
        ref = references.select(*ref_cols).distinct()
        return self._ds.join(ref, left_on=cols, right_on=ref_cols, how="anti")

    # --- terminals ---------------------------------------------------------
    def validate(self) -> ValidationReport:
        """Execute the checks and return per-constraint violation counts (no raise).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> str(ds.dq.in_range("x", 0, 10).validate())
                'ValidationReport(violations: in_range(x, 0, 10)=1)'
        """
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
                dupe_keys = (
                    self._ds.group_by(*u.keys).agg(__dq_n=count()).filter(Col("__dq_n") > 1).count()
                )
                violations[u.name] = int(dupe_keys)
        return ValidationReport(violations)

    def fail(self) -> Dataset:
        """Raise `DataQualityError` if any constraint is violated; else return the
        dataset unchanged — the data-contract gate at a pipeline boundary.

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
                f"data-quality check failed: {report}", violations=report.violations
            )
        return self._ds

    def drop(self) -> Dataset:
        """Return only the rows that satisfy every constraint.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).drop().to_pydict()
                {'x': [1, 2]}
        """
        prepared, valid, helpers = self._prepared()
        kept = prepared.filter(when(valid).then(lit(True)).otherwise(lit(False)))
        return kept.drop(*helpers) if helpers else kept

    def quarantine(self) -> tuple[Dataset, Dataset]:
        """Return ``(clean, rejected)`` — valid rows and violating rows — so the bad
        rows can be written to a dead-letter sink instead of failing the run.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> clean, rejected = ds.dq.in_range("x", 0, 10).quarantine()
                >>> clean.to_pydict(), rejected.to_pydict()
                ({'x': [1, 2]}, {'x': [-3]})
        """
        prepared, valid, helpers = self._prepared()
        keep = when(valid).then(lit(True)).otherwise(lit(False))
        reject = when(valid).then(lit(False)).otherwise(lit(True))
        clean = prepared.filter(keep)
        bad = prepared.filter(reject)
        if helpers:
            clean, bad = clean.drop(*helpers), bad.drop(*helpers)
        return clean, bad

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
