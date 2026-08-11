"""`DatasetDQ` — the `ds.dq` accessor: accumulate constraints, then apply them.

The constraint *values* live in `constraints`, the predicates that build them in `checks`,
the machinery that lowers and counts them in `apply`, and the result in `report`. This
module is the fluent builder and the terminal actions, and deliberately holds no logic of
its own: a method here validates its arguments, names the check, and delegates.

Every constraint method takes the same two modifiers, which is what makes the surface
learnable once rather than method by method:

* ``mostly`` — the fraction of rows that must pass for the constraint to *pass*, Great
  Expectations' tolerance. It moves the pass/fail line only. The violating rows are still
  counted and still dropped, because a tolerated violation is one you chose not to fail the
  run over, not a row that became valid.
* ``severity`` — ``"error"`` enforces, ``"warn"`` only reports. A warning never raises in
  `fail`, never removes a row in `drop`, and never lands on the rejected side of a
  `quarantine`, so a new rule can be watched in production before it is enforced.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import replace
from functools import reduce
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import DataQualityError, PlanError
from batcher.api.dataset.dq import apply
from batcher.api.dataset.dq.checks import aggregates, relations, schema, strings, temporal, values
from batcher.api.dataset.dq.constraints import (
    Constraint,
    ReferenceConstraint,
    RowConstraint,
    Severity,
    UniqueConstraint,
)
from batcher.api.dataset.dq.report import ValidationReport
from batcher.plan.expr_ir import Col, Expr, lit, when

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["DatasetDQ"]


def _require_comparable(ds: Any, column: str, low: Any, high: Any) -> None:
    """Reject an `in_range` bound whose type the column cannot be compared against.

    A numeric bound on a text column reached the engine and returned
    ``Invalid comparison operation: Utf8 >= Int64`` — an Arrow message naming two type
    codes and neither the check, the column, nor the bound. The schema knows both sides
    before anything runs.

    Only a clear mismatch is rejected. Numeric bounds against a numeric column, and any
    bound against a column whose type is not yet known, pass through as before.
    """
    import pyarrow as pa

    schema_ = ds.schema
    if column not in schema_.names:
        return  # the constraint's own filter reports an unknown column
    dtype = schema_.field(column).type
    numeric_bounds = all(
        isinstance(b, (int, float)) and not isinstance(b, bool) for b in (low, high)
    )
    text_column = pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
    if numeric_bounds and text_column:
        raise PlanError(
            f"in_range({column!r}, {low!r}, {high!r}) compares numeric bounds against a "
            f"{dtype} column. Point it at a numeric column, cast this one "
            f"(`bt.col({column!r}).cast('float64')`), or use accepted_values/matches for text."
        )


def _require_fraction(check: str, arg: str, value: float) -> None:
    """Reject a tolerance outside ``[0, 1]`` at the call site that wrote it."""
    if not 0.0 <= value <= 1.0:
        raise PlanError(
            f"{check}({arg}={value!r}): must be a fraction in [0, 1] — 0.99 means "
            "'99% of rows must pass'."
        )


def _require_severity(check: str, value: str) -> None:
    """Reject a severity Batcher does not have, rather than treating it as `error`."""
    if value not in ("error", "warn"):
        raise PlanError(f"{check}(severity={value!r}): use 'error' or 'warn'.")


class DatasetDQ:
    """Accessor for data-quality expectations over a `Dataset` (``ds.dq``).

    Constraint methods accumulate (returning a new `DatasetDQ`); a terminal method
    (`fail`/`drop`/`quarantine`/`validate`/`annotate`) applies them.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"id": [1, 2, 3], "age": [40, -1, 25]})
            >>> ds.dq.not_null("id").in_range("age", 0, 120).drop().to_pydict()
            {'id': [1, 3], 'age': [40, 25]}
    """

    __slots__ = ("_constraints", "_ds", "_where")

    def __init__(
        self,
        ds: Dataset,
        constraints: tuple[Constraint, ...] = (),
        scope: Expr | None = None,
    ) -> None:
        """Bind the data-quality accessor to its `Dataset`; reached as `ds.dq`, not direct."""
        self._ds = ds
        self._constraints = constraints
        self._where = scope

    def __repr__(self) -> str:
        """Render the accessor as the chain of constraint names it has accumulated."""
        names = ", ".join(c.name for c in self._constraints) or "no constraints"
        return f"DatasetDQ({names})"

    # --- accumulation ------------------------------------------------------
    def _add(
        self, c: Constraint, *, mostly: float = 1.0, severity: Severity = "error"
    ) -> DatasetDQ:
        """Attach the tolerance, severity, and active scope, then extend the chain."""
        _require_fraction(c.name, "mostly", mostly)
        _require_severity(c.name, severity)
        if isinstance(c, RowConstraint):
            c = replace(c, mostly=mostly, severity=severity, valid=self._scoped(c.valid))
        elif isinstance(c, (UniqueConstraint, ReferenceConstraint)):
            self._reject_scope(c.name)
            c = replace(c, mostly=mostly, severity=severity)
        else:
            # Relation-level and schema constraints take no tolerance: they are one verdict
            # over the whole table, so the way to widen them is to widen their bounds. No
            # method here offers `mostly` for one, which is why this branch simply drops it.
            self._reject_scope(c.name)
            c = replace(c, severity=severity)
        return DatasetDQ(self._ds, (*self._constraints, c), self._where)

    def _scoped(self, valid: Expr) -> Expr:
        """`valid`, weakened so that rows outside the active `where` scope always pass."""
        if self._where is None:
            return valid
        in_scope = when(self._where).then(lit(True)).otherwise(lit(False))
        return ~in_scope | valid

    def _reject_scope(self, name: str) -> None:
        """Refuse a constraint the `where` scope cannot be pushed into, rather than ignore it."""
        if self._where is not None:
            raise PlanError(
                f"where() scopes row-level constraints, and {name} is not one. Filter the "
                "dataset first (`ds.filter(...).dq....`) so the check reads the subset."
            )

    def where(self, predicate: Expr | None) -> DatasetDQ:
        """Scope the constraints added *after* this call to the rows matching `predicate`.

        Great Expectations' ``row_condition`` and Soda's check filter: a rule that applies to
        one slice of the table and would be wrong applied to the rest. A row outside the
        scope passes vacuously, so scoped and unscoped constraints compose in one chain.

        A NULL predicate puts the row outside the scope. Pass `None` to clear the scope for
        constraints added after that point.

        Args:
            predicate: A boolean expression selecting the rows the following constraints
                apply to, or `None` to clear an active scope.

        Returns:
            A new `DatasetDQ` whose subsequent constraints carry the scope.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict(
                ...     {"country": ["US", "FR", "US"], "state": ["CA", None, None]}
                ... )
                >>> checked = ds.dq.where(bt.col("country") == "US").not_null("state")
                >>> checked.validate().violations
                {'not_null(state)': 1}
        """
        return DatasetDQ(self._ds, self._constraints, predicate)

    def on(self, ds: Dataset) -> DatasetDQ:
        """Rebind this chain of constraints to another `Dataset`.

        A data contract is written once and run against many tables — today's partition and
        yesterday's, the staging copy and the production one. This is how a chain becomes
        reusable without a second way to spell every constraint.

        Args:
            ds: The dataset to apply the accumulated constraints to.

        Returns:
            A new `DatasetDQ` carrying the same constraints, bound to `ds`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> contract = bt.from_pydict({"x": [1]}).dq.in_range("x", 0, 10)
                >>> other = bt.from_pydict({"x": [1, 2, 99]})
                >>> contract.on(other).validate().violations
                {'in_range(x, 0, 10)': 1}
        """
        return DatasetDQ(ds, self._constraints, self._where)

    # --- value constraints -------------------------------------------------
    def not_null(self, *cols: str, mostly: float = 1.0, severity: Severity = "error") -> DatasetDQ:
        """Require every column in `cols` to be non-null.

        Args:
            cols: The columns that must not contain a null.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, None, 3]})
                >>> ds.dq.not_null("id").drop().to_pydict()
                {'id': [1, 3]}
        """
        return self._add(values.not_null(cols), mostly=mostly, severity=severity)

    def unique(
        self, keys: str | list[str], *, mostly: float = 1.0, severity: Severity = "error"
    ) -> DatasetDQ:
        """Require the combination of `keys` to be unique across all rows.

        Args:
            keys: The key column or list of columns whose combination must be unique.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

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
        name = f"unique({', '.join(key_list)})"
        return self._add(UniqueConstraint(name, tuple(key_list)), mostly=mostly, severity=severity)

    def in_range(
        self,
        column: str,
        low: Any,
        high: Any,
        *,
        closed: str = "both",
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column` ∈ ``[low, high]`` (NULL passes; add `not_null` to forbid).

        Args:
            column: The column to bound.
            low: Lower bound.
            high: Upper bound.
            closed: Which ends are inclusive — `"both"`, `"left"`, `"right"`, or `"none"`.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).drop().to_pydict()
                {'x': [1, 2]}
        """
        _require_comparable(self._ds, column, low, high)
        return self._add(
            values.in_range(column, low, high, closed), mostly=mostly, severity=severity
        )

    def accepted_values(
        self,
        column: str,
        allowed: Iterable[Any],
        *,
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column` to be one of `allowed` (NULL passes).

        Args:
            column: The column to test.
            allowed: The permitted set of values.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"c": ["a", "b", "z"]})
                >>> ds.dq.accepted_values("c", ["a", "b"]).drop().to_pydict()
                {'c': ['a', 'b']}
        """
        return self._add(values.accepted_values(column, allowed), mostly=mostly, severity=severity)

    def rejected_values(
        self,
        column: str,
        forbidden: Iterable[Any],
        *,
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column` to be none of `forbidden` (NULL passes).

        The deny-list complement of `accepted_values`, for the sentinel values an upstream
        system writes to mean "unknown".

        Args:
            column: The column to test.
            forbidden: The values that must not appear.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"c": ["ok", "N/A", "fine"]})
                >>> ds.dq.rejected_values("c", ["N/A", "unknown"]).drop().to_pydict()
                {'c': ['ok', 'fine']}
        """
        return self._add(
            values.rejected_values(column, forbidden), mostly=mostly, severity=severity
        )

    def positive(
        self,
        column: str,
        *,
        strict: bool = True,
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column` to be greater than zero — or at least zero (NULL passes).

        Args:
            column: The numeric column to test.
            strict: Whether zero is a violation.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"qty": [3, 0, -1]})
                >>> ds.dq.positive("qty").drop().to_pydict()
                {'qty': [3]}
                >>> ds.dq.positive("qty", strict=False).drop().to_pydict()
                {'qty': [3, 0]}
        """
        return self._add(values.positive(column, strict=strict), mostly=mostly, severity=severity)

    def is_finite(
        self, column: str, *, mostly: float = 1.0, severity: Severity = "error"
    ) -> DatasetDQ:
        """Require `column` to hold a finite number — no NaN, no infinity (NULL passes).

        Args:
            column: The float column to test.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, float("nan"), float("inf")]})
                >>> ds.dq.is_finite("x").drop().to_pydict()
                {'x': [1.0]}
        """
        return self._add(values.is_finite(column), mostly=mostly, severity=severity)

    # --- text constraints --------------------------------------------------
    def matches(
        self, column: str, pattern: str, *, mostly: float = 1.0, severity: Severity = "error"
    ) -> DatasetDQ:
        """Require `column` to match the regex `pattern` (NULL passes).

        Args:
            column: The column to test.
            pattern: The regular expression each value must match.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"code": ["A1", "B2", "xx"]})
                >>> ds.dq.matches("code", r"^[A-Z][0-9]$").drop().to_pydict()
                {'code': ['A1', 'B2']}
        """
        return self._add(strings.matches(column, pattern), mostly=mostly, severity=severity)

    def not_matches(
        self, column: str, pattern: str, *, mostly: float = 1.0, severity: Severity = "error"
    ) -> DatasetDQ:
        """Require `column` **not** to match the regex `pattern` (NULL passes).

        Args:
            column: The column to test.
            pattern: The regular expression no value may match.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"note": ["shipped", "TODO fix", "done"]})
                >>> ds.dq.not_matches("note", r"TODO").drop().to_pydict()
                {'note': ['shipped', 'done']}
        """
        return self._add(strings.not_matches(column, pattern), mostly=mostly, severity=severity)

    def matches_format(
        self, column: str, fmt: str, *, mostly: float = 1.0, severity: Severity = "error"
    ) -> DatasetDQ:
        """Require `column` to match a well-known text format (NULL passes).

        The format is one of ``email``, ``url``, ``uuid``, or ``ipv4``. Use `matches` with
        your own pattern for anything else.

        Args:
            column: The column to test.
            fmt: The name of the format.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"email": ["a@x.io", "nope"]})
                >>> ds.dq.matches_format("email", "email").drop().to_pydict()
                {'email': ['a@x.io']}
        """
        return self._add(strings.matches_format(column, fmt), mostly=mostly, severity=severity)

    def str_length_between(
        self,
        column: str,
        low: int,
        high: int | None = None,
        *,
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column`'s character length to lie in ``[low, high]`` (NULL passes).

        Args:
            column: The text column to test.
            low: Inclusive minimum length.
            high: Inclusive maximum length, or `None` for no upper bound.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"iso": ["US", "FRA", "DE"]})
                >>> ds.dq.str_length_between("iso", 2, 2).drop().to_pydict()
                {'iso': ['US', 'DE']}
        """
        return self._add(
            strings.str_length_between(column, low, high), mostly=mostly, severity=severity
        )

    def not_empty(
        self,
        column: str,
        *,
        strip: bool = True,
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column` not to be the empty string (NULL passes).

        Args:
            column: The text column to test.
            strip: Whether a whitespace-only value counts as empty.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"name": ["ada", "", "   "]})
                >>> ds.dq.not_empty("name").drop().to_pydict()
                {'name': ['ada']}
        """
        return self._add(strings.not_empty(column, strip=strip), mostly=mostly, severity=severity)

    # --- cross-column and temporal constraints -----------------------------
    def compare_columns(
        self,
        left: str,
        op: str,
        right: str,
        *,
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require ``left op right`` for every row where both columns are present.

        Args:
            left: The left-hand column name.
            op: One of ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``.
            right: The right-hand column name.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"start": [1, 5], "end": [3, 2]})
                >>> ds.dq.compare_columns("start", "<=", "end").drop().to_pydict()
                {'start': [1], 'end': [3]}
        """
        return self._add(
            relations.compare_columns(left, op, right), mostly=mostly, severity=severity
        )

    def not_in_future(
        self,
        column: str,
        *,
        tolerance: str | dt.timedelta | int | float = 0,
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require no row's `column` to be dated later than now (NULL passes).

        Args:
            column: The timestamp or date column to test.
            tolerance: How far ahead of now a value may be, as a duration string
                (`"5m"`) or a number of seconds, to absorb clock skew.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import datetime as dt
                >>> import batcher as bt
                >>> past = dt.datetime(2020, 1, 1)
                >>> future = dt.datetime(2999, 1, 1)
                >>> ds = bt.from_pydict({"ts": [past, future]})
                >>> ds.dq.not_in_future("ts").validate().violations
                {'not_in_future(ts)': 1}
        """
        return self._add(
            temporal.not_in_future(column, tolerance=tolerance), mostly=mostly, severity=severity
        )

    def check(
        self,
        predicate: Expr,
        *,
        name: str,
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """A custom constraint — any boolean `predicate` that is TRUE for a valid row.

        Args:
            predicate: A boolean expression that is TRUE for a valid row.
            name: Label for this constraint in the violation report.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

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
        return self._add(
            RowConstraint(name, predicate, total=False), mostly=mostly, severity=severity
        )

    def references(
        self,
        columns: str | list[str],
        *,
        to: Dataset,
        ref_columns: str | list[str] | None = None,
        mostly: float = 1.0,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require every non-null `columns` value to resolve to a key in `to`.

        Referential integrity as a *constraint*, so an orphan can be counted, dropped, or
        quarantined alongside every other check in the chain. `foreign_key` is the other
        half of the same question: it hands back the orphan rows themselves.

        Args:
            columns: The foreign-key column(s) on this dataset.
            to: The dataset holding the referenced keys.
            ref_columns: The key column(s) in `to`; defaults to `columns`.
            mostly: The fraction of rows that must pass for the constraint to pass.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> orders = bt.from_pydict({"customer_id": [1, 2, 9]})
                >>> customers = bt.from_pydict({"customer_id": [1, 2]})
                >>> orders.dq.references("customer_id", to=customers).drop().to_pydict()
                {'customer_id': [1, 2]}
        """
        cols, ref_cols = _key_pair(columns, ref_columns, "references")
        name = f"references({', '.join(cols)})"
        return self._add(
            ReferenceConstraint(name, cols, to, ref_cols), mostly=mostly, severity=severity
        )

    # --- relation-level constraints ----------------------------------------
    def row_count_between(
        self, low: int | None = None, high: int | None = None, *, severity: Severity = "error"
    ) -> DatasetDQ:
        """Require the relation to hold between `low` and `high` rows, inclusive.

        Args:
            low: Inclusive minimum row count, or `None` for no minimum.
            high: Inclusive maximum row count, or `None` for no maximum.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).dq.row_count_between(1).validate().ok
                True
        """
        return self._add(aggregates.row_count_between(low, high), severity=severity)

    def mean_between(
        self,
        column: str,
        low: float | None = None,
        high: float | None = None,
        *,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column`'s mean to lie in ``[low, high]``.

        Args:
            column: The numeric column to average.
            low: Inclusive minimum, or `None`.
            high: Inclusive maximum, or `None`.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]})
                >>> ds.dq.mean_between("x", 1.5, 2.5).validate().ok
                True
        """
        return self._add(aggregates.mean_between(column, low, high), severity=severity)

    def sum_between(
        self,
        column: str,
        low: float | None = None,
        high: float | None = None,
        *,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column`'s sum to lie in ``[low, high]``.

        Args:
            column: The numeric column to total.
            low: Inclusive minimum, or `None`.
            high: Inclusive maximum, or `None`.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]})
                >>> ds.dq.sum_between("x", 6.0, 6.0).validate().ok
                True
        """
        return self._add(aggregates.sum_between(column, low, high), severity=severity)

    def median_between(
        self,
        column: str,
        low: float | None = None,
        high: float | None = None,
        *,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column`'s median to lie in ``[low, high]``.

        Args:
            column: The numeric column to measure.
            low: Inclusive minimum, or `None`.
            high: Inclusive maximum, or `None`.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 100.0]})
                >>> ds.dq.median_between("x", 1.0, 5.0).validate().ok
                True
        """
        return self._add(aggregates.median_between(column, low, high), severity=severity)

    def stddev_between(
        self,
        column: str,
        low: float | None = None,
        high: float | None = None,
        *,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column`'s sample standard deviation to lie in ``[low, high]``.

        Args:
            column: The numeric column to measure.
            low: Inclusive minimum, or `None`.
            high: Inclusive maximum, or `None`.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 1.0, 1.0]})
                >>> ds.dq.stddev_between("x", 0.5, None).validate().ok
                False
        """
        return self._add(aggregates.stddev_between(column, low, high), severity=severity)

    def quantile_between(
        self,
        column: str,
        q: float,
        low: float | None = None,
        high: float | None = None,
        *,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column`'s `q`-quantile to lie in ``[low, high]``.

        Args:
            column: The numeric column to measure.
            q: The quantile, between 0 and 1.
            low: Inclusive minimum, or `None`.
            high: Inclusive maximum, or `None`.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> ds.dq.quantile_between("x", 0.5, 2.0, 3.0).validate().ok
                True
        """
        return self._add(aggregates.quantile_between(column, q, low, high), severity=severity)

    def null_rate_below(
        self, column: str, max_rate: float, *, severity: Severity = "error"
    ) -> DatasetDQ:
        """Require `column`'s share of nulls not to exceed `max_rate`.

        Args:
            column: The column to measure.
            max_rate: The largest acceptable null share, between 0 and 1.
            severity: `"error"` to enforce, `"warn"` to report only.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, None, 3, 4]})
                >>> ds.dq.null_rate_below("x", 0.5).validate().ok
                True

        Returns:
            A new `DatasetDQ` with the constraint added.
        """
        return self._add(aggregates.null_rate_below(column, max_rate), severity=severity)

    def distinct_count_between(
        self,
        column: str,
        low: int | None = None,
        high: int | None = None,
        *,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require `column`'s number of distinct values to lie in ``[low, high]``.

        Args:
            column: The column to measure.
            low: Inclusive minimum, or `None`.
            high: Inclusive maximum, or `None`.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"status": ["new", "done", "new"]})
                >>> ds.dq.distinct_count_between("status", 1, 5).validate().ok
                True
        """
        return self._add(aggregates.distinct_count_between(column, low, high), severity=severity)

    def unique_ratio_above(
        self, column: str, min_ratio: float, *, severity: Severity = "error"
    ) -> DatasetDQ:
        """Require `column`'s distinct-to-row ratio to be at least `min_ratio`.

        Args:
            column: The column to measure.
            min_ratio: The smallest acceptable distinct/row ratio, between 0 and 1.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2, 3, 4]})
                >>> ds.dq.unique_ratio_above("id", 0.9).validate().ok
                True
        """
        return self._add(aggregates.unique_ratio_above(column, min_ratio), severity=severity)

    def fresh_within(
        self,
        column: str,
        max_age: str | dt.timedelta | int | float,
        *,
        severity: Severity = "error",
    ) -> DatasetDQ:
        """Require the newest value of `column` to be no older than `max_age`.

        The check a stalled upstream feed fails and every row-level check passes: nothing
        about the values is wrong, there are just no new ones.

        Args:
            column: The timestamp or date column carrying the row's event time.
            max_age: How stale the newest row may be, as a duration string (`"1d"`,
                `"6h"`) or a number of seconds.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import datetime as dt
                >>> import batcher as bt
                >>> stale = dt.datetime(2020, 1, 1)
                >>> bt.from_pydict({"ts": [stale]}).dq.fresh_within("ts", "1d").validate().ok
                False
        """
        return self._add(temporal.fresh_within(column, max_age), severity=severity)

    # --- schema constraints ------------------------------------------------
    def has_columns(self, *names: str, severity: Severity = "error") -> DatasetDQ:
        """Require every column in `names` to be present, answered from the schema.

        Args:
            names: The columns the contract requires.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1], "amount": [2.0]})
                >>> ds.dq.has_columns("id", "amount").validate().ok
                True
        """
        return self._add(schema.has_columns(self._ds.schema, names), severity=severity)

    def no_unexpected_columns(self, *allowed: str, severity: Severity = "error") -> DatasetDQ:
        """Require the schema to hold no column outside `allowed`.

        Args:
            allowed: The complete set of columns the contract permits.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1], "secret": ["x"]})
                >>> ds.dq.no_unexpected_columns("id").validate().ok
                False
        """
        return self._add(schema.no_unexpected_columns(self._ds.schema, allowed), severity=severity)

    def column_types(
        self, expected: Mapping[str, Any], *, severity: Severity = "error"
    ) -> DatasetDQ:
        """Require each named column to have exactly the given type.

        Args:
            expected: Column name to expected type, as a cast name (`"int64"`) or a
                pyarrow type.
            severity: `"error"` to enforce, `"warn"` to report only.

        Returns:
            A new `DatasetDQ` with the constraint added.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1], "name": ["a"]})
                >>> ds.dq.column_types({"id": "int64", "name": "string"}).validate().ok
                True
        """
        return self._add(schema.column_types(self._ds.schema, expected), severity=severity)

    def suggest(self, columns: list[str] | None = None, *, max_categories: int = 25) -> DatasetDQ:
        """Propose the constraints this data already satisfies, appended to the chain.

        The answer to the blank page. Nobody knows from memory which of two hundred columns
        are never null, which are keys, and which are enumerations with nine values, so the
        contract that gets written is the one somebody remembered. This reads the shape of
        the data instead: completeness, keys, sign, small enumerations, and an observed null
        rate with headroom above it.

        **Executes.** It profiles the relation (one pass), measures the numeric minimums
        (one more), and reads the values of up to eight enumeration candidates (one pass
        each). Everything it proposes is true of *this* data now, which is both the point
        and the limit — read the chain with `repr`, delete what is coincidence, and keep
        what is a contract.

        Args:
            columns: The columns to consider; defaults to every column.
            max_categories: Below this many distinct values, a text column is proposed as
                an enumeration rather than left unconstrained.

        Returns:
            A new `DatasetDQ` with the proposed constraints added to the chain.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2, 3], "side": ["buy", "sell", "buy"]})
                >>> proposed = ds.dq.suggest()
                >>> proposed.validate().ok
                True
                >>> "unique(id)" in proposed.validate().violations
                True
        """
        from batcher.api.dataset.dq.suggest import suggest as _suggest

        proposed = _suggest(self._ds, columns, max_categories=max_categories)
        return DatasetDQ(self._ds, (*self._constraints, *proposed), self._where)

    # --- terminal actions --------------------------------------------------
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
        referencing a customer that does not exist. Use `references` instead when you want
        the same question as a constraint the chain can count, drop, or quarantine.

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
        cols, ref_cols = _key_pair(columns, ref_columns, "foreign_key")
        ref = references.select(*ref_cols).distinct()
        # Drop the unset keys before the anti-join rather than after: a composite key with
        # a NULL in any part is equally "no reference", and filtering first also keeps
        # those rows out of the join itself.
        declared = reduce(lambda a, b: a & b, (Col(c).is_not_null() for c in cols))
        checked = self._ds.filter(declared)
        return checked.join(ref, left_on=list(cols), right_on=list(ref_cols), how="anti")

    def validate(self) -> ValidationReport:
        """Execute the checks and return per-constraint results (no raise).

        Returns:
            A `ValidationReport` of per-constraint results.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> str(ds.dq.in_range("x", 0, 10).validate())
                'ValidationReport(violations: in_range(x, 0, 10)=1)'
        """
        return apply.validate(self._ds, self._constraints)

    def fail(self) -> Dataset:
        """Raise `DataQualityError` on any violation; else return the dataset unchanged.

        The data-contract gate at a pipeline boundary. A `warn`-severity constraint is
        reported but never raises, and a constraint within its `mostly` tolerance passes.

        Returns:
            The input `Dataset`, unchanged, when every constraint holds.

        Raises:
            DataQualityError: If any enforced constraint is violated beyond its tolerance,
                with per-constraint counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.dq.in_range("x", 0, 10).fail().to_pydict()
                {'x': [1, 2, 3]}
        """
        report = self.validate()
        if not report.ok:
            failed = ", ".join(f"{r.name}={r.violations}" for r in report.failed)
            raise DataQualityError(
                f"data-quality check failed: {failed}. Use .drop() to keep only valid rows, "
                "or .quarantine() to route violating rows aside.",
                violations=report.violations,
            )
        return self._ds

    def drop(self) -> Dataset:
        """Return only the rows that satisfy every enforced constraint.

        Returns:
            A lazy `Dataset` of the rows passing every enforced constraint.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).drop().to_pydict()
                {'x': [1, 2]}
        """
        apply.reject_unfilterable(self._constraints, "drop")
        apply.schema_gate(self._constraints, "drop")
        if self._provably_clean():
            return self._ds  # every row passes — the filter is the identity, so don't build one
        columns = list(self._ds.schema.names)
        frame, terms = apply.prepared(self._ds, self._constraints)
        kept = frame.filter(apply.validity(terms))
        return apply.restore(kept, columns)

    def quarantine(self) -> tuple[Dataset, Dataset]:
        """Return ``(clean, rejected)`` so bad rows route to a dead-letter sink.

        Splits the input into valid rows and violating rows instead of failing the run.
        The split is total: every input row lands in exactly one side.

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
        apply.reject_unfilterable(self._constraints, "quarantine")
        apply.schema_gate(self._constraints, "quarantine")
        if self._provably_clean():
            # Nothing violates the contract, so the split is the whole relation and nothing.
            # `limit(0)` is the canonical empty marker — it carries the right schema and the
            # optimizer prunes its input, so the dead-letter side costs nothing to produce.
            return self._ds, self._ds.limit(0)
        columns = list(self._ds.schema.names)
        frame, terms = apply.prepared(self._ds, self._constraints)
        valid = apply.validity(terms)
        clean = frame.filter(valid)
        bad = frame.filter(when(valid).then(lit(False)).otherwise(lit(True)))
        return apply.restore(clean, columns), apply.restore(bad, columns)

    def annotate(self, column: str = "dq_failed") -> Dataset:
        """Add `column`, naming the constraints each row failed, comma-separated.

        A clean row gets the empty string. Warn-severity constraints are named too, which
        is how a rule is observed before it is enforced. This is what makes a dead-letter
        sink triageable: the reason travels with the row.

        Args:
            column: The name of the column to add.

        Returns:
            A lazy `Dataset` with the extra text column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, None], "x": [5, -1]})
                >>> ds.dq.not_null("id").positive("x").annotate().to_pydict()
                {'id': [1, None], 'x': [5, -1], 'dq_failed': ['', 'not_null(id),positive(x)']}
        """
        if column in self._ds.schema.names:
            raise PlanError(
                f"annotate(column={column!r}) would overwrite an existing column; "
                "pass a different name."
            )
        return apply.annotate(self._ds, self._constraints, column)

    def _provably_clean(self) -> bool:
        """Whether metadata proves every constraint holds — see `api.dataset.dq.apply`."""
        return apply.provably_clean(self._ds, self._constraints)


def _key_pair(
    columns: str | list[str], ref_columns: str | list[str] | None, check: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize a foreign-key column pair, rejecting a mismatched arity."""
    cols = (columns,) if isinstance(columns, str) else tuple(columns)
    if ref_columns is None:
        ref_cols = cols
    else:
        ref_cols = (ref_columns,) if isinstance(ref_columns, str) else tuple(ref_columns)
    if not cols:
        raise PlanError(f"{check}() requires at least one key column")
    if len(cols) != len(ref_cols):
        raise PlanError(
            f"{check}(): {len(cols)} key column(s) but {len(ref_cols)} reference column(s) — "
            "they must pair up one to one."
        )
    return cols, ref_cols
