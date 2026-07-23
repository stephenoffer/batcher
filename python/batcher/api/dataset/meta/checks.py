"""The `ds.meta.col(...).check` accessor — predicate questions answered from the bounds.

A min and a max are not merely statistics: they are *values that occur in the column*. That
turns a whole class of questions into arithmetic on two numbers. "Is every amount positive?"
is decided by the minimum. "Is any temperature above the threshold?" is decided by the
maximum — in both directions, which is why a `WHERE x > 1000` over a column whose max is 42
can be answered "no rows" without opening the file.

Every check is over the column's **non-null** values, the way SQL comparisons are: a null
neither satisfies nor violates. A column with no non-null value satisfies every ``all_*``
check vacuously and no ``any_*`` check.

When metadata cannot decide, the check runs the filter that decides it — so the answer is
always the answer, and only the cost moves.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from batcher.api.dataset.meta._facts import MetaBase, answer
from batcher.kyber.shortcuts import checks
from batcher.plan.expr_ir import Col, Expr, lit

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["ColumnChecks"]


class ColumnChecks(MetaBase):
    """Bound-derived predicate checks for one column, reached as ``ds.meta.col("x").check``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"amount": [10, 25, 3]})
            >>> ds.meta.col("amount").check.all_positive()
            True
            >>> ds.meta.col("amount").check.any_greater_than(100)
            False
    """

    __slots__ = ("_column",)

    def __init__(self, ds: Dataset, column: str) -> None:
        """Bind to one column of a dataset; prefer ``ds.meta.col(name).check``."""
        super().__init__(ds)
        self._column = self.require_column(column)

    # --- "every value satisfies" — decided by the bound on the far side ----------------

    def all_greater_than(self, value: Any) -> bool:
        """Whether every non-null value exceeds `value` — the minimum decides it.

        Args:
            value: The exclusive lower threshold.

        Returns:
            ``True`` if no non-null value is at or below `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 9]}).meta.col("x").check.all_greater_than(4)
                True
        """
        return self._all(checks.all_greater_than, value, Col(self._column) <= lit(value))

    def all_greater_equal(self, value: Any) -> bool:
        """Whether every non-null value is at least `value`.

        Args:
            value: The inclusive lower threshold.

        Returns:
            ``True`` if no non-null value is below `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 9]}).meta.col("x").check.all_greater_equal(5)
                True
        """
        return self._all(checks.all_greater_equal, value, Col(self._column) < lit(value))

    def all_less_than(self, value: Any) -> bool:
        """Whether every non-null value is below `value` — the maximum decides it.

        Args:
            value: The exclusive upper threshold.

        Returns:
            ``True`` if no non-null value is at or above `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 9]}).meta.col("x").check.all_less_than(10)
                True
        """
        return self._all(checks.all_less_than, value, Col(self._column) >= lit(value))

    def all_less_equal(self, value: Any) -> bool:
        """Whether every non-null value is at most `value`.

        Args:
            value: The inclusive upper threshold.

        Returns:
            ``True`` if no non-null value is above `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 9]}).meta.col("x").check.all_less_equal(9)
                True
        """
        return self._all(checks.all_less_equal, value, Col(self._column) > lit(value))

    def all_between(self, low: Any, high: Any) -> bool:
        """Whether every non-null value lies in ``[low, high]``, inclusive.

        The range check a data-quality gate runs on every row, answered instead from the two
        numbers in the footer.

        Args:
            low: The inclusive lower bound.
            high: The inclusive upper bound.

        Returns:
            ``True`` if the column's whole range fits inside ``[low, high]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"age": [31, 44]}).meta.col("age").check.all_between(0, 120)
                True
        """
        violating = (Col(self._column) < lit(low)) | (Col(self._column) > lit(high))
        value = self.ask(checks.all_between, self._column, low, high)
        return answer(value, lambda: self._none_survive(violating))

    def all_positive(self) -> bool:
        """Whether every non-null value is strictly greater than zero.

        Returns:
            ``True`` if the minimum is above zero.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).meta.col("x").check.all_positive()
                True
        """
        return self.all_greater_than(0)

    def all_non_negative(self) -> bool:
        """Whether every non-null value is zero or greater.

        Returns:
            ``True`` if the minimum is at least zero.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [0, 2]}).meta.col("x").check.all_non_negative()
                True
        """
        return self.all_greater_equal(0)

    def all_negative(self) -> bool:
        """Whether every non-null value is strictly less than zero.

        Returns:
            ``True`` if the maximum is below zero.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [-1, -2]}).meta.col("x").check.all_negative()
                True
        """
        return self.all_less_than(0)

    def all_non_positive(self) -> bool:
        """Whether every non-null value is zero or less.

        Returns:
            ``True`` if the maximum is at most zero.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [0, -2]}).meta.col("x").check.all_non_positive()
                True
        """
        return self.all_less_equal(0)

    def all_zero(self) -> bool:
        """Whether every non-null value is exactly zero.

        Returns:
            ``True`` if the column's range is ``[0, 0]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [0, 0]}).meta.col("x").check.all_zero()
                True
        """
        return self.all_between(0, 0)

    # --- "some value satisfies" — the other bound decides it, in both directions --------

    def any_greater_than(self, value: Any) -> bool:
        """Whether some non-null value exceeds `value` — the maximum decides it.

        Both directions come free: a maximum above the threshold proves a match exists, and a
        maximum at or below it proves none does. The second half is the one that skips a scan.

        Args:
            value: The exclusive threshold.

        Returns:
            ``True`` if any non-null value is above `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 9]}).meta.col("x").check.any_greater_than(100)
                False
        """
        return self._any(checks.any_greater_than, value, Col(self._column) > lit(value))

    def any_greater_equal(self, value: Any) -> bool:
        """Whether some non-null value is at least `value`.

        Args:
            value: The inclusive threshold.

        Returns:
            ``True`` if any non-null value is at or above `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 9]}).meta.col("x").check.any_greater_equal(9)
                True
        """
        return self._any(checks.any_greater_equal, value, Col(self._column) >= lit(value))

    def any_less_than(self, value: Any) -> bool:
        """Whether some non-null value is below `value` — the minimum decides it.

        Args:
            value: The exclusive threshold.

        Returns:
            ``True`` if any non-null value is below `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 9]}).meta.col("x").check.any_less_than(1)
                False
        """
        return self._any(checks.any_less_than, value, Col(self._column) < lit(value))

    def any_less_equal(self, value: Any) -> bool:
        """Whether some non-null value is at most `value`.

        Args:
            value: The inclusive threshold.

        Returns:
            ``True`` if any non-null value is at or below `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 9]}).meta.col("x").check.any_less_equal(5)
                True
        """
        return self._any(checks.any_less_equal, value, Col(self._column) <= lit(value))

    # --- membership --------------------------------------------------------------------

    def contains(self, value: Any) -> bool:
        """Whether the column holds `value` somewhere.

        Metadata is asymmetric here and that is the point: **absence** is provable — a value
        outside ``[min, max]``, or one a membership bloom rejects, is not in the column, and
        that is the answer that skips the scan. Presence generally is not (bounds say a value
        *could* be there), so a "maybe" runs the filter.

        Args:
            value: The value to look for.

        Returns:
            ``True`` if the column contains `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 5, 9]}).meta.col("x").check.contains(99)
                False
        """
        decided = self.ask(checks.contains, self._column, value)
        return answer(decided, lambda: self._any_survive(Col(self._column) == lit(value)))

    def may_contain(self, value: Any) -> bool:
        """Whether the column *might* hold `value` — free, and never wrong when it says no.

        The one-sided form of `contains`, for a caller that wants a cheap filter rather than
        an answer: it never executes. ``False`` is a proof of absence, so skipping a file, a
        partition, or an entire query on it is sound. ``True`` only means metadata could not
        rule the value out.

        Args:
            value: The value to test for possible membership.

        Returns:
            ``False`` if `value` is provably absent, else ``True``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 5, 9]}).meta.col("x").check.may_contain(99)
                False
        """
        decided = self.ask(checks.may_contain, self._column, value)
        return True if decided is None else decided

    def never_equals(self, value: Any) -> bool:
        """Whether the column provably never equals `value` — the complement of `contains`.

        The spelling a pruning decision wants: ``if ds.meta.col("k").check.never_equals(v):``
        reads as the skip it authorises.

        Args:
            value: The value to rule out.

        Returns:
            ``True`` if no row holds `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 5, 9]}).meta.col("x").check.never_equals(99)
                True
        """
        return not self.contains(value)

    def any_in(self, values: Iterable[Any]) -> bool:
        """Whether the column holds any of `values` (SQL ``IN``).

        Refuted for free when *every* candidate is outside the bounds or rejected by the
        bloom — the ``IN`` list a partition filter is made of, answered without a scan.

        Args:
            values: The candidate values.

        Returns:
            ``True`` if the column holds at least one of them.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 5]}).meta.col("x").check.any_in([90, 99])
                False
        """
        candidates = list(values)
        if not candidates:
            return False  # `x IN ()` matches nothing
        if all(self.ask(checks.may_contain, self._column, v) is False for v in candidates):
            return False  # every candidate provably absent
        matches = [self.ask(checks.contains, self._column, v) for v in candidates]
        if any(m is True for m in matches):
            return True
        predicate: Expr = Col(self._column).is_in(candidates)
        return self._any_survive(predicate)

    def none_in(self, values: Iterable[Any]) -> bool:
        """Whether the column holds none of `values` — the complement of `any_in`.

        Args:
            values: The candidate values.

        Returns:
            ``True`` if no row holds any of them.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 5]}).meta.col("x").check.none_in([90, 99])
                True
        """
        return not self.any_in(values)

    # --- the two execution fallbacks every check above shares ---------------------------

    def _all(self, shortcut: Any, value: Any, violating: Expr) -> bool:
        """An `all_*` check: metadata, else "no row violates it"."""
        return answer(
            self.ask(shortcut, self._column, value), lambda: self._none_survive(violating)
        )

    def _any(self, shortcut: Any, value: Any, satisfying: Expr) -> bool:
        """An `any_*` check: metadata, else "some row satisfies it"."""
        return answer(
            self.ask(shortcut, self._column, value), lambda: self._any_survive(satisfying)
        )

    def _none_survive(self, violating: Expr) -> bool:
        """Whether no row satisfies `violating` — the executed form of an `all_*` check.

        A null value makes the comparison NULL, which `WHERE` drops, so a null is never
        counted as a violation. That is exactly the "nulls neither satisfy nor violate" rule
        the metadata path applies, so the two agree by construction rather than by luck.
        """
        return self._ds.filter(violating).is_empty()

    def _any_survive(self, satisfying: Expr) -> bool:
        """Whether some row satisfies `satisfying` — the executed form of an `any_*` check."""
        return not self._ds.filter(satisfying).is_empty()
