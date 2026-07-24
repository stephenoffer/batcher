"""`ValidationReport` — per-constraint violation counts, and the ways to read them."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ValidationReport"]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Per-constraint violation counts from `DatasetDQ.validate`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, -3]})
            >>> report = ds.dq.in_range("x", 0, 10).validate()
            >>> report.ok, report.total_violations
            (False, 1)
    """

    violations: dict[str, int]

    @property
    def ok(self) -> bool:
        """True when no constraint has any violating row.

        Returns:
            True if every constraint has zero violations.

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
        """Total number of violating rows summed across every constraint.

        Returns:
            The sum of the per-constraint violation counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).validate().total_violations
                1
        """
        return sum(self.violations.values())

    def __bool__(self) -> bool:
        """Truthy when the data passed every constraint — ``if report: ...`` reads as "ok".

        The boolean view of `ok`, so a report can gate a branch directly instead of
        checking ``report.ok``.

        Returns:
            True if every constraint has zero violations.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> report = bt.from_pydict({"x": [1, 2, 3]}).dq.in_range("x", 0, 10).validate()
                >>> "clean" if report else "dirty"
                'clean'
        """
        return self.ok

    def __str__(self) -> str:
        """Render ``ValidationReport(ok)`` or the per-constraint violation counts."""
        if self.ok:
            return "ValidationReport(ok)"
        bad = ", ".join(f"{k}={v}" for k, v in self.violations.items() if v)
        return f"ValidationReport(violations: {bad})"
