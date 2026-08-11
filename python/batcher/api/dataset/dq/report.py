"""`ValidationReport` — per-constraint results, and the ways to read them.

A report is a sequence of `ConstraintResult`s, one per constraint, in the order the chain
declared them. The count of violating rows is the number everyone wants first, so it stays
the headline (`violations`, `total_violations`, `str(report)`); the rest — how many rows
were considered, the tolerance that was applied, whether the constraint merely warns — is
what turns a red/green light into something a monitoring dashboard can chart.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ConstraintResult", "ValidationReport"]


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """One constraint's outcome: how many rows violated it, and whether that is a failure.

    Those are different questions once a tolerance is involved. ``mostly=0.99`` over a
    billion rows tolerates ten million violations and still passes, and a `warn` constraint
    never fails at all — so `violations` alone cannot gate a pipeline, and `ok` alone cannot
    tell you the data got worse.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, -3]})
            >>> result = ds.dq.in_range("x", 0, 10).validate().results[0]
            >>> result.violations, result.rows, result.ok
            (1, 3, False)
    """

    name: str
    violations: int
    rows: int = 0
    severity: str = "error"
    mostly: float = 1.0
    kind: str = "row"
    value: float | None = None
    detail: str = ""

    @property
    def pass_rate(self) -> float:
        """The fraction of rows satisfying the constraint; 1.0 over an empty relation.

        Returns:
            The share of considered rows that were valid, between 0.0 and 1.0.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3, 4]})
                >>> ds.dq.in_range("x", 0, 10).validate().results[0].pass_rate
                0.75
        """
        if self.rows <= 0:
            return 1.0
        return max(0.0, (self.rows - self.violations) / self.rows)

    @property
    def ok(self) -> bool:
        """Whether the constraint passed, after its tolerance is applied.

        Returns:
            True when the pass rate reaches `mostly` (or nothing violated it).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3, 4]})
                >>> ds.dq.in_range("x", 0, 10, mostly=0.7).validate().results[0].ok
                True
        """
        if self.violations == 0:
            return True
        if self.mostly >= 1.0:
            return False
        return self.pass_rate >= self.mostly

    @property
    def blocking(self) -> bool:
        """Whether this result should fail a run — a violated `error`-severity constraint.

        Returns:
            True when the constraint failed and its severity is `"error"`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10, severity="warn").validate().results[0].blocking
                False
        """
        return not self.ok and self.severity == "error"

    def to_dict(self) -> dict[str, object]:
        """This result as a plain dictionary, for logging or a metrics sink.

        Returns:
            A JSON-serializable mapping of every field plus the derived rates.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).validate().results[0].to_dict()["violations"]
                1
        """
        out: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "severity": self.severity,
            "violations": self.violations,
            "rows": self.rows,
            "mostly": self.mostly,
            "pass_rate": self.pass_rate,
            "ok": self.ok,
        }
        if self.value is not None:
            out["value"] = self.value
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Per-constraint results from `DatasetDQ.validate`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, -3]})
            >>> report = ds.dq.in_range("x", 0, 10).validate()
            >>> report.ok, report.total_violations
            (False, 1)
    """

    results: tuple[ConstraintResult, ...] = ()

    @property
    def violations(self) -> dict[str, int]:
        """Violating-row counts keyed by constraint name, in declaration order.

        Returns:
            A mapping from each constraint's name to its violation count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).validate().violations
                {'in_range(x, 0, 10)': 1}
        """
        return {r.name: r.violations for r in self.results}

    @property
    def ok(self) -> bool:
        """True when every constraint passed, after tolerances and severities apply.

        A `warn` constraint never makes a report not-ok; read `warnings` for those.

        Returns:
            True if no blocking constraint failed.

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
        return not any(r.blocking for r in self.results)

    @property
    def failed(self) -> tuple[ConstraintResult, ...]:
        """The results that failed and would block a run.

        Returns:
            The blocking failures, in declaration order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> [r.name for r in ds.dq.in_range("x", 0, 10).validate().failed]
                ['in_range(x, 0, 10)']
        """
        return tuple(r for r in self.results if r.blocking)

    @property
    def warnings(self) -> tuple[ConstraintResult, ...]:
        """The results that failed but were declared `severity="warn"`.

        Returns:
            The non-blocking failures, in declaration order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> report = ds.dq.in_range("x", 0, 10, severity="warn").validate()
                >>> report.ok, [r.name for r in report.warnings]
                (True, ['in_range(x, 0, 10)'])
        """
        return tuple(r for r in self.results if not r.ok and r.severity != "error")

    @property
    def total_violations(self) -> int:
        """Total number of violating rows summed across every constraint.

        A relation-level constraint (a row count, a mean) contributes 1 when it fails,
        because there is no violating row to count.

        Returns:
            The sum of the per-constraint violation counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).validate().total_violations
                1
        """
        return sum(r.violations for r in self.results)

    @property
    def rows(self) -> int:
        """How many rows the row-level constraints were evaluated over.

        Returns:
            The relation's row count, or 0 when no row-level constraint ran.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).validate().rows
                3
        """
        return max((r.rows for r in self.results), default=0)

    def result(self, name: str) -> ConstraintResult:
        """The result for one constraint, by the name shown in the report.

        Args:
            name: The constraint's name, as it appears in `violations`.

        Returns:
            That constraint's `ConstraintResult`.

        Raises:
            KeyError: If no constraint of that name is in the report.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> report = ds.dq.in_range("x", 0, 10).validate()
                >>> report.result("in_range(x, 0, 10)").violations
                1
        """
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(f"no constraint named {name!r} in this report; have {list(self.violations)}")

    def to_dict(self) -> dict[str, object]:
        """The whole report as plain data, for a log line or a metrics sink.

        Returns:
            A JSON-serializable mapping with the summary counts and every result.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, -3]})
                >>> ds.dq.in_range("x", 0, 10).validate().to_dict()["ok"]
                False
        """
        return {
            "ok": self.ok,
            "rows": self.rows,
            "total_violations": self.total_violations,
            "constraints": [r.to_dict() for r in self.results],
        }

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
        bad = ", ".join(f"{r.name}={r.violations}" for r in self.results if r.violations)
        if not bad:
            return "ValidationReport(ok)"
        return f"ValidationReport(violations: {bad})"
