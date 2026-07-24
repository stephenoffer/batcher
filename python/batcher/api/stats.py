"""`RunStats` — measured per-operator execution metrics for a `Dataset` run.

The control-plane view of what the data plane actually did: one `OpStat` per
operator (rows in/out, wall time, peak bytes, spill, execution backend), plus a
bottleneck classification. This is the answer to Ray Data's documented gap — no
execution-plan display and weak per-operator metrics (ray-project/ray#55052):
`Dataset.explain()` shows the *planned* shape with estimates; `Dataset.stats()`
shows the *measured* per-operator reality after a run, so "where is my time going"
is a fact, not a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher._internal.mathx import safe_div

if TYPE_CHECKING:
    from batcher.plan.profile import QueryProfile

__all__ = ["OpStat", "RunStats"]


@dataclass(frozen=True, slots=True)
class OpStat:
    """Measured metrics for one operator in an executed plan.

    The measured fields (`rows_in`/`rows_out`/`elapsed_ms`/`result_bytes`/`spilled`/
    `backend`) are joined to Kyber's planned `est_rows`/`provenance`, so each operator
    carries both what was estimated and what happened (`est_error`). `result_bytes` is the
    operator's *output* size, not peak working set — read `spilled` for memory pressure.
    """

    op_id: int
    kind: str
    rows_in: int
    rows_out: int
    elapsed_ms: float
    result_bytes: int
    spilled: bool
    backend: str
    est_rows: float = float("nan")
    provenance: str = ""
    cpu_util: float = 0.0

    @property
    def selectivity(self) -> float:
        """``rows_out / rows_in`` (1.0 when the operator had no input rows)."""
        return safe_div(self.rows_out, self.rows_in, 1.0)

    @property
    def est_error(self) -> float:
        """``rows_out / est_rows`` — how far the estimate missed (`nan` if unknown)."""
        if math.isnan(self.est_rows) or self.est_rows <= 0:
            return float("nan")
        return self.rows_out / self.est_rows

    def to_dict(self) -> dict[str, object]:
        """This operator's measurements as a flat, JSON-encodable dict.

        Includes the derived `selectivity` and `est_error` alongside the raw fields, so a
        metrics pipeline gets the interpreted numbers without re-deriving them.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2, 3]}).filter(bt.col("a") > 1)
                >>> _ = ds.collect()
                >>> sorted(ds.stats().ops[0].to_dict())[:3]
                ['backend', 'cpu_util', 'elapsed_ms']

        Returns:
            A dict of this operator's fields plus `selectivity` and `est_error`.
        """
        return {
            "op_id": self.op_id,
            "kind": self.kind,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "elapsed_ms": self.elapsed_ms,
            "result_bytes": self.result_bytes,
            "spilled": self.spilled,
            "backend": self.backend,
            "est_rows": self.est_rows,
            "provenance": self.provenance,
            "cpu_util": self.cpu_util,
            "selectivity": self.selectivity,
            "est_error": self.est_error,
        }


@dataclass(frozen=True, slots=True)
class RunStats:
    """Per-operator measurements for one materialized run, with a bottleneck call.

    Returned by `Dataset.stats()`. Iterate `ops` for the per-operator detail, read
    `bottleneck` for the operator that dominated wall time, and `str(stats)` for a
    formatted table. Times are wall-clock milliseconds measured by the engine.
    """

    ops: tuple[OpStat, ...]
    total_ms: float
    rows: int

    @classmethod
    def from_profile(cls, profile: QueryProfile) -> RunStats:
        """Build from a `QueryProfile` — the measured operators with planned estimates joined.

        Covers the single-node, out-of-core spill, and distributed paths uniformly (the
        profile is assembled from whichever path actually ran). On a distributed run the
        driver tree is unmeasured, so the measured `worker_ops` (the map sub-plan) carry
        the per-operator detail — they are included so `stats()` is never empty there.
        """
        measured = [o for o in profile.ops if o.measured] + list(profile.worker_ops)
        parsed = tuple(
            OpStat(
                op_id=o.op_id,
                kind=o.kind,
                rows_in=o.rows_in,
                rows_out=o.rows_out,
                elapsed_ms=o.elapsed_ms,
                result_bytes=o.result_bytes,
                spilled=o.spilled,
                backend=o.backend,
                est_rows=o.est_rows,
                provenance=o.provenance,
                cpu_util=o.cpu_util,
            )
            for o in measured
        )
        return cls(ops=parsed, total_ms=profile.total_ms, rows=profile.rows)

    @property
    def bottleneck(self) -> OpStat | None:
        """The operator that took the most wall time, or `None` if no ops ran."""
        return max(self.ops, key=lambda o: o.elapsed_ms, default=None)

    @property
    def spilled(self) -> bool:
        """Whether any operator spilled to disk during the run."""
        return any(o.spilled for o in self.ops)

    @property
    def spill_count(self) -> int:
        """How many operators spilled to disk — the memory-pressure headline number."""
        return sum(1 for o in self.ops if o.spilled)

    @property
    def rows_in(self) -> int:
        """Rows read by the scan operators, i.e. how much data the run actually touched."""
        return sum(o.rows_in for o in self.ops if o.kind == "scan")

    @property
    def rows_out(self) -> int:
        """Rows the run produced — the same number `len()` on the collected result gives."""
        return self.rows

    @property
    def peak_memory_bytes(self) -> int:
        """The largest single operator output, the best available proxy for peak memory.

        The engine measures each operator's *output* size, not its working set, so this is
        a lower bound on true peak memory rather than a measurement of it. Read `spilled`
        for the authoritative signal that memory was actually tight.
        """
        return max((o.result_bytes for o in self.ops), default=0)

    def summary(self) -> str:
        """A three-line digest: wall time, rows, memory, and the bottleneck call.

        What to print when you want the shape of a run without the full per-operator
        table. `str(stats)` gives the table; this gives the headline.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2, 3]}).filter(bt.col("a") > 1)
                >>> _ = ds.collect()
                >>> ds.stats().summary().splitlines()[0].startswith("wall time:")
                True

        Returns:
            A short multi-line summary, ready to print or log.
        """
        spill = f", {self.spill_count} operator(s) spilled" if self.spill_count else ""
        return "\n".join(
            [
                f"wall time: {self.total_ms:.2f} ms across {len(self.ops)} operator(s)",
                f"rows: {self.rows_in:,} read -> {self.rows_out:,} out"
                f", peak output {self.peak_memory_bytes / 1e6:.1f} MB{spill}",
                self.bottleneck_summary(),
            ]
        )

    def to_dict(self) -> dict[str, object]:
        """The whole run as a JSON-encodable dict, totals plus a list of per-operator dicts.

        The export format: hand it to `json.dumps`, push the totals to Prometheus, or ship
        it as a job artifact. Nothing here is a Batcher type, so no consumer needs to
        import Batcher to read it.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2, 3]}).filter(bt.col("a") > 1)
                >>> _ = ds.collect()
                >>> sorted(ds.stats().to_dict())[:4]
                ['bottleneck', 'ops', 'peak_memory_bytes', 'rows_in']

        Returns:
            A dict of run totals, with the per-operator detail under ``"ops"``.
        """
        bottleneck = self.bottleneck
        return {
            "total_ms": self.total_ms,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "peak_memory_bytes": self.peak_memory_bytes,
            "spilled": self.spilled,
            "spill_count": self.spill_count,
            "bottleneck": bottleneck.kind if bottleneck is not None else None,
            "ops": [o.to_dict() for o in self.ops],
        }

    def to_pandas(self) -> object:
        """The per-operator table as a `pandas.DataFrame`, one row per operator.

        For sorting, filtering, and plotting a profile interactively. Requires `pandas`;
        use `to_dict` if you would rather not add the dependency.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2, 3]}).filter(bt.col("a") > 1)
                >>> _ = ds.collect()
                >>> list(ds.stats().to_pandas().columns)[:2]
                ['op_id', 'kind']

        Returns:
            A DataFrame whose columns are the `OpStat` fields plus the derived ones.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            msg = "RunStats.to_pandas() needs pandas — install it, or use to_dict() instead"
            raise ImportError(msg) from exc
        return pd.DataFrame([o.to_dict() for o in self.ops])

    def __repr__(self) -> str:
        """The formatted table, so a bare `ds.stats()` in a REPL shows the profile."""
        return self.__str__()

    def _repr_html_(self) -> str:
        """Render the per-operator table as HTML for Jupyter and friends.

        Notebooks call this automatically, so `ds.stats()` in a cell renders a real table
        instead of a monospace blob. Spilled operators are flagged in their own column so
        memory pressure is visible at a glance.
        """
        head = "".join(
            f"<th style='text-align:right;padding:2px 8px'>{c}</th>"
            for c in ("op", "kind", "rows in", "rows out", "ms", "out KB", "backend", "spill")
        )
        rows = []
        for o in self.ops:
            cells = (
                o.op_id,
                o.kind,
                f"{o.rows_in:,}",
                f"{o.rows_out:,}",
                f"{o.elapsed_ms:.2f}",
                f"{o.result_bytes // 1024:,}",
                o.backend,
                "yes" if o.spilled else "",
            )
            rows.append(
                "<tr>"
                + "".join(f"<td style='text-align:right;padding:2px 8px'>{c}</td>" for c in cells)
                + "</tr>"
            )
        caption = self.summary().replace("\n", "<br>")
        return (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:90%'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            f"<p style='font-family:monospace;font-size:85%'>{caption}</p>"
        )

    def bottleneck_summary(self) -> str:
        """One line naming the dominant operator and whether the run is I/O- or
        compute-bound — the triage Ray users do by hand from ``ds.stats()`` logs."""
        b = self.bottleneck
        if b is None:
            return "no operators executed"
        share = (b.elapsed_ms / self.total_ms * 100.0) if self.total_ms else 0.0
        kind = "I/O-bound (read dominates)" if b.kind == "scan" else f"compute-bound ({b.kind})"
        spill = " — SPILLED to disk" if self.spilled else ""
        return f"bottleneck: {b.kind} (op {b.op_id}), {share:.0f}% of wall time — {kind}{spill}"

    def __str__(self) -> str:
        header = (
            f"{'op':>3}  {'kind':<12}{'rows_in':>12}{'rows_out':>12}"
            f"{'ms':>10}{'out_kb':>12}  backend"
        )
        lines = [header, "-" * len(header)]
        for o in self.ops:
            lines.append(
                f"{o.op_id:>3}  {o.kind:<12}{o.rows_in:>12}{o.rows_out:>12}"
                f"{o.elapsed_ms:>10.2f}{o.result_bytes // 1024:>12}  "
                f"{o.backend}{' [spill]' if o.spilled else ''}"
            )
        lines.append("-" * len(header))
        lines.append(f"total: {self.total_ms:.2f} ms, {self.rows} rows out")
        lines.append(self.bottleneck_summary())
        return "\n".join(lines)
