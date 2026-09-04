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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from batcher._internal.humanize import byte_size, display_width, duration_ms, pad, percent, plural
from batcher._internal.mathx import safe_div
from batcher._internal.optional import require
from batcher.plan.profile import QueryUsage
from batcher.plan.profile.render.cells import share_bar

if TYPE_CHECKING:
    from batcher.plan.profile import QueryProfile

__all__ = ["OpStat", "RunStats"]


@dataclass(frozen=True, slots=True)
class OpStat:
    """Measured metrics for one operator in an executed plan.

    The measured fields (`rows_in`/`rows_out`/`elapsed_ms`/`result_bytes`/`spilled`/
    `backend`) are joined to Kyber's planned `est_rows`/`provenance`, so each operator
    carries both what was estimated and what happened (`est_error`). `result_bytes` is the
    operator's *output* size, not peak working set — read `spill_bytes` for how much went to
    disk and `RunStats.usage` for what the run cost the machine.

    The hardware fields (`cpu_ms`, `peak_rss_bytes`, `io_read_bytes`, `io_write_bytes`) are
    only measured on an executor where each operator owns an exclusive wall interval. The
    streaming executor interleaves its operators and reports `0` for them, and by the
    engine's convention **`0` means unmeasured, not zero** — `RunStats.usage` is the reading
    that holds on every tier.
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
    #: CPU milliseconds summed across this operator's worker threads.
    cpu_ms: float = 0.0
    #: Worker threads the operator ran across — the denominator behind `cpu_util`.
    threads: int = 0
    #: Logical bytes the operator routed to disk. `0` when it did not spill.
    spill_bytes: int = 0
    #: Growth in the process's resident set during this operator, in bytes.
    peak_rss_bytes: int = 0
    #: Bytes that reached a block device on read, page-cache hits excluded.
    io_read_bytes: int = 0
    #: Bytes that reached a block device on write, spill writes included.
    io_write_bytes: int = 0

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
                ['backend', 'cpu_ms', 'cpu_util']

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
            "cpu_ms": self.cpu_ms,
            "threads": self.threads,
            "spill_bytes": self.spill_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
            "selectivity": self.selectivity,
            "est_error": self.est_error,
        }


@dataclass(frozen=True, slots=True)
class RunStats:
    """Per-operator measurements for one materialized run, with a bottleneck call.

    Returned by `Dataset.stats()`. Iterate `ops` for the per-operator detail, read
    `bottleneck` for the operator that dominated wall time, `findings` for what the engine
    concluded about the run, and `str(stats)` for a formatted table. Times are wall-clock
    milliseconds measured by the engine.
    """

    ops: tuple[OpStat, ...]
    total_ms: float
    rows: int
    #: What the engine concluded about this run — a spilling operator, a starved GPU, a
    #: filter running after the work it should have removed. A table of numbers still has
    #: to be *read*, and reading it is the skill every performance guide is written to
    #: teach; these are that reading, done. Empty for a healthy run.
    findings: tuple[dict[str, object], ...] = ()
    #: What the run cost the machine — CPU milliseconds, resident-set growth, page faults,
    #: real block-device bytes — measured across the whole execution rather than per
    #: operator. The per-operator hardware fields hold only on a materializing executor;
    #: this one holds on every tier, including the streaming default, and is summed across
    #: workers on a distributed run.
    usage: QueryUsage = field(default_factory=QueryUsage)

    @classmethod
    def from_profile(cls, profile: QueryProfile) -> RunStats:
        """Build from a `QueryProfile` — the measured operators with planned estimates joined.

        Covers the single-node, out-of-core spill, and distributed paths uniformly (the
        profile is assembled from whichever path actually ran). On a distributed run the
        driver tree is unmeasured, so the measured `worker_ops` (the map sub-plan) carry
        the per-operator detail — they are included so `stats()` is never empty there.

        The insight rules run here too, against the same profile the dashboard uses, so a
        user reading `stats()` in a terminal gets the same conclusions as one who opened the
        web UI. They were previously reachable only through `bt.start_ui()`, which meant the
        engine knew a run had spilled or starved its GPU and told nobody who had not gone
        looking.
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
                cpu_ms=o.cpu_ms,
                threads=o.threads,
                spill_bytes=o.spill_bytes,
                peak_rss_bytes=o.peak_rss_bytes,
                io_read_bytes=o.io_read_bytes,
                io_write_bytes=o.io_write_bytes,
            )
            for o in measured
        )
        return cls(
            ops=parsed,
            total_ms=profile.total_ms,
            rows=profile.rows,
            findings=tuple(_findings(profile)),
            usage=profile.usage,
        )

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
        """A short digest: wall time, rows, memory, the bottleneck call, and any findings.

        What to print when you want the shape of a run without the full per-operator
        table. `str(stats)` gives the table; this gives the headline. The findings line
        appears only when there is something to act on, so a healthy run stays three lines.

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
        spill = f", {plural(self.spill_count, 'operator')} spilled" if self.spill_count else ""
        lines = [
            f"wall time: {self.total_ms:.2f} ms across {plural(len(self.ops), 'operator')}",
            f"rows: {self.rows_in:,} read -> {self.rows_out:,} out"
            f", peak output {byte_size(self.peak_memory_bytes)}{spill}",
            self.bottleneck_summary(),
        ]
        wall = self.wall_clock_summary()
        if wall:
            lines.append(wall)
        # Only when the platform reported it. A line reading "0.0 cores busy" on a host that
        # cannot measure CPU time would say the query was idle, which is the opposite of
        # "unmeasured" and exactly the misreading the zero convention exists to prevent.
        if self.usage.measured:
            lines.append(
                f"machine: {self.usage.cpu_ms:.1f} ms CPU"
                f" ({self.usage.cores_busy:.1f} cores busy)"
                + (
                    f", {byte_size(self.usage.peak_rss_bytes)} resident growth"
                    if self.usage.peak_rss_bytes
                    else ""
                )
            )
        actionable = [f for f in self.findings if f.get("severity") in ("critical", "warning")]
        if actionable:
            lines.append(f"findings: {'; '.join(str(f.get('title')) for f in actionable)}")
        return "\n".join(lines)

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
                >>> sorted(ds.stats().to_dict()["usage"])[:3]
                ['cores_busy', 'cpu_ms', 'invol_ctx_switches']

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
            "usage": self.usage.to_dict(),
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
        pd = require(
            "pandas",
            feature="RunStats.to_pandas()",
            provides="pandas",
            extra="pandas",
        )
        return pd.DataFrame([o.to_dict() for o in self.ops])

    def __repr__(self) -> str:
        """The formatted table, so a bare `ds.stats()` in a REPL shows the profile."""
        return self.__str__()

    def _repr_html_(self) -> str:
        """Render the per-operator table as HTML for Jupyter and friends.

        Notebooks call this automatically, so `ds.stats()` in a cell renders a real table
        instead of a monospace blob. Spilled operators are flagged in their own column so
        memory pressure is visible at a glance.

        Every cell is escaped. An operator id or a summary line can carry a column name, and
        a column name comes out of a file — so the only safe assumption about any of it is
        that a person did not write it.
        """
        import html

        style = "text-align:right;padding:2px 8px"

        def cell(value: object, tag: str) -> str:
            return f"<{tag} style='{style}'>{html.escape(str(value))}</{tag}>"

        head = "".join(
            cell(c, "th")
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
            rows.append("<tr>" + "".join(cell(c, "td") for c in cells) + "</tr>")
        caption = html.escape(self.summary()).replace("\n", "<br>")
        return (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:90%'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            f"<p style='font-family:monospace;font-size:85%'>{caption}</p>"
        )

    def bottleneck_summary(self) -> str:
        """One line naming the dominant operator and whether the run is I/O- or
        compute-bound — the triage Ray users do by hand from ``ds.stats()`` logs.

        The share is against **total operator time**, always, and the line says so. Two
        denominators were in use before: wall time for a run whose stages did not overlap,
        and operator time for one whose did. Both were defensible in isolation and together
        they were not, because neither matches what the per-operator table above shows and
        because wall time is not the operators' denominator in the first place: it includes
        planning, optimization, admission and result assembly, which on a short query is
        most of it. The same operator therefore appeared as "71%" in the table and "4% of
        wall time" here. `wall_clock_summary` reports the split that reading actually needs.

        Returns:
            One line naming the operator, its share, and the boundedness call.
        """
        b = self.bottleneck
        if b is None:
            return "no operators executed"
        busy = sum(o.elapsed_ms for o in self.ops)
        share = (b.elapsed_ms / busy * 100.0) if busy else 0.0
        kind = "I/O-bound (read dominates)" if b.kind == "scan" else f"compute-bound ({b.kind})"
        spill = " — SPILLED to disk" if self.spilled else ""
        return f"bottleneck: {b.kind} (op {b.op_id}), {share:.0f}% of operator time — {kind}{spill}"

    def wall_clock_summary(self) -> str:
        """How the wall clock divides between the operators and everything else.

        The half of a run no per-operator table can show. A query that spends 200 µs in its
        operators and 40 ms getting there is not an operator problem, and until this line
        existed the only evidence of that was a bottleneck percentage that looked broken.

        Stages that overlap can sum past the wall clock, which is concurrency rather than
        an error; the line says so instead of reporting a negative remainder.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2, 3]}).filter(bt.col("a") > 1)
                >>> _ = ds.collect()
                >>> ds.stats().wall_clock_summary().startswith("operators:")
                True

        Returns:
            One line, or ``""`` when nothing was measured.
        """
        busy = sum(o.elapsed_ms for o in self.ops)
        if not self.total_ms:
            return ""
        share = percent(busy / self.total_ms)
        if busy > self.total_ms:
            return (
                f"operators: {duration_ms(busy)} of {duration_ms(self.total_ms)} wall clock "
                f"({share}) — stages overlapped, so operator time sums past the clock"
            )
        head = (
            f"operators: {duration_ms(busy)} of {duration_ms(self.total_ms)} wall clock ({share})"
        )
        rest = self.total_ms - busy
        if rest <= 0:
            return head
        return (
            f"{head}; {duration_ms(rest)} elsewhere "
            "(planning, optimization, admission, result assembly)"
        )

    def __str__(self) -> str:
        """The per-operator table, then the totals, then anything worth acting on.

        Column widths come from the data and the padding is display-width-correct
        (`_internal.humanize`), so a long operator kind or a stage named in a
        double-width script shifts nothing to its right. The size column reports bytes in
        binary units rather than the integer kilobytes it used to: ``result_bytes // 1024``
        floors every operator under a mebibyte to ``0``, which on a small query made the
        whole column read zero and told a reader the run produced nothing.
        """
        share_basis = sum(o.elapsed_ms for o in self.ops)
        head = ["OP", "KIND", "ROWS IN", "ROWS OUT", "TIME", "OP SHARE", "OUT", "BACKEND"]
        rows = [
            [
                str(o.op_id),
                o.kind,
                f"{o.rows_in:,}",
                f"{o.rows_out:,}",
                duration_ms(o.elapsed_ms),
                _share_cell(o.elapsed_ms, share_basis),
                byte_size(o.result_bytes),
                o.backend + (" [spill]" if o.spilled else ""),
            ]
            for o in self.ops
        ]
        widths = [
            max(display_width(cell) for cell in (head[c], *(r[c] for r in rows)))
            for c in range(len(head))
        ]
        aligned = ("right", "left", "right", "right", "right", "left", "right", "left")

        def line(cells: Sequence[str]) -> str:
            return "  ".join(
                pad(cell, width, align=side)
                for cell, width, side in zip(cells, widths, aligned, strict=True)
            ).rstrip()

        body = [line(head), *(line(r) for r in rows)]
        rule = "\u2500" * max(display_width(text) for text in body)
        lines = [body[0], rule, *body[1:], rule]
        lines.append(f"total: {self.total_ms:.2f} ms, {plural(self.rows, 'row')} out")
        lines.append(self.bottleneck_summary())
        wall = self.wall_clock_summary()
        if wall:
            lines.append(wall)
        # Only warnings and criticals are printed. An `info` finding is context for someone
        # already investigating, and printing it under every healthy run is how a reader
        # learns to skip this section — which costs them the one that mattered.
        actionable = [f for f in self.findings if f.get("severity") in ("critical", "warning")]
        if actionable:
            lines.append("")
            lines.append(f"findings ({len(actionable)}):")
            for finding in actionable:
                lines.append(f"  [{finding.get('severity')}] {finding.get('title')}")
                lines.append(f"      {finding.get('action')}")
        return "\n".join(lines)


def _share_cell(elapsed_ms: float, basis: float) -> str:
    """One operator's share of the run, as a bar and a percentage.

    The bar is what makes the column scannable: a list of percentages has to be read value
    by value, while a column of bars shows the distribution at a glance.
    """
    share = elapsed_ms / basis if basis else 0.0
    return f"{share_bar(share, 6, True)} {pad(percent(share), 4, align='right')}"


def _findings(profile: QueryProfile) -> list[dict[str, object]]:
    """The insight rules' conclusions about `profile`, or `[]` if they cannot run.

    Deriving findings must never be the reason `stats()` fails: they are commentary on a
    measurement that already succeeded, so any failure inside a rule costs the commentary
    and nothing else.
    """
    try:
        from batcher.observe.insights import derive_insights

        return derive_insights(profile.to_dict())
    except Exception as exc:  # pragma: no cover - commentary must not break a measurement
        from batcher._internal.logging import note_suppressed

        note_suppressed("api", "derive run findings", exc)
        return []
