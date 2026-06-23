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

from dataclasses import dataclass

__all__ = ["OpStat", "RunStats"]


@dataclass(frozen=True, slots=True)
class OpStat:
    """Measured metrics for one operator in an executed plan."""

    op_id: int
    kind: str
    rows_in: int
    rows_out: int
    elapsed_ms: float
    peak_bytes: int
    spilled: bool
    backend: str

    @property
    def selectivity(self) -> float:
        """``rows_out / rows_in`` (1.0 when the operator had no input rows)."""
        return self.rows_out / self.rows_in if self.rows_in else 1.0


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
    def from_ops(cls, ops: list[dict], total_ms: float, rows: int) -> RunStats:
        """Build from the engine's raw `ExecMetrics` op dicts (see
        `core.execute_local_metered`)."""
        parsed = tuple(
            OpStat(
                op_id=int(o.get("op_id", 0)),
                kind=str(o.get("kind", "")),
                rows_in=int(o.get("rows_in", 0)),
                rows_out=int(o.get("rows_out", 0)),
                elapsed_ms=float(o.get("elapsed_ns", 0)) / 1e6,
                peak_bytes=int(o.get("peak_bytes", 0)),
                spilled=bool(o.get("spilled", False)),
                backend=str(o.get("backend", "")),
            )
            for o in ops
        )
        return cls(ops=parsed, total_ms=total_ms, rows=rows)

    @property
    def bottleneck(self) -> OpStat | None:
        """The operator that took the most wall time, or `None` if no ops ran."""
        return max(self.ops, key=lambda o: o.elapsed_ms, default=None)

    @property
    def spilled(self) -> bool:
        """Whether any operator spilled to disk during the run."""
        return any(o.spilled for o in self.ops)

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
            f"{'ms':>10}{'peak_kb':>12}  backend"
        )
        lines = [header, "-" * len(header)]
        for o in self.ops:
            lines.append(
                f"{o.op_id:>3}  {o.kind:<12}{o.rows_in:>12}{o.rows_out:>12}"
                f"{o.elapsed_ms:>10.2f}{o.peak_bytes // 1024:>12}  "
                f"{o.backend}{' [spill]' if o.spilled else ''}"
            )
        lines.append("-" * len(header))
        lines.append(f"total: {self.total_ms:.2f} ms, {self.rows} rows out")
        lines.append(self.bottleneck_summary())
        return "\n".join(lines)
