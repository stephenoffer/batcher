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
        return self.rows_out / self.rows_in if self.rows_in else 1.0

    @property
    def est_error(self) -> float:
        """``rows_out / est_rows`` — how far the estimate missed (`nan` if unknown)."""
        if math.isnan(self.est_rows) or self.est_rows <= 0:
            return float("nan")
        return self.rows_out / self.est_rows


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
