"""Profile value types and rendering — `Decision`, `OpProfile`, `QueryProfile`.

The pure data + tree/JSON rendering half of the profile package. No subsystem imports;
the assembly logic (walking IR, joining estimates to metrics) lives in `collect`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from batcher._internal.mathx import safe_div

__all__ = ["Decision", "OpProfile", "QueryProfile"]


@dataclass(frozen=True, slots=True)
class Decision:
    """One optimizer/resource/execution decision, recorded for `EXPLAIN` + the event log.

    The generalized form of Kyber's join build-side note: any subsystem hand-off worth
    explaining (a chosen join order, a pushed predicate, a spill verdict, an adaptive
    re-optimization) is one `Decision`. `subsystem` is "kyber" | "carbonite" | "core";
    `category` is a short tag (e.g. "selection", "admission", "adaptive"); `detail`
    carries the structured specifics (row counts, costs, provenance).
    """

    subsystem: str
    category: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subsystem": self.subsystem,
            "category": self.category,
            "summary": self.summary,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class OpProfile:
    """One operator's planned estimate joined to its measured execution.

    `est_rows` is `nan` when Kyber left the operator unbudgeted (unknown source size);
    the measured fields are zero and `measured` is `False` for a planned-only profile
    (`explain()` without `analyze`). `depth` is the operator's indentation in the plan
    tree (0 = root).
    """

    op_id: int
    kind: str
    depth: int
    # Planned (Kyber).
    est_rows: float = float("nan")
    provenance: str = ""
    algorithm: str = ""
    # Measured (Core/engine); valid only when `measured`.
    measured: bool = False
    rows_in: int = 0
    rows_out: int = 0
    elapsed_ms: float = 0.0
    # Size of the operator's *output* (Arrow result-array bytes) — NOT its peak working
    # set. A spilling operator can show a tiny `result_bytes` while having processed far
    # more. The engine reports the true peak separately (`OpMetric.peak_bytes`: a breaker's
    # materialized input plus its result); that is what Carbonite's memory model consumes.
    result_bytes: int = 0
    spilled: bool = False
    # Logical bytes routed to disk when `spilled` — the magnitude, not just the fact. 0 for
    # an in-memory operator (or a spill nested in a helper that does not surface its volume).
    spill_bytes: int = 0
    # Measured process peak-RSS growth (bytes) during this operator — the ground-truth memory
    # high-water complementing the Arrow-size estimate. 0 when it set no new high-water.
    peak_rss_bytes: int = 0
    backend: str = ""
    cpu_util: float = 0.0
    threads: int = 0

    @property
    def selectivity(self) -> float:
        """``rows_out / rows_in`` (1.0 when the operator had no input rows)."""
        return safe_div(self.rows_out, self.rows_in, 1.0)

    @property
    def est_error(self) -> float:
        """``rows_out / est_rows`` — how far the estimate missed (`nan` if unknown).

        The number the adaptive controller acts on: ``1.0`` is a perfect estimate,
        ``10.0`` means the operator produced 10x the rows Kyber planned for.
        """
        if not self.measured or math.isnan(self.est_rows) or self.est_rows <= 0:
            return float("nan")
        return self.rows_out / self.est_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "kind": self.kind,
            "depth": self.depth,
            "est_rows": None if math.isnan(self.est_rows) else self.est_rows,
            "provenance": self.provenance,
            "algorithm": self.algorithm,
            "measured": self.measured,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "elapsed_ms": self.elapsed_ms,
            "result_bytes": self.result_bytes,
            "spilled": self.spilled,
            "spill_bytes": self.spill_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "backend": self.backend,
            "cpu_util": self.cpu_util,
            "threads": self.threads,
            "selectivity": self.selectivity,
            "est_error": None if math.isnan(self.est_error) else self.est_error,
        }


@dataclass(frozen=True, slots=True)
class QueryProfile:
    """A whole query's plan + run, the object behind `explain()` and the event log.

    Holds one `OpProfile` per operator (already joined planned↔measured), the
    cross-cutting decisions, and the assembled-once metadata (`carbonite_summary`,
    `adaptive_stages`, the logical/optimized IR). `render()` produces the human tree;
    `to_dict()` the machine-readable document.
    """

    ops: tuple[OpProfile, ...]
    total_ms: float = 0.0
    rows: int = 0
    query_id: str = ""
    measured: bool = False
    distributed: bool = False
    decisions: tuple[Decision, ...] = ()
    carbonite_summary: str = ""
    adaptive_stages: tuple[dict[str, Any], ...] = ()
    logical_ir: dict[str, Any] | None = None
    optimized_ir: dict[str, Any] | None = None
    # The memory budget (bytes) the run was admitted against — the soft envelope Carbonite
    # aims to keep the working set under. `0` when unknown (a planned-only profile). Lets the
    # utilization summary report peak memory as a fraction of budget (the >80% memory target).
    memory_budget_bytes: int = 0
    # Distributed map sub-plan operators (a separate op-id space from the driver tree,
    # so kept apart rather than joined). Populated only for the distributed aggregate path.
    worker_ops: tuple[OpProfile, ...] = ()

    @property
    def spilled(self) -> bool:
        """Whether any operator spilled to disk during the run."""
        return any(o.spilled for o in self.ops)

    @property
    def total_spill_bytes(self) -> int:
        """Total logical bytes spilled to disk across every operator this run."""
        return sum(o.spill_bytes for o in self.ops)

    @property
    def peak_rss_bytes(self) -> int:
        """The measured process peak-RSS high-water growth across the run (max per op)."""
        return max((o.peak_rss_bytes for o in self.ops), default=0)

    @property
    def bottleneck(self) -> OpProfile | None:
        """The operator that took the most wall time, or `None` if nothing measured."""
        measured = [o for o in self.ops if o.measured]
        return max(measured, key=lambda o: o.elapsed_ms, default=None)

    def cpu_utilization_overall(self) -> float:
        """Wall-time-weighted mean per-core CPU utilization across measured ops, in [0, 1].

        The single number for "did this run keep the cores busy?" — each operator's measured
        per-core busy fraction weighted by the wall time it held, so a brief idle op doesn't
        drag down a long saturated one. `0.0` when no operator recorded CPU time (all
        sub-millisecond, or an engine that reports none).
        """
        measured = [o for o in self.ops if o.measured and o.cpu_util > 0.0 and o.elapsed_ms > 0.0]
        total = sum(o.elapsed_ms for o in measured)
        if total <= 0.0:
            return 0.0
        return sum(o.cpu_util * o.elapsed_ms for o in measured) / total

    def utilization_summary(self) -> str:
        """One line grading the run against the hardware-saturation targets (CPU >90%, and the
        peak working set it held), or `""` when nothing measured CPU time.

        A low CPU number is not automatically a problem — an I/O-bound scan or a GPU-dispatch
        stage *should* leave the CPU idle — so the line names the likely cause rather than
        crying wolf. It exists so a user can see, out of the box, whether the engine is
        saturating the box or leaving it idle, and why."""
        util = self.cpu_utilization_overall()
        if util <= 0.0:
            return ""
        peak = self.peak_rss_bytes
        if peak and self.memory_budget_bytes > 0:
            pct = peak / self.memory_budget_bytes * 100.0
            mem = f", peak memory {human_bytes(peak)} ({pct:.0f}% of budget, target >80%)"
        elif peak:
            mem = f", peak working set {human_bytes(peak)}"
        else:
            mem = ""
        if util >= 0.9:
            verdict = "cores saturated"
        elif util >= 0.5:
            verdict = "partly CPU-bound (headroom — larger inputs or wider fan-out would fill it)"
        else:
            verdict = "CPU idle — I/O-, launch-, or GPU-dispatch-bound (not CPU-limited)"
        return f"cpu utilization: {util * 100:.0f}% of cores (target >90%){mem} — {verdict}"

    def bottleneck_summary(self) -> str:
        """One line naming the dominant operator and whether the run is I/O- or compute-bound."""
        b = self.bottleneck
        if b is None:
            return "no operators measured"
        share = (b.elapsed_ms / self.total_ms * 100.0) if self.total_ms else 0.0
        # Classify by the *measured* per-core CPU busy fraction when it was recorded: a low
        # fraction is I/O- or launch-bound whatever the operator kind (a cached scan is not
        # I/O-bound; a stalled join can be memory-bound). Fall back to the kind heuristic when
        # CPU time was not measured (a sub-millisecond op, or an older engine).
        if b.cpu_util > 0:
            kind = "I/O/launch-bound" if b.cpu_util < 0.5 else f"compute-bound ({b.kind})"
        else:
            kind = "I/O-bound (read dominates)" if b.kind == "scan" else f"compute-bound ({b.kind})"
        spill = f" — SPILLED {human_bytes(self.total_spill_bytes)} to disk" if self.spilled else ""
        return f"bottleneck: {b.kind} (op {b.op_id}), {share:.0f}% of wall time — {kind}{spill}"

    def render(self, *, analyze: bool | None = None) -> str:
        """Render the plan as an indented tree.

        `analyze=True` shows the measured columns (actual rows, time, memory, spill);
        `analyze=False` shows the planned estimate only. Defaults to whichever the
        profile carries (`measured`).
        """
        show = self.measured if analyze is None else analyze
        lines = [self._render_op(o, show) for o in self.ops]
        if show:
            lines.append("")
            lines.append(f"total: {self.total_ms:.2f} ms, {self.rows:,} rows out")
            lines.append(self.bottleneck_summary())
            util = self.utilization_summary()
            if util:
                lines.append(util)
        if self.decisions:
            lines.append("")
            lines.append("decisions:")
            lines.extend(f"  - [{d.subsystem}/{d.category}] {d.summary}" for d in self.decisions)
        if show and self.worker_ops:
            lines.append("")
            lines.append("distributed map sub-plan (summed across workers):")
            for o in self.worker_ops:
                lines.append("  " + self._render_op(o, analyze=True).lstrip())
        if show and self.adaptive_stages:
            lines.append("")
            lines.append("adaptive re-optimization:")
            for s in self.adaptive_stages:
                lines.append(
                    f"  - {s.get('kind', '?')} (op {s.get('op_id', '?')}): "
                    f"est≈{s.get('est_rows', 0):,.0f} actual={s.get('actual_rows', 0):,} "
                    f"→ {s.get('action', '')}"
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def _render_op(self, o: OpProfile, analyze: bool) -> str:
        label = f"{'  ' * o.depth}{o.kind}"
        est = "est≈?" if math.isnan(o.est_rows) else f"est≈{o.est_rows:,.0f}"
        prov = f" ({o.provenance})" if o.provenance else ""
        algo = f" [{o.algorithm}]" if o.algorithm else ""
        if not analyze or not o.measured:
            return f"{label:<32}{est}{prov}{algo}"
        share = (o.elapsed_ms / self.total_ms * 100.0) if self.total_ms else 0.0
        err = "" if math.isnan(o.est_error) else f" ({o.est_error:.1f}x)"
        # Show the measured spill *volume* when known, not just the fact of spilling — the
        # magnitude is what tells a 1 GB spill from a 100 GB one at a glance.
        if o.spilled:
            spill = f" [spill {human_bytes(o.spill_bytes)}]" if o.spill_bytes else " [spill]"
        else:
            spill = ""
        rss = f"  rss+{human_bytes(o.peak_rss_bytes)}" if o.peak_rss_bytes else ""
        # Per-core CPU busy fraction: a low value flags an I/O- or launch-bound op (a scan,
        # a GPU dispatch) that packs several per core; near 1.0 flags a CPU-bound one.
        cpu = f"  cpu={o.cpu_util * 100:.0f}%" if o.cpu_util > 0 else ""
        return (
            f"{label:<32}{est} actual={o.rows_out:,}{err}"
            f"  {o.elapsed_ms:.1f}ms ({share:.0f}%){cpu}"
            f"  out={human_bytes(o.result_bytes)}{rss}  {o.backend}{spill}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "measured": self.measured,
            "distributed": self.distributed,
            "total_ms": self.total_ms,
            "rows": self.rows,
            "spilled": self.spilled,
            "cpu_utilization": self.cpu_utilization_overall(),
            "peak_rss_bytes": self.peak_rss_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "total_spill_bytes": self.total_spill_bytes,
            "carbonite_summary": self.carbonite_summary,
            "ops": [o.to_dict() for o in self.ops],
            "worker_ops": [o.to_dict() for o in self.worker_ops],
            "decisions": [d.to_dict() for d in self.decisions],
            "adaptive_stages": list(self.adaptive_stages),
            "logical_ir": self.logical_ir,
            "optimized_ir": self.optimized_ir,
        }


def human_bytes(n: int) -> str:
    """Compact human-readable byte size (e.g. ``512KB``, ``3.4MB``)."""
    if n < 1024:
        return f"{n}B"
    units = ("KB", "MB", "GB", "TB")
    size = float(n)
    for unit in units:
        size /= 1024.0
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.0f}{unit}" if size >= 10 else f"{size:.1f}{unit}"
    return f"{n}B"
