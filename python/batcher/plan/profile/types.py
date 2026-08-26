"""Profile value types and rendering — `Decision`, `OpProfile`, `QueryProfile`.

The pure data + tree/JSON rendering half of the profile package. No subsystem imports;
the assembly logic (walking IR, joining estimates to metrics) lives in `collect`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from batcher._internal.hardware import hardware_profile
from batcher._internal.humanize import byte_size
from batcher._internal.mathx import safe_div
from batcher.plan.feedback import CONTENDED_PREEMPTIONS_PER_CORE_SECOND, preemption_rate

__all__ = ["Decision", "OpProfile", "QueryProfile", "QueryUsage"]


@dataclass(frozen=True, slots=True)
class QueryUsage:
    """What one run cost the machine, measured across the whole execution.

    The per-operator counters on `OpProfile` are only sound on an executor where each
    operator owns an exclusive wall interval. The streaming executor — the default tier for
    most queries — interleaves its operators, so it reports zeros there and says so, which
    left the common case with no CPU, memory, or disk accounting at all. This is the
    measurement that holds on every tier, because the attribution problem is the operator
    boundary and not the counters: the OS reports process-wide totals, and the delta across
    one execution belongs to that execution whatever happened inside it.

    Every field is `0` when the platform could not report it, and by the engine's convention
    **`0` means unmeasured, not zero** — do not read a zero `io_read_bytes` as "read nothing".
    """

    #: Wall-clock milliseconds the engine was executing, excluding planning and admission.
    wall_ms: float = 0.0
    #: CPU milliseconds (user + system) summed across every worker thread.
    cpu_ms: float = 0.0
    #: Growth in the process's peak resident set during the run, in bytes.
    peak_rss_bytes: int = 0
    #: Page faults served from memory — first touch of newly committed pages.
    minor_faults: int = 0
    #: Page faults that required disk I/O. Any meaningful count means the box was paging
    #: against the query, which looks exactly like slow compute from the inside.
    major_faults: int = 0
    #: Times the run gave up a CPU to wait on something (I/O, a lock, a queue).
    vol_ctx_switches: int = 0
    #: Times the scheduler took a CPU away — contention for cores the process was told it had.
    invol_ctx_switches: int = 0
    #: Bytes that actually reached a block device on read, page-cache hits excluded.
    io_read_bytes: int = 0
    #: Bytes that actually reached a block device on write, spill writes included.
    io_write_bytes: int = 0

    @property
    def cores_busy(self) -> float:
        """Mean cores kept busy over the run — ``cpu_ms / wall_ms``.

        The single number that separates a query which saturated the box from one that ran a
        wide plan on one core, and the one figure a per-operator utilization ratio cannot be
        summed into. `0.0` when either term is unmeasured.
        """
        return safe_div(self.cpu_ms, self.wall_ms)

    @property
    def measured(self) -> bool:
        """Whether the platform reported anything at all for this run."""
        return self.wall_ms > 0 or self.cpu_ms > 0

    def to_dict(self) -> dict[str, Any]:
        """The usage as a flat, JSON-encodable dict, with `cores_busy` derived."""
        return {
            "wall_ms": self.wall_ms,
            "cpu_ms": self.cpu_ms,
            "cores_busy": self.cores_busy,
            "peak_rss_bytes": self.peak_rss_bytes,
            "minor_faults": self.minor_faults,
            "major_faults": self.major_faults,
            "vol_ctx_switches": self.vol_ctx_switches,
            "invol_ctx_switches": self.invol_ctx_switches,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
        }

    @classmethod
    def from_metrics(cls, doc: dict[str, Any] | None) -> QueryUsage:
        """Build from the engine's `ExecMetrics.query` block (nanoseconds → milliseconds).

        Tolerant of an absent or partial block, so a profile assembled against an engine
        build that predates the measurement degrades to all-zero rather than raising.

        Args:
            doc: The ``query`` object from an `ExecMetrics` document, or `None`.

        Returns:
            The usage, all-zero when nothing was reported.
        """
        if not doc:
            return cls()
        return cls(
            wall_ms=float(doc.get("wall_ns", 0)) / 1e6,
            cpu_ms=float(doc.get("cpu_ns", 0)) / 1e6,
            peak_rss_bytes=int(doc.get("peak_rss_bytes", 0)),
            minor_faults=int(doc.get("minor_faults", 0)),
            major_faults=int(doc.get("major_faults", 0)),
            vol_ctx_switches=int(doc.get("vol_ctx_switches", 0)),
            invol_ctx_switches=int(doc.get("invol_ctx_switches", 0)),
            io_read_bytes=int(doc.get("io_read_bytes", 0)),
            io_write_bytes=int(doc.get("io_write_bytes", 0)),
        )

    def merged(self, other: QueryUsage) -> QueryUsage:
        """This usage plus `other` — how several workers' readings become one run's.

        Sums everything except the resident-set high-water, which is a *level* per process:
        summing peaks across workers would report memory no single machine ever held, so the
        largest is carried instead. Wall time sums too, which is deliberate — across W
        workers it is the run's total occupancy, and dividing `cpu_ms` by it still gives the
        mean cores busy per worker.

        Args:
            other: The reading to fold in.

        Returns:
            A new `QueryUsage`.
        """
        return QueryUsage(
            wall_ms=self.wall_ms + other.wall_ms,
            cpu_ms=self.cpu_ms + other.cpu_ms,
            peak_rss_bytes=max(self.peak_rss_bytes, other.peak_rss_bytes),
            minor_faults=self.minor_faults + other.minor_faults,
            major_faults=self.major_faults + other.major_faults,
            vol_ctx_switches=self.vol_ctx_switches + other.vol_ctx_switches,
            invol_ctx_switches=self.invol_ctx_switches + other.invol_ctx_switches,
            io_read_bytes=self.io_read_bytes + other.io_read_bytes,
            io_write_bytes=self.io_write_bytes + other.io_write_bytes,
        )


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
    #: What *this* operator does, in its own terms: the join type and keys, the group keys
    #: and aggregates, the sort keys, the filter predicate. Already rendered.
    #:
    #: The single most useful thing on a plan line after the operator's type, and it was
    #: missing from `explain()` entirely — "hash_join" alone has never told anyone which
    #: join it was, and a plan with four of them printed four identical lines. Every
    #: comparable engine prints it (Postgres's ``Hash Cond:``, Spark's ``[id#3 = id#7]``,
    #: DuckDB's key list), and the dashboard already showed it.
    #:
    #: Carried as finished text for the same reason `pushed` is: rendering an IR node lives
    #: in `observe`, which this layer may not import, so the caller that has both
    #: (`api.terminal.profile`) does the rendering and passes the result down.
    detail: str = ""
    #: What the plan handed to this scan's *source* to apply for itself — the pushed filter
    #: and the column projection, already rendered. Empty for every operator that is not a
    #: scan, and for a scan the plan pushed nothing to.
    #:
    #: Carried as finished text rather than as the predicate IR because rendering an
    #: expression lives in `observe`, which this layer may not import. The caller that has
    #: both (`api.terminal.profile`) does the rendering and passes the result down.
    pushed: str = ""
    # Measured (Core/engine); valid only when `measured`.
    measured: bool = False
    rows_in: int = 0
    rows_out: int = 0
    elapsed_ms: float = 0.0
    # CPU milliseconds summed across every worker thread this operator ran on — the
    # numerator `cpu_util` divides. Kept as its own field rather than left implicit in the
    # ratio because a ratio cannot be summed: totalling CPU across a query, or across a
    # fleet, needs the quantity and not its normalization. `0` when the platform cannot
    # report process CPU time, which by the engine's convention means unmeasured.
    cpu_ms: float = 0.0
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
    # --- Measured hardware consumption. 0 everywhere means unmeasured, not zero. ---
    # Page faults the operator took; the major count is the one that matters most, because a
    # nonzero value means the machine was paging against the query and every other field here
    # is describing the symptom rather than the cause.
    minor_faults: int = 0
    major_faults: int = 0
    # Context switches. The involuntary count measures contention for cores this process was
    # told it owned; the voluntary count marks genuine blocking on I/O or a lock.
    vol_ctx_switches: int = 0
    invol_ctx_switches: int = 0
    # Bytes that actually reached the block device, page-cache hits excluded. The difference
    # between a warm and a cold scan, which no other field distinguishes.
    io_read_bytes: int = 0
    io_write_bytes: int = 0

    @property
    def preemption_rate(self) -> float:
        """Involuntary context switches per core-second — how hard this op fought for cores.

        Above `CONTENDED_PREEMPTIONS_PER_CORE_SECOND` the operator spent a meaningful share of
        its life being evicted from the CPU, which means low utilization here is contention
        rather than a plan that failed to parallelize.
        """
        return preemption_rate(self.invol_ctx_switches, self.elapsed_ms, self.threads)

    @property
    def contended(self) -> bool:
        """Whether this operator was measurably competing for CPU with other work."""
        return self.preemption_rate > CONTENDED_PREEMPTIONS_PER_CORE_SECOND

    @property
    def paging(self) -> bool:
        """Whether the operator took disk-backed page faults — the box paging against it.

        Deliberately a bare "any at all" rather than a rate: on a machine with headroom a
        query's operators take essentially no major faults, so even a handful is a signal
        worth surfacing rather than noise to be thresholded away.
        """
        return self.major_faults > 0

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
            "detail": self.detail,
            "pushed": self.pushed,
            "measured": self.measured,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "elapsed_ms": self.elapsed_ms,
            "cpu_ms": self.cpu_ms,
            "result_bytes": self.result_bytes,
            "spilled": self.spilled,
            "spill_bytes": self.spill_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "backend": self.backend,
            "cpu_util": self.cpu_util,
            "threads": self.threads,
            "minor_faults": self.minor_faults,
            "major_faults": self.major_faults,
            "vol_ctx_switches": self.vol_ctx_switches,
            "invol_ctx_switches": self.invol_ctx_switches,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
            "preemption_rate": self.preemption_rate,
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
    # What the run cost the machine, measured across the execution as a whole rather than
    # per operator — the only such measurement that is sound on the streaming tier, which is
    # where most queries run. Summed across workers on a distributed run.
    usage: QueryUsage = field(default_factory=QueryUsage)

    @property
    def machine(self) -> str:
        """A short name for the machine class this profile was assembled on.

        Every timing here is relative to a machine, and a profile read out of a log or compared
        against one from another node is otherwise unattributed. It is also the key the
        engine's learned costs are stored under, so it answers the two questions a surprising
        plan raises: which machine shape ran this, and would what it learned apply anywhere
        else.

        On a **distributed** profile this is the *driver's* machine, which ran none of the
        operators — so `render` omits it there rather than inviting the timings above it to be
        read as facts about this box.
        """
        profile = hardware_profile()
        return f"{profile.label()} [{profile.fingerprint()}]"

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
        """Per-core CPU utilization for the whole run, in [0, 1].

        The single number for "did this run keep the cores busy?". Taken from the
        **query-level** `usage` measurement (`cores_busy / threads`) whenever the engine
        reported one, and only from the wall-time-weighted mean of the per-operator figures
        otherwise. `0.0` when nothing measured CPU time.

        Preferring `usage` is not a refinement, it is the difference between a number and a
        constant. The per-operator ratio is only sound on an executor where each operator
        owns an exclusive wall interval; the streaming executor — the default tier — reports
        `cpu_ns` as busy time summed over the units of work that ran the operator, and
        `wall_span_ns` as the interval they spanned. When an operator is *one* unit that
        parallelizes internally with rayon (every hash aggregate, join and sort), those two
        are the same quantity, and `cpu_ns / (wall_span_ns x threads)` collapses to exactly
        `1 / threads` — the very fabricated constant `stream::meter` set `cpu_ns = 0` to stop
        reporting. Measured here: a 16 M-row group-by on 15 cores, at a true 13.5 cores busy,
        printed `cpu utilization: 7% of cores ... CPU idle — not CPU-limited`, confidently
        backwards, while `usage.cores_busy` in the same document read 12.3.

        `QueryUsage` has no such attribution problem, because the attribution problem *is*
        the operator boundary: the OS reports process-wide totals and the delta across one
        execution belongs to that execution whatever ran inside it.
        """
        threads = max((o.threads for o in self.ops if o.threads > 0), default=0)
        if self.usage.measured and threads > 0:
            busy = self.usage.cores_busy
            if busy > 0.0:
                return min(1.0, busy / threads)
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
        # Share of *measured operator time*, not of `total_ms`. `total_ms` is the whole
        # terminal operation — planning, optimization, admission, the FFI crossing and
        # result assembly included — and the operators cover only the engine call inside
        # it. Against that denominator the dominant operator of a short query reported "2%
        # of wall time", which reads as "nothing here is the bottleneck" for the operator
        # that owned two thirds of the engine's work. The unaccounted remainder is a real
        # and often larger cost, and `render` now reports it on its own line rather than
        # by deflating every operator's share until none of them looks like the answer.
        ops_ms = sum(o.elapsed_ms for o in self.ops if o.measured)
        share = (b.elapsed_ms / ops_ms * 100.0) if ops_ms else 0.0
        # Classify by the *measured* per-core CPU busy fraction when it was recorded: a low
        # fraction is I/O- or launch-bound whatever the operator kind (a cached scan is not
        # I/O-bound; a stalled join can be memory-bound). Fall back to the kind heuristic when
        # CPU time was not measured (a sub-millisecond op, or an older engine).
        #
        # `cpu_utilization_overall()` rather than `b.cpu_util`, for the reason spelled out
        # there: on the streaming tier an operator that parallelizes internally reports a
        # per-op ratio pinned at `1 / threads`, so reading it here labelled every hash
        # aggregate, join and sort "I/O/launch-bound" — the diagnosis a reader is most likely
        # to act on, and the opposite of the truth. The bottleneck *dominates* the run by
        # construction (it is the longest operator), so the run-level figure describes it.
        util = self.cpu_utilization_overall()
        if util > 0:
            kind = "I/O/launch-bound" if util < 0.5 else f"compute-bound ({b.kind})"
        else:
            kind = "I/O-bound (read dominates)" if b.kind == "scan" else f"compute-bound ({b.kind})"
        spill = f" — SPILLED {human_bytes(self.total_spill_bytes)} to disk" if self.spilled else ""
        return f"bottleneck: {b.kind} (op {b.op_id}), {share:.0f}% of operator time — {kind}{spill}"

    def render(self, *, analyze: bool | None = None) -> str:
        """Render the plan as an operator tree with its summary sections.

        `analyze=True` shows the measured columns (actual rows, estimate miss, time,
        share, spill); `analyze=False` shows the planned estimate only. Defaults to
        whichever the profile carries (`measured`).

        Args:
            analyze: Force the measured or planned form; `None` follows `measured`.

        Returns:
            The rendered profile, in plain text.
        """
        from batcher.plan.profile.render import render_profile

        return render_profile(self, analyze=analyze)

    def __str__(self) -> str:
        return self.render()

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
            "machine": self.machine,
            "usage": self.usage.to_dict(),
            "ops": [o.to_dict() for o in self.ops],
            "worker_ops": [o.to_dict() for o in self.worker_ops],
            "decisions": [d.to_dict() for d in self.decisions],
            "adaptive_stages": list(self.adaptive_stages),
            "logical_ir": self.logical_ir,
            "optimized_ir": self.optimized_ir,
        }


def human_bytes(n: int) -> str:
    """Compact human-readable byte size (e.g. ``1.5 KiB``, ``3.4 MiB``).

    Delegates to `_internal.humanize.byte_size`, which is also what the web dashboard's
    ``UI.bytes`` renders, so the same spill volume reads identically in a terminal and in
    the browser. It did not before: this function divided by 1024 and labelled the result
    ``KB``, so every byte size in ``explain(analyze=True)`` was overstated by 2.4% per
    power of the unit and by 10% at the terabyte rung.

    Args:
        n: A byte count.

    Returns:
        The size with its binary unit, or an em dash when `n` is zero.
    """
    return byte_size(n)
