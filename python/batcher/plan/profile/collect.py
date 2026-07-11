"""Profile assembly — join Kyber's estimates to Core's measurements by `op_id`.

The tree structure and `op_id` ordering come from a pre-order walk of the lowered IR
(`PhysicalPlan.ir`), which both the optimizer (`annotate_ops`) and the engine (`IdGen`)
number identically. `ProfileCollector` is the mutable sink the orchestrator fills during
one terminal run; `build_op_profiles`/`merge_metric_ops` are the pure joiners.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from batcher.plan.feedback import cpu_utilization
from batcher.plan.physical import PhysicalOp
from batcher.plan.profile.types import Decision, OpProfile, QueryProfile

__all__ = ["ProfileCollector", "build_op_profiles", "merge_metric_ops", "worker_op_profiles"]


@dataclass
class ProfileCollector:
    """A mutable sink the orchestrator fills during one terminal run; `api` reads it back.

    Handed down via `ExecutionContext.profile`. Each Kyber→Carbonite→Core hand-off
    records into it (the optimized IR + per-operator estimates, the admission verdict,
    the measured `ExecMetrics`, any spill/distributed path taken). Only the conductor
    reads it — subsystems append, never read — so it carries the whole picture without
    any subsystem importing another. `to_profile` joins it into a `QueryProfile`.
    """

    optimized_ir: dict[str, Any] | None = None
    logical_ir: dict[str, Any] | None = None
    physical_ops: tuple[PhysicalOp, ...] = ()
    metric_ops: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    carbonite_summary: str = ""
    adaptive_stages: list[dict[str, Any]] = field(default_factory=list)
    distributed: bool = False
    # Raw `ExecMetrics` op-lists shipped back by distributed workers (the map sub-plan),
    # one list per worker. Merged into `QueryProfile.worker_ops` by `to_profile`.
    worker_metrics: list[list[dict[str, Any]]] = field(default_factory=list)

    def to_profile(
        self, *, total_ms: float, rows: int, query_id: str = "", memory_budget_bytes: int = 0
    ) -> QueryProfile:
        """Assemble the collected planned + measured facts into a `QueryProfile`."""
        ir = self.optimized_ir or {}
        ops = build_op_profiles(ir, self.physical_ops, self.metric_ops or None)
        worker_ops = (
            worker_op_profiles(merge_metric_ops(self.worker_metrics)) if self.worker_metrics else ()
        )
        return QueryProfile(
            ops=ops,
            total_ms=total_ms,
            rows=rows,
            query_id=query_id,
            measured=bool(self.metric_ops) or bool(worker_ops),
            distributed=self.distributed,
            decisions=tuple(self.decisions),
            carbonite_summary=self.carbonite_summary,
            adaptive_stages=tuple(self.adaptive_stages),
            logical_ir=self.logical_ir,
            optimized_ir=self.optimized_ir,
            memory_budget_bytes=memory_budget_bytes,
            worker_ops=worker_ops,
        )


def build_op_profiles(
    ir: Mapping[str, Any],
    physical_ops: Sequence[PhysicalOp] = (),
    metric_ops: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[OpProfile, ...]:
    """Join the planned `PhysicalOp`s and measured `ExecMetrics` dicts by `op_id`.

    `physical_ops` supply the estimate (indexed by `op_id`); `metric_ops` (raw
    `ExecMetrics`, or `None` for a planned-only profile) supply the measurement. An
    operator missing from either side keeps its defaults.
    """
    planned = {int(op.op_id): op for op in physical_ops}
    measured = {int(m.get("op_id", -1)): m for m in (metric_ops or [])}
    out: list[OpProfile] = []
    for op_id, (depth, node) in enumerate(_walk_ir(ir)):
        kind = str(node.get("op", "?"))
        p = planned.get(op_id)
        est_rows = float(p.properties.est_rows) if p is not None else float("nan")
        provenance = str(p.properties.provenance) if p is not None else ""
        algorithm = p.algorithm if p is not None and p.algorithm else ""
        m = measured.get(op_id)
        if m is None:
            out.append(
                OpProfile(
                    op_id=op_id,
                    kind=kind,
                    depth=depth,
                    est_rows=est_rows,
                    provenance=provenance,
                    algorithm=algorithm,
                )
            )
            continue
        out.append(
            OpProfile(
                op_id=op_id,
                kind=str(m.get("kind", kind)),
                depth=depth,
                est_rows=est_rows,
                provenance=provenance,
                algorithm=algorithm or ("spill" if m.get("spilled") else ""),
                measured=True,
                rows_in=int(m.get("rows_in", 0)),
                rows_out=int(m.get("rows_out", 0)),
                elapsed_ms=float(m.get("elapsed_ns", 0)) / 1e6,
                # The engine now reports both: `result_bytes` (output arrays) and
                # `peak_bytes` (true peak working set). Older engines only had the latter,
                # holding output size, so fall back to it.
                result_bytes=int(m.get("result_bytes", m.get("peak_bytes", 0))),
                spilled=bool(m.get("spilled", False)),
                spill_bytes=int(m.get("spill_bytes", 0)),
                peak_rss_bytes=int(m.get("peak_rss_bytes", 0)),
                backend=str(m.get("backend", "")),
                cpu_util=cpu_utilization(
                    m.get("cpu_ns", 0), m.get("elapsed_ns", 0), m.get("threads", 1)
                ),
                threads=int(m.get("threads", 0)),
            )
        )
    return tuple(out)


def merge_metric_ops(per_worker: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Combine many workers' `ExecMetrics` op lists into one, summed by `op_id`.

    Distributed workers run the *same* sub-plan, so their per-operator metrics share an
    `op_id`. Each field merges by what it physically is:

    * rows, result bytes, CPU time and thread count **sum** — each worker did its own share;
    * wall time takes the **max** — the workers ran concurrently, so the stage lasted as
      long as its slowest one. Summing it reported a W-worker stage as W times slower;
    * peak bytes takes the **max**, i.e. the worst single worker (a lower bound on
      cluster-wide memory — never route it to sizing);
    * spill is sticky.

    Returns a list shaped like a single run's `ExecMetrics`.
    """
    acc: dict[int, dict[str, Any]] = {}
    for ops in per_worker:
        for m in ops:
            op_id = int(m.get("op_id", -1))
            cur = acc.get(op_id)
            if cur is None:
                acc[op_id] = {
                    "op_id": op_id,
                    "kind": m.get("kind", ""),
                    "rows_in": int(m.get("rows_in", 0)),
                    "rows_build": int(m.get("rows_build", 0)),
                    "rows_out": int(m.get("rows_out", 0)),
                    "elapsed_ns": int(m.get("elapsed_ns", 0)),
                    "cpu_ns": int(m.get("cpu_ns", 0)),
                    "peak_bytes": int(m.get("peak_bytes", 0)),
                    "result_bytes": int(m.get("result_bytes", m.get("peak_bytes", 0))),
                    "threads": int(m.get("threads", 0)),
                    "spilled": bool(m.get("spilled", False)),
                    "spill_bytes": int(m.get("spill_bytes", 0)),
                    "peak_rss_bytes": int(m.get("peak_rss_bytes", 0)),
                    "backend": m.get("backend", ""),
                }
                continue
            # Rows and CPU-time are additive across workers: each processed its own share.
            cur["rows_in"] += int(m.get("rows_in", 0))
            cur["rows_build"] += int(m.get("rows_build", 0))
            cur["rows_out"] += int(m.get("rows_out", 0))
            cur["cpu_ns"] += int(m.get("cpu_ns", 0))
            cur["result_bytes"] += int(m.get("result_bytes", m.get("peak_bytes", 0)))
            # Wall time is NOT: the workers ran *concurrently*, so the stage took as long as
            # its slowest one. Summing it reported a W-worker stage as W times slower than
            # it was, and is what `worker_op_profiles` renders as `elapsed_ms`.
            cur["elapsed_ns"] = max(cur["elapsed_ns"], int(m.get("elapsed_ns", 0)))
            # The worst single worker's peak — every worker holds its own concurrently, so
            # this is a lower bound on cluster-wide memory. Never route it to sizing.
            cur["peak_bytes"] = max(cur["peak_bytes"], int(m.get("peak_bytes", 0)))
            # Threads add up: `cpu_ns / (elapsed_ns x threads)` is then the cluster-wide
            # per-core utilization, which is only coherent if the denominator counts every
            # core that contributed the summed CPU time.
            cur["threads"] += int(m.get("threads", 0))
            cur["spilled"] = cur["spilled"] or bool(m.get("spilled", False))
            # Spill volume is additive: each worker spilled its own share, so the cluster-wide
            # spill total is their sum (unlike peak bytes, which is a concurrent max).
            cur["spill_bytes"] += int(m.get("spill_bytes", 0))
            # Peak RSS is a high-water, not additive — take the worst single worker (a lower
            # bound on cluster-wide RSS, same convention as peak_bytes).
            cur["peak_rss_bytes"] = max(cur["peak_rss_bytes"], int(m.get("peak_rss_bytes", 0)))
    return [acc[k] for k in sorted(acc)]


def worker_op_profiles(merged: Sequence[Mapping[str, Any]]) -> tuple[OpProfile, ...]:
    """`OpProfile`s for the distributed map sub-plan (flat, measured-only, no estimate).

    These are the workers' own sub-plan operators — a *separate* op-id space from the
    driver tree, so they are kept apart (not joined into the planned tree) and rendered
    as their own section. `depth=0`; planned fields stay empty.
    """
    out: list[OpProfile] = []
    for m in merged:
        out.append(
            OpProfile(
                op_id=int(m.get("op_id", 0)),
                kind=str(m.get("kind", "?")),
                depth=0,
                measured=True,
                rows_in=int(m.get("rows_in", 0)),
                rows_out=int(m.get("rows_out", 0)),
                elapsed_ms=float(m.get("elapsed_ns", 0)) / 1e6,
                result_bytes=int(m.get("result_bytes", m.get("peak_bytes", 0))),
                spilled=bool(m.get("spilled", False)),
                spill_bytes=int(m.get("spill_bytes", 0)),
                peak_rss_bytes=int(m.get("peak_rss_bytes", 0)),
                backend=str(m.get("backend", "")),
                cpu_util=cpu_utilization(
                    m.get("cpu_ns", 0), m.get("elapsed_ns", 0), m.get("threads", 1)
                ),
                threads=int(m.get("threads", 0)),
            )
        )
    return tuple(out)


def _is_plan_node(value: Any) -> bool:
    """Whether an IR value is a relational plan node (not an expression).

    A relational node carries the ``"op"`` tag; an expression carries the ``"e"`` tag —
    and a *binary* expression carries **both** (``{"e": "binary", "op": "gt", ...}``). So
    a plan node is exactly ``"op"`` present *and* ``"e"`` absent. Getting this wrong would
    walk a predicate as a plan child and shift every later operator's `op_id`.
    """
    return isinstance(value, Mapping) and "op" in value and "e" not in value


def _walk_ir(ir: Mapping[str, Any], depth: int = 0) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield ``(depth, op_dict)`` pre-order over the relational IR tree.

    A child plan is any nested value that `_is_plan_node` accepts (a list contributes
    each such element); predicates/projections are skipped. Pre-order matches the `op_id`
    numbering used by both `annotate_ops` and the engine's `IdGen`.
    """
    if not _is_plan_node(ir):
        return
    yield depth, ir
    for value in ir.values():
        if _is_plan_node(value):
            yield from _walk_ir(value, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if _is_plan_node(item):
                    yield from _walk_ir(item, depth + 1)
