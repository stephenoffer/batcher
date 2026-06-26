"""Profile assembly — join Kyber's estimates to Core's measurements by `op_id`.

The tree structure and `op_id` ordering come from a pre-order walk of the lowered IR
(`PhysicalPlan.ir`), which both the optimizer (`_annotate_ops`) and the engine (`IdGen`)
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

    def to_profile(self, *, total_ms: float, rows: int, query_id: str = "") -> QueryProfile:
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
                # The engine's wire key is `peak_bytes`, but it measures result-array size
                # (see `OpProfile.result_bytes`); relabel it honestly on the way in.
                result_bytes=int(m.get("peak_bytes", 0)),
                spilled=bool(m.get("spilled", False)),
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
    `op_id`. Rows and time sum across workers, result/peak bytes takes the worst single
    worker, spill is sticky. Returns a list shaped like a single run's `ExecMetrics`.
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
                    "rows_out": int(m.get("rows_out", 0)),
                    "elapsed_ns": int(m.get("elapsed_ns", 0)),
                    "cpu_ns": int(m.get("cpu_ns", 0)),
                    "peak_bytes": int(m.get("peak_bytes", 0)),
                    "threads": int(m.get("threads", 0)),
                    "spilled": bool(m.get("spilled", False)),
                    "backend": m.get("backend", ""),
                }
                continue
            cur["rows_in"] += int(m.get("rows_in", 0))
            cur["rows_out"] += int(m.get("rows_out", 0))
            cur["elapsed_ns"] += int(m.get("elapsed_ns", 0))
            cur["cpu_ns"] += int(m.get("cpu_ns", 0))
            cur["peak_bytes"] = max(cur["peak_bytes"], int(m.get("peak_bytes", 0)))
            cur["threads"] = max(cur["threads"], int(m.get("threads", 0)))
            cur["spilled"] = cur["spilled"] or bool(m.get("spilled", False))
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
                result_bytes=int(m.get("peak_bytes", 0)),
                spilled=bool(m.get("spilled", False)),
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
    numbering used by both `_annotate_ops` and the engine's `IdGen`.
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
