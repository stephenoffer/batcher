"""`PhysicalPlan` — what Kyber emits and Core executes.

A physical plan is the relational IR (ready to ship to the engine) plus the
metadata Kyber attaches: per-operator resource bounds and cardinality/cost
estimates tagged with their *provenance* (how much to trust them). Carbonite and
the adaptive controller read provenance to decide how defensively to budget and
how eagerly to re-optimize when reality diverges.

The bootstrap `PhysicalPlan` carries the lowered IR document directly; a richer
per-operator `PhysicalOp` DAG is filled in as the optimizer and runtime grow.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from batcher.plan.ids import OpId
from batcher.plan.ir_tags import Op
from batcher.plan.resource import ResourceBounds
from batcher.plan.schema import SchemaRef
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["PhysicalOp", "PhysicalPlan", "PlanProperties", "Provenance"]


@dataclass(frozen=True, slots=True)
class PlanProperties:
    """Estimated properties of an operator's output.

    `provenance` is the unified trust scale from `plan.stats`; `column_stats`
    carries the per-column statistics the estimator propagated, so Carbonite and
    the adaptive controller see not just *how many* rows but *what* the columns
    look like and how much to trust it.

    `signature` and `est_rows_raw` close the cardinality feedback loop. `signature` is
    the operator's structural plan signature — a *stable* identity across executions,
    unlike `op_id`, which is only a position in this plan's walk. `est_rows_raw` is the
    raw **structural** estimate (before any learned correction), which Core reports back
    so Kyber can measure how wrong its own row formulas are (`actual / est_rows_raw`).
    It is `0.0` for operators whose estimate came from a past measurement or a proof
    rather than from the structural estimator — their q-error is ~1.0 by construction and
    averaging it in would decay a learned correction back to 1.0.

    `expr_factor` is the per-row cost of the expressions this operator evaluates, relative
    to a plain comparison (1.0 for operators that evaluate none). Core echoes it back so
    `calibration` can divide it out of the measured wall time: without that, a workload of
    regex filters would fit a huge `filter_row` coefficient that then gets multiplied by
    the regex's factor *again* at planning time.
    """

    est_rows: float = float("nan")
    #: Byte-true bytes per output row, as `annotate_ops` sized this operator with
    #: (`estimator.row_width`: learned per-column widths when measured, else the flat
    #: `optimizer.row_bytes`). Published because `m_max_bytes == est_rows * row_size` is a
    #: *lossy* product: a consumer that needs one factor back cannot divide by the flat
    #: default without being wrong by `row_size / row_bytes` — an order of magnitude on the
    #: wide payloads (blobs, embeddings) the learned width exists to model. Carbonite's spill
    #: sizing did exactly that. NaN when the operator was not sized.
    row_size: float = float("nan")
    confidence: float = 0.0
    provenance: Provenance = Provenance.DEFAULT
    column_stats: Mapping[str, ColumnStat] = field(default_factory=dict)
    signature: str = ""
    est_rows_raw: float = float("nan")
    expr_factor: float = 1.0


@dataclass(frozen=True, slots=True)
class PhysicalOp:
    """A physical operator: a relational op bound to a backend/algorithm + bounds."""

    op_id: OpId
    kind: str
    backend: str
    algorithm: str
    bounds: ResourceBounds
    inputs: tuple[OpId, ...]
    properties: PlanProperties = PlanProperties()
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    """An executable plan: the lowered IR plus Kyber's annotations."""

    ir: dict[str, Any]
    output_schema: SchemaRef | None
    ops: tuple[PhysicalOp, ...] = ()
    # Per scan `source_id`, the column projection to read (projection pushdown).
    # Empty/absent means "read all columns".
    source_projections: dict[int, list[str]] = field(default_factory=dict)
    # Per scan `source_id`, the predicate IR of a `Filter` directly above it
    # (predicate pushdown). A pushdown-capable source translates the pushable
    # subset to its backend filter to skip I/O; the engine keeps the `Filter`
    # operator as a safe re-check, so an absent/partial translation is correct.
    source_predicates: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Per scan `source_id`, the most rows that source ever has to produce — a `Limit`
    # reaching the scan through nothing but projections (row-cap pushdown). A source may
    # stop early; one that ignores this reads what it always did, because the engine keeps
    # its own `Limit`. Absent means "no cap", which is also what one unbounded scan of a
    # source that another scan limits must produce. See
    # `kyber.rules.source_limits.required_limits_per_source`.
    source_limits: dict[int, int] = field(default_factory=dict)
    # Per scan `source_id`, the ordering its row cap is taken *in* — one
    # ``(column, descending, nulls_first)`` per sort key. Only ever set alongside a
    # `source_limits` entry: ordering a source without capping it buys nothing and makes
    # the server sort rows it would have streamed. A source that cannot express the
    # ordering must also ignore the cap, since "the first n" is meaningless without it.
    source_orderings: dict[int, tuple[tuple[str, bool, bool], ...]] = field(default_factory=dict)
    #: Kyber's verdict that this plan's grouped aggregate is cheaper materialized than
    #: streamed. Set from the estimated group count, which only the control plane has; the
    #: engine pairs it with its own memory-affordability check. See
    #: `EngineConfig.prefer_materializing_aggregate` for the measurements behind the
    #: threshold, and `MATERIALIZE_AGG_MIN_GROUPS` for the threshold itself.
    prefer_materializing_aggregate: bool = False
    #: One-slot memo for `to_json`. A list rather than a plain string because the dataclass
    #: is frozen: appending to a mutable default needs no `object.__setattr__` escape, and
    #: `compare=False` keeps the memo out of plan equality. See `to_json` for why it exists.
    _json_memo: list[str] = field(default_factory=list, init=False, repr=False, compare=False)

    def to_json(self) -> str:
        """Serialize the relational IR for the engine, memoized per plan instance.

        Kyber's plan cache hands the *same* `PhysicalPlan` object back for a re-issued
        query, so without the memo a repeated `collect()` re-serializes an identical IR on
        every execution. That is 7 us of a small query's ~360 us, and it buys nothing: the
        plan is frozen, so its IR cannot have changed since the last call.
        """
        memo = self._json_memo
        if not memo:
            memo.append(json.dumps(self.ir))
        return memo[0]

    def scanned_source_ids(self) -> frozenset[int]:
        """The `source_id`s this plan's IR actually reads.

        Sources cross the FFI boundary positionally, so a caller keeps every source at its
        own index while touching only the ones the plan reaches. `plan.visitor` answers the
        same question for a `LogicalPlan`; this answers it for the lowered IR, which is what
        the executor is handed and the only form some callers hold.

        It is load-bearing rather than an optimization. The adaptive staging loop
        (`api.adaptive.staging`) executes a *sub*-plan against the query's full source list,
        so a stage that scans two tables was handed all six — and reading a source the stage
        never scans makes it resident, with no projection to narrow it (the pushdown analysis
        records nothing for a scan the plan does not contain, and an absent entry means "read
        every column"). On TPC-H q9 at sf10 the first stage read 8.9 GiB where its plan needed
        2.9 GiB, and the query was OOM-killed.

        The walk is structural — any nested `{"op": "scan", "source_id": n}` — so an IR shape
        this module does not know about cannot hide a scan from it.

        Returns:
            Every `source_id` reachable in the IR document. Empty for a plan that scans
            nothing.
        """
        found: set[int] = set()
        _collect_scans(self.ir, found)
        return frozenset(found)

    def op_budgets(self) -> dict[int, int]:
        """Per-operator spill budgets (bytes) keyed by pre-order `op_id`.

        Kyber sizes each stateful operator's peak memory envelope
        (`ResourceBounds.m_max_bytes`); this surfaces those bounds as the side map
        Core ships to the engine so the data plane budgets each operator
        individually instead of applying one global `memory_budget_bytes` to every
        operator. Only positively-sized operators are included — an absent entry
        means "fall back to the global budget", which is exactly the behaviour for
        unsized (streaming/unknown) operators that Kyber leaves at `m_max_bytes=0`.
        """
        return {
            int(op.op_id): op.bounds.m_max_bytes for op in self.ops if op.bounds.m_max_bytes > 0
        }


def _collect_scans(node: Any, found: set[int]) -> None:
    """Add every `{"op": "scan", "source_id": n}` reachable in `node` to `found`.

    Structural rather than schema-driven: it descends into every dict value and list
    element, so an IR shape that nests a scan under a key this module has never heard of
    still reports it. Under-reporting is the failure that matters — a caller skips a source
    the plan does read — so the walk is deliberately indiscriminate.
    """
    if isinstance(node, dict):
        if node.get("op") == Op.SCAN:
            source_id = node.get("source_id")
            if isinstance(source_id, int):
                found.add(source_id)
        for value in node.values():
            _collect_scans(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_scans(value, found)
