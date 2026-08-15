"""Learned top-N bounds: remember the k-th best value a top-N returned, and use it on the
next run of the same shape as a predicate that the scan can answer without decoding the rows
it excludes.

`ORDER BY x DESC LIMIT 10` over a wide table decodes every projected column of every row and
then discards all but ten. The thing that would make it cheap -- a value that separates the
ten from the rest -- is not knowable before the scan, so the engine does the only thing it
can: read everything, then select. `bc-runtime`'s `TopNBound` recovers part of this
*within* a query by publishing the running k-th best and skipping morsels whose min/max
exclude it, but that bound starts at infinity on every run and only tightens after rows have
already been read and decoded.

The k-th best value of a query, however, is one of the most stable things about it. A
leaderboard's tenth-place score, a log's slowest request, a store's highest price: these
move slowly relative to how often the query is asked. So it is worth remembering, and once
remembered the top-N starts with a bound instead of earning one -- which turns the query
into a highly selective filter that predicate pushdown, row-group zone maps and `bc-io`'s
late materialization all already know how to make cheap. Measured at **19.1x** on a 2.5 GB
20-column Parquet table (1,645 ms to 86 ms).

## Why a stale bound cannot return a wrong answer

The seeded plan filters to rows at or beyond the remembered bound. Every row it removes is
strictly worse than every row it keeps, so if `k` rows survive, those `k` *are* the true
global top-k -- no matter how wrong the bound was, and with no reference to the data it was
learned from. The bound is only a guess about **how many** rows survive, never about which.

That leaves exactly one failure mode: too few survivors, when the data has moved so that
fewer than `k` rows now clear the old bound. It is detected by counting the result, and the
caller answers it by re-running unseeded ([`TopNSeed.k`] is the threshold). So the cost of a
stale bound is one wasted cheap scan, and the cost of a fresh one is a 19x saving.

This is why the seeding lives here and the verification lives in `api`: Kyber decides what
the plan should be, but "run it, look at the answer, run something else" is orchestration,
and a plan -> plan pass cannot express it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batcher._internal.logging import note_suppressed
from batcher.metadata.hub import MetadataHub
from batcher.plan.expr_ir import Col, Expr, lit
from batcher.plan.logical import Filter, Limit, LogicalPlan, Sort

__all__ = ["TopNSeed", "record_topn_bound", "seed_topn_bound"]

_TOPN_NAMESPACE = "tuning.topn_kth"

# Above this many requested rows the seeding is not attempted.
#
# The saving comes from the bound being *selective*; a `LIMIT 1000000` bound sits far out in
# the distribution's tail and excludes almost nothing, so the added predicate is evaluated
# over the whole relation to remove very little. It also raises the chance of the stale-bound
# fallback, whose cost scales with how often it fires. Small limits are where top-N queries
# actually live -- a leaderboard, a "worst offenders" list, a sample.
_MAX_SEEDED_LIMIT = 10_000


@dataclass(frozen=True, slots=True)
class TopNSeed:
    """A plan rewritten to start from a remembered k-th best value.

    Attributes:
        plan: The seeded plan, to run in place of the original.
        k: Rows the top-N asks for. A result with at least this many rows is provably the
            true top-k; fewer means the bound was stale and the original plan must be run.
        signature: The shape key the bound was read under.
    """

    plan: LogicalPlan
    k: int
    signature: str


def _topn_shape(plan: LogicalPlan) -> tuple[Sort, int] | None:
    """The `Sort` and requested row count of a top-N, or `None` if this is not one.

    Both spellings reach here, because the seeding runs before Kyber has fused them:
    `Limit(Sort(...))` as written by `.sort().limit()`, and a `Sort` already carrying a
    `limit` from an earlier fusion.
    """
    if isinstance(plan, Limit) and isinstance(plan.input, Sort) and not plan.offset:
        return (plan.input, plan.n) if plan.input.limit is None else None
    if isinstance(plan, Sort) and plan.limit is not None:
        return plan, plan.limit
    return None


def _seedable_key(sort: Sort) -> tuple[str, bool] | None:
    """The leading sort key's column and direction, when a bound on it would be sound.

    Three conditions, each load-bearing:

    * the key is a plain column, so it can be named in a predicate the scan can push down;
    * it sorts nulls **last**, because a bound predicate drops nulls and a nulls-first
      ordering would have wanted them at the top -- and unlike a short result, that loss is
      invisible to the row count the caller checks;
    * there is at least one key.

    Only the *leading* key needs a bound. A removed row is strictly beyond the bound on that
    key, so it sorts after every kept row whatever the later keys say, and multi-key sorts
    are seeded exactly like single-key ones.
    """
    if not sort.keys:
        return None
    leading = sort.keys[0]
    if not isinstance(leading.expr, Col) or leading.nulls_first:
        return None
    return leading.expr.name, leading.descending


def _bound_key(sort: Sort, column: str, descending: bool, k: int) -> str:
    """The shape key a bound is stored under.

    Deliberately **not** `plan_signature` of the whole top-N. That signature does not
    separate a descending sort from its ascending twin, so the two would share a stored
    value — and the k-th *largest* is not a bound on the k-th *smallest*. The verification
    would catch it (a bad bound only ever costs a fallback), but the loop would then
    alternate between the two directions overwriting each other and never seed anything.

    `k` is part of the key for the same reason. A bound learned for `LIMIT 10` is too tight
    for `LIMIT 100`, which would fall back on every run.
    """
    import hashlib

    from batcher.kyber.signature import plan_signature

    payload = f"{plan_signature(sort.input)}|{column}|{int(descending)}|{k}"
    return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:16]


def _bound_is_comparable(node: LogicalPlan, column: str, bound: Any) -> bool:
    """Whether `column >= bound` is a comparison the engine can actually evaluate.

    The bound is keyed by `plan_signature`, which identifies the *relation* (its scan token
    carries `Scan.source_key`) but not its column **types** — the schema is deliberately not
    in the IR. A source keeps its key when its schema changes, so a path rewritten with a
    different type for the same column reads back the previous run's bound, and the seeded
    plan becomes `Filter(x >= 42)` over a string column:

        RuntimeError: Invalid argument error: Invalid comparison operation: Utf8 >= Int64

    That is a *raised query*, and it contradicts the guarantee this module is built on —
    "the cost of a stale bound is one wasted cheap scan". The verification in `api` cannot
    catch it either: it counts the survivors of a plan that never ran. Overwriting a Parquet
    path with a new schema is ordinary in a medallion pipeline, so this is reachable without
    anything exotic.

    Comparability is asked of `plan.types.promote` — the engine's own never-lossy type
    lattice — rather than by classifying types here, so this agrees with what the engine
    will do by construction: `None` means no common supertype exists (int against string,
    timestamp against int, date against string), and anything else is a comparison the
    engine performs. An int bound against a float column still seeds, which is the point of
    using the lattice rather than requiring equality.

    Unknown types answer **True**, preserving the previous behavior: `available_schema` is
    `None` for a plan whose type analysis is not certain, and a bound that cannot be checked
    is no worse off than it was before this guard existed.

    Args:
        node: The top-N's input, whose schema the predicate is evaluated against.
        column: The sort key the bound is on.
        bound: The remembered k-th best value.

    Returns:
        Whether the seeded predicate would type-check.
    """
    import pyarrow as pa

    from batcher.plan.types import infer_type, promote

    schema = node.available_schema()
    if schema is None:
        return True
    column_type = infer_type(Col(column), schema)
    if column_type is None:
        return True
    try:
        bound_type = pa.scalar(bound).type
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as exc:
        note_suppressed("kyber", "type the learned top-n bound", exc)
        return False
    return promote(column_type, bound_type) is not None


def _bound_predicate(column: str, descending: bool, value: Any) -> Expr:
    """`column >= value` for a descending top-N, `column <= value` for an ascending one.

    Inclusive on purpose. The remembered value *is* the k-th row, so excluding it would
    discard a row the answer needs and turn every accurate bound into a fallback.
    """
    from batcher.plan.expr_ir import col

    return col(column) >= lit(value) if descending else col(column) <= lit(value)


def seed_topn_bound(plan: LogicalPlan, hub: MetadataHub | None) -> TopNSeed | None:
    """Rewrite a top-N to start from its remembered k-th best value, if there is one.

    Args:
        plan: The plan as written, before optimization.
        hub: The metadata hub holding learned bounds, or `None`.

    Returns:
        The seeded plan and the row count that validates it, or `None` when this is not a
        seedable top-N or no bound has been learned for it.
    """
    if hub is None:
        return None
    try:
        shape = _topn_shape(plan)
        if shape is None:
            return None
        sort, k = shape
        if k <= 0 or k > _MAX_SEEDED_LIMIT:
            return None
        keyed = _seedable_key(sort)
        if keyed is None:
            return None
        column, descending = keyed

        signature = _bound_key(sort, column, descending, k)
        bound = hub.get_keyed_param(_TOPN_NAMESPACE, signature)
        if bound is None:
            return None
        if not _bound_is_comparable(sort.input, column, bound):
            return None

        guarded = Filter(sort.input, _bound_predicate(column, descending, bound))
        seeded_sort = Sort(guarded, sort.keys, sort.limit)
        seeded: LogicalPlan = (
            seeded_sort if plan is sort else Limit(seeded_sort, plan.n, plan.offset)
        )
        return TopNSeed(plan=seeded, k=k, signature=signature)
    except Exception as exc:  # pragma: no cover - a hint must never break a query
        note_suppressed("kyber", "seed top-n bound", exc)
        return None


def record_topn_bound(hub: MetadataHub | None, plan: LogicalPlan, table: Any) -> None:
    """Remember the k-th best value this top-N returned, for the next run of the shape.

    Recorded only from a **full** result (`num_rows == k`). A short one means the relation
    has fewer than `k` rows in total, and its worst row is not a bound on anything -- storing
    it would seed the next run with a value the whole relation clears.

    Args:
        hub: The metadata hub, or `None`.
        plan: The plan as written, keyed the same way [`seed_topn_bound`] reads it.
        table: The result, which must still carry the sort key column.
    """
    if hub is None:
        return
    try:
        shape = _topn_shape(plan)
        if shape is None:
            return
        sort, k = shape
        if k <= 0 or k > _MAX_SEEDED_LIMIT or table.num_rows != k:
            return
        keyed = _seedable_key(sort)
        if keyed is None:
            return
        column, descending = keyed
        if column not in table.column_names:
            # The sort key was projected away, so the result cannot say where the cut fell.
            return

        import pyarrow.compute as pc

        chunk = table.column(column)
        edge = pc.min(chunk) if descending else pc.max(chunk)
        value = edge.as_py()
        if value is None:  # an all-null top-k bounds nothing
            return

        hub.put_keyed_param(_TOPN_NAMESPACE, _bound_key(sort, column, descending, k), value)
    except Exception as exc:  # pragma: no cover - recording must never break a query
        note_suppressed("kyber", "record top-n bound", exc)
