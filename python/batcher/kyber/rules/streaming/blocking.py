"""Blocking-operator avoidance under a stream.

A *blocking* operator cannot emit its first row until it has seen its last input row
(`kyber.streaming.is_blocking_under_stream`). Over a bounded source that is merely a
pipeline breaker — it costs memory and latency, and then it finishes. Over an unbounded
source there is no last row, so the operator never emits *anything*: the query does not
run slowly, it produces nothing at all, forever.

Every rule here removes a blocking operator that is provably unnecessary, and every one
is gated on `has_unbounded_input`. That gate is not a performance heuristic, it is a
scope limit: these rewrites are all *order-discarding*, and the ordering they discard is
unobservable only because a blocking operator downstream (a dedup, a distinct union) or
a structural row bound already made it so. Restricting them to unbounded plans keeps
them off the bounded paths where the existing stats-gated rules
(`skip_sort_of_single_row`, `remove_redundant_distinct`, `drop_distinct_when_unique`)
already do this work with an exact row count in hand.

**Why stats-gated rules are not enough here, and why this family exists.** Every
existing Sort/Distinct-elimination rule requires `Provenance.EXACT` metadata — an exact
row count, an exact ndv. A stream has none of that and never will: an unbounded source
cannot report how many rows it has. So on exactly the plans where a redundant blocking
operator is fatal rather than merely wasteful, none of those rules can fire. The proofs
below are therefore **structural**: they read the shape of the plan and nothing else, so
they hold with no cardinality estimate at all.

**What is deliberately absent.** There is no `Distinct` → `WatermarkDedup` rewrite.
`WatermarkDedup` forgets a key once the watermark passes it, so a duplicate arriving
later than the allowed lateness is re-emitted — a strictly different result from
`Distinct`, which is exact over all time. Converting one to the other trades a hang for
a wrong answer, and Kyber's contract is semantics-preserving rewrites, so the refusal is
total rather than conditional. (It could not be conditional in any case: a watermark is
carried on `Dataset` and only lands in the plan on an `Aggregate`, so a rule matching a
`Distinct` cannot see whether one is available.) Turning a stream `distinct()` into a
watermark dedup is a *user* decision about correctness, and belongs in the API.

Also absent, for the same order-sensitivity reason: nothing here strips a `Sort` from
beneath a `WatermarkDedup` or a `WatermarkStreamJoin`. Both keep the *first* qualifying
row and both advance a watermark from *arrival* order, so reordering their input changes
which rows survive and which are dropped as late. `push_filter_through_watermark_dedup`
in `watermark.py` documents the same hazard from the other direction.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.kyber.streaming import has_unbounded_input
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Sample,
    Sort,
    Union,
)

__all__ = [
    "stream_drop_distinct_over_at_most_one_row",
    "stream_drop_keyless_sort",
    "stream_drop_sort_in_distinct_union_branch",
    "stream_drop_sort_over_at_most_one_row",
    "stream_drop_sort_under_distinct",
]


def _at_most_one_row(node: LogicalPlan) -> bool:
    """Whether `node` provably yields at most one row, from its shape alone.

    Structural, so it holds over a stream where no exact row count exists:

    - `Limit(n <= 1)` yields at most `n` rows by definition, whatever the offset.
    - a top-N `Sort(limit <= 1)` likewise caps its own output.
    - a fixed-count `Sample(n <= 1)` keeps exactly the `n` smallest-hash rows.
    - a **keyless** `Aggregate` has exactly one output row (the global aggregate); a
      *grouped* one has a row per group and is excluded.

    Nothing here consults `ctx.estimator`: an estimate could be wrong, and on a stream
    it is always the unknown-rows sentinel anyway.

    Args:
        node: The plan node to bound.

    Returns:
        True when `node`'s output cannot contain two rows.
    """
    if isinstance(node, Limit):
        return node.n <= 1
    if isinstance(node, Sort):
        return node.limit is not None and node.limit <= 1
    if isinstance(node, Sample):
        return node.n is not None and node.n <= 1
    if isinstance(node, Aggregate):
        return not node.group_keys
    return False


def _strip_full_sort(node: LogicalPlan) -> LogicalPlan | None:
    """Remove a full `Sort` from `node`, descending through order-transparent operators.

    Returns the rewritten subtree, or None when there is no such `Sort` to remove.

    "Order-transparent" here means an operator whose **output multiset is a function of
    its input multiset alone** — it neither reads a row's position nor lets one row's
    fate depend on another's. `Project` and `Filter` are exactly that: both are row-wise
    maps, so reordering their input reorders their output and changes nothing else. That
    is what lets a caller who has already proven the ordering unobservable at *its* level
    reach past them to the sort.

    Everything else stops the descent, and the exclusions carry the weight of the proof.
    `Limit` and `Sample(n=…)` select a *positional* or *relative* subset, so which rows
    they keep depends on the order they receive. `Distinct`, `Aggregate`, and `Union`
    combine rows across the stream. `WatermarkDedup` and `WatermarkStreamJoin` keep the
    first row per key and advance a watermark from arrival order. `MapBatches` is an
    opaque Python callable that may read batch order. None of those may be crossed, so
    the default is to stop.

    Only a **full** sort (`limit is None`) is removed: a top-N is not an ordering, it is
    a row filter, and dropping it would change which rows exist rather than which order
    they arrive in. A top-N is also not blocking under a stream, so there is nothing to
    fix.

    Args:
        node: The subtree to search, rooted at the operator below the caller's node.

    Returns:
        The subtree with the sort removed, or None if no full sort was reachable.
    """
    if isinstance(node, Sort):
        return node.input if node.limit is None else None
    if isinstance(node, Project):
        stripped = _strip_full_sort(node.input)
        return None if stripped is None else Project(stripped, node.items)
    if isinstance(node, Filter):
        stripped = _strip_full_sort(node.input)
        return None if stripped is None else Filter(stripped, node.predicate)
    return None


@rule(
    name="stream_drop_keyless_sort",
    phase=Phase.REWRITE,
    matches=(Sort,),
    category=RuleCategory.REWRITE,
)
def stream_drop_keyless_sort(node: Sort, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a `Sort` with no sort keys over a stream — it orders nothing, and it hangs.

    A sort with an empty key tuple imposes no ordering constraint whatsoever: every pair
    of rows compares equal, so the operator is an identity on its input multiset. In
    batch it is a wasted materialization. Over an unbounded input it is worse than
    wasted — it is blocking (`is_blocking_under_stream` classifies a `limit`-less `Sort`
    as such), so it buffers the stream forever and emits nothing. Removing it is exactly
    semantics-preserving and converts a query that could never produce a row into one
    that streams.

    Keyless sorts are not hand-written; they are *produced*, by
    `prune_constant_sort_keys` reducing an all-constant key list and by planner-generated
    orderings whose keys are later eliminated. That path is stats-gated and drops the
    sort itself only when it can prove every key constant, which on a stream it cannot —
    so the degenerate keyless residue is exactly what reaches here.

    A keyless *top-N* (`limit is not None`) is refused. With no keys to compare, which
    rows a partial sort surfaces depends on its stability, which is not a guarantee the
    engine makes; rewriting it to a `Limit` would be asserting one. It is also not
    blocking, so there is no hang to fix.

    Args:
        node: The `Sort` under consideration.
        ctx: The optimizer context, read for the bound sources' boundedness.

    Returns:
        The sort's input when the sort is a keyless full sort over a stream, else None.
    """
    if node.keys or node.limit is not None:
        return None
    if not has_unbounded_input(node, ctx):
        return None
    return node.input


@rule(
    name="stream_drop_sort_under_distinct",
    phase=Phase.REWRITE,
    matches=(Distinct,),
    category=RuleCategory.REWRITE,
)
def stream_drop_sort_under_distinct(node: Distinct, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a full `Sort` beneath a `Distinct` over a stream — the dedup discards its order.

    `Distinct` is a set operation. Its result is the set of distinct rows of its input,
    which is a function of the input *multiset* and not of the order in which the rows
    arrive; and its own output order is unspecified in both the original plan and the
    rewritten one, so no observable property of the query changes. This is the same
    argument `eliminate_sort_before_aggregate` makes for `Aggregate`, applied to the
    other order-destroying breaker.

    Under a stream the rewrite is worth far more than the sort it saves. Both operators
    are blocking, so the original plan can never emit; afterwards only the `Distinct`
    remains, and one blocking operator is what a watermark-bounded dedup can replace at
    the API level. Two nested ones cannot be replaced by anything.

    The search descends through `Project` and `Filter` (see `_strip_full_sort`), which
    are row-wise and therefore order-transparent, so `DISTINCT` over a projected,
    filtered, sorted stream is reached too. A top-N sort is left alone: it selects rows
    rather than ordering them, so it is load-bearing.

    Args:
        node: The `Distinct` under consideration.
        ctx: The optimizer context, read for the bound sources' boundedness.

    Returns:
        The distinct over the de-sorted input, or None when nothing was removed.
    """
    if not has_unbounded_input(node, ctx):
        return None
    stripped = _strip_full_sort(node.input)
    if stripped is None:
        return None
    # `replace`, not `Distinct(stripped)` — the positional rebuild took one of the node's four
    # fields, so a streaming `DISTINCT ON (k) ORDER BY ...` came back as a **whole-row**
    # `DISTINCT`: one row per key became one row per distinct row, which is a wrong answer and
    # not a slower one. Same three-of-five rebuild as `projections.py::_rewrite`, on a node with
    # four fields instead of five.
    return dataclasses.replace(node, input=stripped)


@rule(
    name="stream_drop_sort_in_distinct_union_branch",
    phase=Phase.REWRITE,
    matches=(Union,),
    category=RuleCategory.REWRITE,
)
def stream_drop_sort_in_distinct_union_branch(
    node: Union, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop a full `Sort` from a branch of a `UNION` (distinct) over a stream.

    A `Union` with `distinct=True` deduplicates the concatenation of its branches, so —
    exactly as for `Distinct` above — its result is a set determined by the branches'
    multisets, and the order within any one branch is not observable in it. A blocking
    sort in a branch is therefore removable, and it must be: a `UNION` cannot emit while
    any branch is still buffering, so one sorted branch stalls the whole operator.

    Restricted to `distinct=True`. Under `UNION ALL` the engine concatenates branch
    output, so a branch's internal ordering *does* survive into the result and dropping
    the sort would be an observable change. That distinction is the entire gate, and it
    is why this is a separate rule rather than a case of the `Distinct` one.

    Branches are rewritten independently and any number may change; the rule returns None
    when no branch had a removable sort, which keeps it idempotent under the REWRITE
    fixpoint.

    Args:
        node: The `Union` under consideration.
        ctx: The optimizer context, read for the bound sources' boundedness.

    Returns:
        The union with de-sorted branches, or None when no branch changed.
    """
    if not node.distinct:
        return None
    if not has_unbounded_input(node, ctx):
        return None
    branches = []
    changed = False
    for branch in node.inputs:
        stripped = _strip_full_sort(branch)
        if stripped is None:
            branches.append(branch)
        else:
            branches.append(stripped)
            changed = True
    if not changed:
        return None
    return Union(tuple(branches), distinct=True)


@rule(
    name="stream_drop_distinct_over_at_most_one_row",
    phase=Phase.REWRITE,
    matches=(Distinct,),
    category=RuleCategory.REWRITE,
)
def stream_drop_distinct_over_at_most_one_row(
    node: Distinct, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop a `Distinct` whose input provably holds at most one row.

    A relation of zero or one rows cannot contain a duplicate pair, so deduplicating it
    returns it unchanged — the rewrite is an identity on the result and removes a
    blocking operator from the plan.

    The bound is **structural** (`_at_most_one_row`): a `Limit(n <= 1)`, a top-1 `Sort`, a
    `Sample(n <= 1)`, or a keyless global `Aggregate`. That matters because the existing
    `remove_redundant_distinct` and `drop_distinct_when_unique` prove the same fact from
    `Provenance.EXACT` cardinality metadata, which an unbounded source can never supply.
    On a stream those rules are structurally unable to fire; this one reads only the plan
    shape, so a `distinct()` over a bounded prefix of a stream is recognized.

    The rewrite is not gated on the *whole* plan being blocked, only on it being
    unbounded: the input bound is what makes the `Distinct` redundant, and the stream is
    what makes removing it matter.

    Args:
        node: The `Distinct` under consideration.
        ctx: The optimizer context, read for the bound sources' boundedness.

    Returns:
        The distinct's input when that input holds at most one row, else None.
    """
    if not _at_most_one_row(node.input):
        return None
    if not has_unbounded_input(node, ctx):
        return None
    return node.input


@rule(
    name="stream_drop_sort_over_at_most_one_row",
    phase=Phase.REWRITE,
    matches=(Sort,),
    category=RuleCategory.REWRITE,
)
def stream_drop_sort_over_at_most_one_row(node: Sort, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a `Sort` whose input provably holds at most one row.

    A relation of zero or one rows is already ordered under every key set and every
    direction, so the sort returns it unchanged. Removing it eliminates a pipeline
    breaker that, over a stream, is also a blocking operator.

    As with the `Distinct` case above, the row bound is **structural**
    (`_at_most_one_row`) rather than estimated. `skip_sort_of_single_row` proves the same
    thing from an EXACT row count and so cannot fire on an unbounded plan; the shapes
    matched here — a `Limit(1)`, a top-1 sort, a keyless aggregate — bound the row count
    by construction no matter what the source does.

    A `limit` on the sort itself is only safe to discard when it cannot remove the row
    that is there: `limit >= 1` keeps the single row, so the sort is an identity, while
    `limit == 0` yields the empty relation and is a genuine change. The zero case is
    refused and left to the empty-propagation rules, which model it properly.

    Args:
        node: The `Sort` under consideration.
        ctx: The optimizer context, read for the bound sources' boundedness.

    Returns:
        The sort's input when that input holds at most one row, else None.
    """
    if node.limit is not None and node.limit < 1:
        return None
    if not _at_most_one_row(node.input):
        return None
    if not has_unbounded_input(node, ctx):
        return None
    return node.input
