"""Streaming analysis for the optimizer — what is unbounded, and what that forbids.

Kyber was streaming-blind. Its only signal about an unbounded input was the sentinel
`rows >= config.optimizer.unknown_rows` with `Provenance.DEFAULT`, which a stream shares
with every *bounded* source whose size merely could not be estimated (`from_batches`, an
un-pushed SQL scan). A rule could not distinguish "big, unknown" from "never ends", so it
could not reason about the one thing that actually differs.

That distinction matters because unboundedness changes which rewrites are *legal*, not
just which are fast:

- **A blocking operator never emits.** A full `Sort`, or a `Distinct` over an unbounded
  input, must see the last row before it can produce the first. Under a stream there is
  no last row. A rewrite that *introduces* one turns a working query into a hang.
- **State must be bounded by something that advances.** A grouped aggregate over a
  stream holds one entry per group forever unless a watermark closes and evicts it. A
  rewrite that adds a group key, or drops the watermark, converts bounded state into a
  slow memory leak — and it leaks in production, never in a bounded test.
- **Some rewrites become *more* valuable.** Pushing a filter below a stateful streaming
  operator shrinks the state itself, not merely the rows scanned; that is a far larger
  win under a stream than in batch, where it only saves CPU.

**Every node type is classified, and a test proves it.** The two predicates below used to
answer from a short `isinstance` chain covering six of the twenty-seven `LogicalPlan`
nodes, with everything unlisted falling through to "streams fine, retains nothing" — the
permissive default. That default is exactly backwards for this question. A `Distinct`
answered "retains nothing" while holding one entry per distinct value forever; a
`Sample(n=...)` answered "not blocking" while being a reservoir that cannot emit until
the stream ends; and `TransformWithState` answered "retains nothing" while its own
docstring names it as the shape this module is *entitled to complain about*. None of
those is visible in a bounded test, because a bounded input always ends and always
releases its state. So `STREAM_CLASSIFIED` names every node either predicate reasons
about, and `tests/unit/test_kyber_streaming_rules.py` fails when a `LogicalPlan`
subclass is added without a decision here — the same "every tag is classified" contract
the device tier runs on, for the same reason: the failure mode is silence.

Kyber's lane is unchanged: everything here is a pure function of the plan and the bound
sources. Nothing executes, and nothing records runtime metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    MapBatches,
    Project,
    RangeJoin,
    RowId,
    Sample,
    Scan,
    Sort,
    StreamingSessionWindow,
    TransformWithState,
    Union,
    Unnest,
    Unpivot,
    WatermarkDedup,
    WatermarkStreamJoin,
    Window,
)
from batcher.plan.visitor import walk

if TYPE_CHECKING:
    from batcher.kyber.pass_base import OptimizerContext

__all__ = [
    "STREAM_CLASSIFIED",
    "blocking_operators",
    "emits_incrementally",
    "has_unbounded_input",
    "is_blocking_under_stream",
    "is_unbounded_scan",
    "retains_unbounded_state",
    "unbounded_source_ids",
    "unbounded_state_operators",
]

#: Every `LogicalPlan` node type the predicates below have a considered answer for.
#:
#: Membership is the contract, not the answer: a node here has been reasoned about, and
#: the reasoning is in `is_blocking_under_stream` / `retains_unbounded_state`. A node
#: *not* here would silently take the permissive default from both, which is how a
#: leaking operator ships. The exhaustiveness test walks `LogicalPlan.__subclasses__`
#: against this set, so adding a node without deciding fails the build.
STREAM_CLASSIFIED: frozenset[type[LogicalPlan]] = frozenset(
    {
        # --- stateless and row-wise: stream freely, retain nothing -----------------
        Scan,
        Filter,
        Project,
        MapBatches,
        Unnest,
        Unpivot,
        RowId,
        Limit,
        # --- bounded by a watermark: stream, and evict on watermark advance --------
        WatermarkDedup,
        StreamingSessionWindow,
        WatermarkStreamJoin,
        # --- the ones whose answer depends on a field -----------------------------
        Sort,
        Distinct,
        Aggregate,
        Sample,
        Union,
        TransformWithState,
        # --- unconditionally stateful breakers ------------------------------------
        Window,
        Join,
        AsofJoin,
        RangeJoin,
    }
)


def is_unbounded_scan(node: LogicalPlan, ctx: OptimizerContext) -> bool:
    """Whether `node` is a `Scan` of a source that never ends.

    Answered from the bound source itself (`io.source.is_bounded`), not from a row
    estimate — an unknown row count means "not estimated", which a finite source can
    report just as easily as a stream.

    Args:
        node: The plan node to test.
        ctx: The optimizer context, whose `sources` are the bound inputs.

    Returns:
        True only when `node` is a scan of a declared-unbounded source.
    """
    from batcher.io.source import is_bounded

    if not isinstance(node, Scan):
        return False
    sources = ctx.sources or []
    if not 0 <= node.source_id < len(sources):
        return False  # an unbound or relabeled scan — assume bounded, the safe default
    return not is_bounded(sources[node.source_id])


def unbounded_source_ids(plan: LogicalPlan, ctx: OptimizerContext) -> frozenset[int]:
    """The `source_id`s of every unbounded scan in `plan`.

    Args:
        plan: The plan to walk.
        ctx: The optimizer context carrying the bound sources.

    Returns:
        The set of unbounded source ids, empty for a wholly bounded plan.
    """
    return frozenset(n.source_id for n in walk(plan) if is_unbounded_scan(n, ctx))


def has_unbounded_input(plan: LogicalPlan, ctx: OptimizerContext) -> bool:
    """Whether any leaf of `plan` is an unbounded source.

    This is the predicate most streaming rules gate on: a rewrite that is merely
    suboptimal in batch can be *incorrect* — a hang, or unbounded state — here.

    Args:
        plan: The plan to walk.
        ctx: The optimizer context carrying the bound sources.

    Returns:
        True when the plan reads at least one stream.
    """
    return bool(unbounded_source_ids(plan, ctx))


def is_blocking_under_stream(node: LogicalPlan) -> bool:
    """Whether `node` cannot emit its first row until its input ends.

    These are the operators that turn an unbounded input into a query that produces
    nothing at all, as opposed to one that merely uses more memory:

    - a full `Sort`, though a top-N `Sort` carrying a `limit` is *not* blocking — it
      keeps a bounded running best-N and can emit at any point;
    - a `Distinct`, which must have seen every prior row to rule the next one duplicate
      — unless it carries a fused `limit`, which stops at the first `limit` distinct
      rows and so settles on a prefix;
    - a `Union` with `distinct=True`, which is that same dedup over the concatenation;
    - a `Window`, whose frame may extend to the end of the partition, and whose
      partitions a stream never finishes;
    - a keyless (global) `Aggregate` with no watermark, which has exactly one result row
      that is only correct once the input is exhausted;
    - a fixed-count `Sample(n=...)`, which is a reservoir: *which* n rows it keeps is
      only decided by the last arrival. The fraction form is a per-row hash test and
      streams freely, which is the same distinction `is_partition_independent` draws;
    - an `AsofJoin` or `RangeJoin`, both of which order and materialize their right side
      before the first left row can be answered.

    A grouped `Aggregate` is deliberately absent: it emits a running result per group and
    is bounded by the watermark, which is why it is the one stateful operator streaming
    supports today. So are the three watermark-bounded nodes (`WatermarkDedup`,
    `StreamingSessionWindow`, `WatermarkStreamJoin`), which exist precisely to emit on an
    advancing watermark rather than at end-of-input, and `TransformWithState`, which
    emits once per key per micro-batch.

    A plain `Join` is also absent, and that is a *scope* limit rather than a claim: a
    hash join blocks only if its build side is unbounded, which one node cannot see.
    `retains_unbounded_state` is what flags it.

    Args:
        node: The plan node to classify.

    Returns:
        True when the operator must see the whole input before emitting.
    """
    if isinstance(node, Sort):
        return node.limit is None
    if isinstance(node, Distinct):
        return node.limit is None
    if isinstance(node, Union):
        return node.distinct
    if isinstance(node, Window):
        return True
    if isinstance(node, Aggregate):
        return not node.group_keys and node.watermark is None
    if isinstance(node, Sample):
        return node.n is not None
    return isinstance(node, (AsofJoin, RangeJoin))


def blocking_operators(plan: LogicalPlan) -> list[LogicalPlan]:
    """Every node in `plan` that cannot emit before its input ends.

    Args:
        plan: The plan to walk.

    Returns:
        The blocking nodes, in an unspecified order (empty when the plan can stream).
    """
    return [n for n in walk(plan) if is_blocking_under_stream(n)]


def emits_incrementally(plan: LogicalPlan, ctx: OptimizerContext) -> bool:
    """Whether `plan` can produce output before its input is exhausted.

    Args:
        plan: The plan to classify.
        ctx: The optimizer context carrying the bound sources.

    Returns:
        True for a bounded plan (which always terminates) or an unbounded plan with no
        blocking operator; False when a stream feeds an operator that can never emit.
    """
    if not has_unbounded_input(plan, ctx):
        return True
    return not blocking_operators(plan)


def retains_unbounded_state(node: LogicalPlan) -> bool:
    """Whether `node` accumulates state that nothing will ever release.

    A grouped aggregate without a watermark keeps one entry per group for the life of
    the query: correct, and a memory leak measured in days. With a watermark, closed
    windows evict and the state is bounded by the number of *open* windows. The same
    question separates every stateful operator here:

    - a `Distinct` holds one entry per *distinct value*, forever, with nothing to evict
      it — the shape a watermark-bounded `WatermarkDedup` exists to replace. A `Distinct`
      carrying a fused `limit` stops at that many rows and is bounded by it;
    - a `Union` with `distinct=True` is the same dedup over the concatenation;
    - a full `Sort` buffers the whole input; a top-N `Sort` holds `limit` rows;
    - a `Window` holds every row of every partition it has seen, and a stream never
      closes a partition — `rank_limit` bounds each partition's heap but not the number
      of partitions;
    - a plain `Join`, an `AsofJoin`, and a `RangeJoin` all buffer a whole side, which an
      interval or watermark (`WatermarkStreamJoin`) is what bounds;
    - `TransformWithState` holds one user state per key until `ttl_micros` expires it;
      ``0`` means never, which is only correct for a bounded key space and is the shape
      this predicate is documented to complain about.

    A fixed-count `Sample(n=...)` is deliberately absent: a reservoir holds exactly `n`
    rows, so it is *blocking* without being unbounded — the two properties are
    independent and this is the case that proves it.

    This is the predicate that distinguishes "streams but leaks" from "streams
    safely" — a distinction no bounded test can make, because a bounded input always
    releases the state at end-of-input.

    Args:
        node: The plan node to classify.

    Returns:
        True when the operator's state has no eviction mechanism.
    """
    if isinstance(node, Aggregate):
        return bool(node.group_keys) and node.watermark is None
    if isinstance(node, Distinct):
        return node.limit is None
    if isinstance(node, Union):
        return node.distinct
    if isinstance(node, Sort):
        return node.limit is None
    if isinstance(node, TransformWithState):
        return node.ttl_micros <= 0
    return isinstance(node, (Window, Join, AsofJoin, RangeJoin))


def unbounded_state_operators(plan: LogicalPlan) -> list[LogicalPlan]:
    """Every node in `plan` whose retained state nothing will release.

    The state-side counterpart of `blocking_operators`, and the reason both exist: a
    plan can stream perfectly and still leak, which is the failure that only appears in
    production. Callers that report to a user want the offending nodes, not a bare bool.

    **A `Limit` bounds the operator directly beneath it**, which `retains_unbounded_state`
    cannot see because it classifies one node. `sort(...).head(10)` and
    `distinct().head(10)` each hold ten rows, not the stream: the engine fuses the limit
    into the operator (`Sort.limit`, `Distinct.limit`, and the Kyber rules that set them),
    and both nodes carry the field precisely because they can. Reporting them would be a
    false alarm on two of the most ordinary things anyone types against a topic — and a
    warning that cries wolf on `head(10)` is one nobody reads on the query that does leak.

    Only the *direct* input is discounted. A row-wise operator between the limit and the
    breaker still leaves the breaker unbounded in general (a `Filter` under a `Limit` does
    not cap what the operator below the filter retains), so the narrow reading is the safe
    one, and it is the one that matches what the fusion rules actually rewrite.

    Args:
        plan: The plan to walk.

    Returns:
        The leaking nodes, in an unspecified order (empty when every operator's state is
        bounded by something that advances).
    """
    capped = {
        id(n.input)
        for n in walk(plan)
        if isinstance(n, Limit) and isinstance(n.input, (Sort, Distinct))
    }
    return [n for n in walk(plan) if retains_unbounded_state(n) and id(n) not in capped]
