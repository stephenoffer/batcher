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

Kyber's lane is unchanged: everything here is a pure function of the plan and the bound
sources. Nothing executes, and nothing records runtime metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Join,
    Limit,
    LogicalPlan,
    Scan,
    Sort,
    Window,
)
from batcher.plan.visitor import walk

if TYPE_CHECKING:
    from batcher.kyber.pass_base import OptimizerContext

__all__ = [
    "blocking_operators",
    "emits_incrementally",
    "has_unbounded_input",
    "is_blocking_under_stream",
    "is_unbounded_scan",
    "retains_unbounded_state",
    "unbounded_source_ids",
]


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

    - a full `Sort` (a top-N `Sort` with a `limit` is *not* blocking — it keeps a bounded
      running best-N and can emit at any point);
    - a `Distinct`, which must have seen every prior row to rule the next one duplicate;
    - a `Window`, whose frame may extend to the end of the partition;
    - a keyless (global) `Aggregate` with no watermark, which has exactly one result row
      that is only correct once the input is exhausted.

    A grouped `Aggregate` is deliberately absent: it emits a running result per group and
    is bounded by the watermark, which is why it is the one stateful operator streaming
    supports today.

    Args:
        node: The plan node to classify.

    Returns:
        True when the operator must see the whole input before emitting.
    """
    if isinstance(node, Sort):
        return node.limit is None
    if isinstance(node, (Distinct, Window)):
        return True
    if isinstance(node, Aggregate):
        return not node.group_keys and node.watermark is None
    return False


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
    windows evict and the state is bounded by the number of *open* windows. A `Join`
    likewise buffers both sides forever unless an interval or watermark bounds it.

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
    if isinstance(node, Join):
        return True
    if isinstance(node, Limit):
        return False
    return False
