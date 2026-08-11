"""Plan-level common-subplan elimination: which repeated subplans to compute once.

Kyber already eliminates a repeated *expression* inside a `Project` — `rules/extra/cse.py`
binds `regexp_extract(url, p)` to one synthetic column when three output columns each
compute it. This module is that idea one level up, on the plan itself, which is where the
same waste is far more expensive: a subtree appearing twice in a query is **executed
twice**, so a `GROUP BY` feeding both operands of a join runs twice, and a CTE referenced
three times runs three times.

It is not a hypothetical shape. It is what `agg.join(agg.filter(...))` builds, what a SQL
`WITH` clause referenced more than once builds, and what any self-join over a derived table
builds. Measured on a 4 M-row `GROUP BY` over a 200 K-key column feeding both operands of a
join, interleaved A/B, medians of six: **293 ms** without this and **150 ms** with it,
**1.95x** — against 158 ms for the shared aggregate on its own, so the rewritten query costs
about what computing the shared half once costs, which is the floor. The gap grows with the
shared subtree's cost and with how many times it appears.

**This module only decides.** It returns the subplans worth materializing and never
executes, rewrites, or measures anything; `api.subplan_reuse` runs them and splices the
results back in. That is the same split `gating` and `staging` use for the adaptive loop,
and it is here for the same reason: the policy stays pure, so it is unit-testable without
running a query, and the execution stays in the layer allowed to execute.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from batcher._internal.logging import note_suppressed
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Join,
    LogicalPlan,
    Sort,
    Window,
)
from batcher.plan.visitor import walk

__all__ = ["common_subplans", "structural_key"]

# Recomputing a subtree is only worth avoiding when it is expensive, and a pipeline breaker
# is exactly what "expensive" means here: it materializes its input and reduces or reorders
# the whole relation, so running it twice does that twice.
#
# A repeated scan/filter/project pipeline is deliberately NOT a candidate. It streams, it
# fuses into whatever reads it, and materializing it would insert a full copy the fused form
# never made — turning a free repeat into a paid one. `Limit` and `Union` are breakers to the
# stage loop (they bound or concatenate) but not costly ones, so they are not on this list
# either; a subtree is a candidate when it *contains* one of these, wherever it sits.
_EXPENSIVE = (Aggregate, Join, Sort, Distinct, Window)


def common_subplans(
    plan: LogicalPlan,
    estimator: Callable[[], object],
    *,
    max_bytes: int,
    row_bytes: int,
    max_nodes: int = 400,
) -> list[LogicalPlan]:
    """The subplans in `plan` that are worth computing once instead of once per appearance.

    A candidate has to clear five bars, and each one exists because skipping it is a way to
    make a query *slower* rather than faster:

    1. **It appears at least twice.** One appearance has nothing to share.
    2. **It contains a pipeline breaker** (`_EXPENSIVE`). Recomputing a streaming
       scan/filter/project is cheaper than materializing it.
    3. **Its result fits `max_bytes`.** Reuse buys a saved execution and costs the memory to
       hold the result for the rest of the query, so an intermediate too large to hold is
       not a candidate however often it repeats.
    4. **It is not the whole plan.** Materializing the root and then scanning it back is
       pure overhead — there is no second appearance to serve.
    5. **Materializing it saves a large enough share of the plan's cost**
       (`_MIN_SAVED_SHARE`). See
       `_worth_materializing`: bars 1-4 say the subtree *can* be shared and say nothing
       about whether sharing it pays, and without this bar it frequently does not.

    Candidates are returned **largest subtree first**, and one nested inside an accepted
    candidate is dropped: materializing the outer subtree already collapses every
    appearance of the inner one that lies within it. That makes the returned list
    non-overlapping, which is what lets the caller materialize them independently and in any
    order.

    Args:
        plan: The plan about to be executed.
        estimator: Builds a `CardinalityEstimator` over the plan's bound sources — the same
            one the optimizer costs with, so the size gate reads the optimizer's own
            numbers. A **factory**, not the estimator, and called at most once: building it
            collects every source's statistics, which is real work (a footerless source
            measures itself) and is pure waste on a plan with nothing repeated — which is
            almost every plan. The structural question is asked first and answers most
            calls without it.
        max_bytes: The largest materialized result worth holding for reuse.
        row_bytes: The per-row width fallback for a node whose schema gives no better one.
        max_nodes: Skip the analysis for a plan larger than this. The scan is quadratic in
            plan size (each node's structural key encodes its whole subtree), which is
            nothing at the size real plans reach and not worth risking on a generated one.

    Returns:
        The subplans to compute once, outermost first. Empty when nothing qualifies, which
        is the overwhelmingly common case and costs one walk of the plan.
    """
    nodes = list(walk(plan))
    if len(nodes) > max_nodes:
        return []
    # No pipeline breaker anywhere in the plan ⇒ no candidate can exist, because bar 2 below
    # requires every candidate to *contain* one. Checking it here rather than only at the
    # per-candidate filter is exact, not a heuristic: a subtree of a plan with no `_EXPENSIVE`
    # node has none either.
    #
    # It is worth doing early because the work it skips is the expensive half. `structural_key`
    # serializes a node's whole subtree IR to JSON, and the loop below calls it once per node
    # before anything has been filtered — so a plain `select` over a scan, which cannot share
    # anything, paid two whole-plan JSON encodes per query. Measured on a warm 20,000-row
    # query, `json.encoder.iterencode` was 0.045 s against 0.062 s for the entire engine call.
    if not any(isinstance(n, _EXPENSIVE) for n in nodes):
        return []
    keyed: list[tuple[str, LogicalPlan]] = []
    for node in nodes:
        key = structural_key(node)
        if key is not None:
            keyed.append((key, node))
    appearances = Counter(k for k, _ in keyed)
    repeated = {k for k, n in appearances.items() if n >= 2}
    if not repeated:
        return []

    root_key = structural_key(plan)
    # One representative per repeated key, largest subtree first, so an outer candidate is
    # considered before anything nested inside it.
    seen: dict[str, LogicalPlan] = {}
    for key, node in keyed:
        if key in repeated and key not in seen:
            seen[key] = node
    ordered = sorted(seen.items(), key=lambda kv: -_subtree_size(kv[1]))

    accepted: list[LogicalPlan] = []
    covered: set[str] = set()
    sized = None
    for key, node in ordered:
        if key == root_key or key in covered:
            continue
        if not any(isinstance(n, _EXPENSIVE) for n in walk(node)):
            continue
        if sized is None:
            sized = estimator()
        if not _fits(node, sized, max_bytes, row_bytes):
            continue
        if not _worth_materializing(node, plan, sized, appearances[key]):
            continue
        accepted.append(node)
        covered.update(k for k in map(structural_key, walk(node)) if k is not None)
    return accepted


#: Share of the whole plan's cost a repeated subtree must carry before materializing it pays.
#:
#: Bars 1-4 answer "can this be shared". They do not answer "does sharing it pay", and the
#: difference is not academic: TPC-H q20 has a repeated semi-join that clears all four and is
#: **2.2x slower** materialized (measured cold, so no plan-cache effect either way; 4.3x warm).
#: The reason is that materializing is not free the way "run it once instead of twice" makes it
#: sound. It costs a *separate engine round trip* — its own plan serialization, its own FFI
#: crossing, its own Arrow table build — and it forfeits the fusion the subtree had with its
#: parent, turning a streamed branch into a scan of a materialized table. On q20 that fixed
#: cost measured ~18 ms against a ~19 ms query: about as much as everything else put together.
#:
#: So the saving has to be worth a whole extra execution, and the natural way to say that is
#: as a share of the plan. The two ends are far apart, which is why this is a threshold and
#: not a tuning knob:
#:
#: | subtree                                        | share | materialized |
#: |------------------------------------------------|------:|--------------|
#: | TPC-H q20's repeated `partsupp ⋈ part` semi-join| 13.5% | 2.2x slower  |
#: | the 4M-row `GROUP BY` feeding both join operands| 49.8% | 1.95x faster |
#:
#: A sixth sits between them with a wide margin either side, and it is where the arithmetic
#: puts it. The cost model counts a subtree once per appearance, so a subtree of share `s`
#: appearing `a` times has `(a-1)/a` of that share removed by materializing it: the *saving*
#: is `s * (a-1)/a`. At `a = 2` a sixth is the familiar "a third of the plan"; at `a = 3` it
#: asks only a quarter, which is right — three appearances save two runs for the same one
#: fixed cost, so the same fixed cost is worth clearing a lower bar for.
_MIN_SAVED_SHARE = 1.0 / 6.0


def _worth_materializing(node: LogicalPlan, plan: LogicalPlan, estimator, appearances: int) -> bool:
    """Whether materializing `node` saves enough of `plan`'s cost to pay for itself.

    Priced in Kyber's own currency (`CostModel`) rather than in rows, because what a subtree
    costs is not its row count: a semi-join over 800,000 rows that emits 8,508 is cheap, and
    an aggregate over four million is not, and reuse must be able to tell them apart.

    A cost the model cannot produce is not evidence for materializing, so it declines — the
    same direction `_fits` takes for a missing size estimate.

    Args:
        node: The candidate subtree.
        plan: The whole plan it sits in, which is what its cost is judged against.
        estimator: The `CardinalityEstimator` the sizes come from.
        appearances: How many times `node` occurs in `plan`. Materializing replaces all but
            one of them, so this is what turns a share of the cost into a saving.

    Returns:
        Whether the saved share clears `_MIN_SAVED_SHARE`.
    """
    from batcher.kyber.cost.model import CostModel

    try:
        model = CostModel(estimator)
        total = model.cost(plan).total()
        if total <= 0 or appearances < 2:
            return False
        share = model.cost(node).total() / total
        return share * (appearances - 1) / appearances >= _MIN_SAVED_SHARE
    except Exception as exc:  # pragma: no cover - a cost failure must not break planning
        note_suppressed("kyber", "cost a common-subplan candidate", exc)
        return False


def _fits(node: LogicalPlan, estimator, max_bytes: int, row_bytes: int) -> bool:
    """Whether `node`'s result is small enough to hold for the rest of the query."""
    try:
        stats = estimator.estimate(node)
        width = estimator.row_width(node, row_bytes)
    except Exception:
        # An estimate this analysis cannot obtain is not a reason to guess: without a size
        # there is no way to bound what materializing would hold, so decline.
        return False
    return stats.rows * width <= max_bytes


def _subtree_size(node: LogicalPlan) -> int:
    return sum(1 for _ in walk(node))


def structural_key(node: LogicalPlan) -> str | None:
    """An exact identity for "this subtree computes the same relation as that one".

    The node's own JSON IR, which is the wire form the engine executes — so two subtrees
    with equal keys are, by construction, two requests for the identical computation over
    the identical sources (`Scan` carries its `source_id`). Deliberately **not**
    `kyber.signature.plan_signature`, which normalizes literals so statistics generalize
    across runs: that is the right key for learning and exactly the wrong one here, where
    `filter(x > 10)` and `filter(x > 99)` must not be treated as the same relation.

    `None` for a subtree that has no IR — a `MapBatches` raises by design, being opaque
    user code. That is also the answer this analysis wants: a UDF is free to be
    non-deterministic or to have side effects, so collapsing two appearances into one run
    could change the result. Its ancestors return `None` for the same reason, since their
    IR contains it.

    That is the *whole* determinism argument, and it is worth being explicit that nothing
    else is needed. Everything the IR can express is a deterministic function of its input:
    the scalar `Expr` algebra is pure (`rules/extra/cse.py::_is_hoistable` rests on the same
    fact), `current_timestamp`/`current_date` are bound to a `Lit` at plan-build time rather
    than read per row, and even `Sample` — the one node that sounds like a counterexample —
    keeps a row iff a *seeded* hash of its values falls under the fraction, with the seed in
    the node and therefore in this key. So two subtrees with equal IR return equal
    relations, and running one of them once is indistinguishable from running both.

    Delegated to `LogicalPlan.content_key`, which is the engine's one definition of "these
    two plans are the same computation" and is already what `kyber.plan_cache` keys on.
    Two things follow, and both are why this is not merely a tidy-up. It is **memoized per
    node instance**, where the raw `json.dumps` here was not — and this function is called
    once per node of a plan whose every node encodes its whole subtree, so re-serializing
    was quadratic *and* repeated on every `collect()`. Measured on TPC-H q8, `iterencode`
    was 0.103 s of the analysis's 0.132 s. And `content_key` folds in each node's
    `identity_suffix()`, so a `Scan`'s *schema* is part of the key: two sources with the
    same column names and different column types no longer read as the same relation, which
    the bare IR could not distinguish because a `Scan`'s IR is only its `source_id`.
    """
    try:
        node.to_ir()  # memoized; raises for an opaque `MapBatches` and for its ancestors
    except Exception:
        return None
    return node.content_key()
