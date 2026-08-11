"""Compute a repeated subplan once and read it back (control plane, `api`).

The seam: `kyber.common_subplan` decides *which* subtrees repeat often enough and cheaply
enough to be worth materializing; this module is the half allowed to act on that. It runs
each chosen subplan, wraps the result as an in-memory source, and rewrites every appearance
of the subtree into a `Scan` over it — the same splice `api.adaptive.staging` performs at a
stage boundary, driven by structure rather than by a measured cardinality.

Keeping the two apart is the same split the adaptive loop uses: the decision stays pure and
testable without a query running, and the execution stays in the layer allowed to execute.

**Where it applies.** The single-node relational executor, which is `collect` and every
terminal built on it. Deliberately not the other two routes, and neither is an oversight:

* The **adaptive** route already materializes at every breaker and splices by object
  identity, so a `Dataset` the caller reused is executed once there by construction. Adding
  a second materializing pass in front of it would pay twice for the same thing.
* The **distributed** route would have to keep the shared intermediate partitioned on the
  workers rather than collect it to the driver, which is `staging`'s `MaterializedSource`
  machinery rather than this module's `InMemorySource`. That is a real gap and the reason
  it is stated here rather than left to be discovered.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import weakref
from collections import OrderedDict

import pyarrow as pa

from batcher._internal.logging import get_logger, log_kv, note_suppressed
from batcher.io.source import InMemorySource, Source
from batcher.plan.logical import LogicalPlan, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.visitor import children, transform_up

__all__ = ["reuse_common_subplans"]

_log = get_logger("api.subplan_reuse")

#: Plans already known to have nothing worth reusing, so a re-issued query does not re-derive
#: it. Bounded, and holding only the fact — never a materialized result, which would be a data
#: cache and is `Dataset.cache()`'s job.
#:
#: The analysis is not the cheap walk its docstring claimed. `_one_id_per_source` rebuilds the
#: plan whenever one source object is bound twice (every self-join, and TPC-H q8's two
#: `nation` bindings), and the fresh nodes defeat `content_key`'s per-instance memo, so every
#: `collect()` re-keyed every node. Measured warm at scale 1: **q8 16.2 ms, q5 6.5 ms, q9
#: 7.2 ms — 37%, 18% and 12% of those queries' entire wall time, to conclude "nothing
#: repeats"** each time.
#:
#: The verdict is keyed the way Kyber keys the optimizer memo (`plan_cache.cache_key`), which
#: is what makes caching a *cost-based* rejection sound as well as a structural one: that key
#: already folds in the learned generation, the calibration epoch, the measured read costs and
#: the source statistics, so it moves exactly when the numbers the rejection was taken on move.
#: Restating a narrower key here would have meant either re-deriving the analysis on every call
#: or serving a verdict that had outlived its evidence.
_NO_REUSE: OrderedDict[tuple, tuple[weakref.ref, ...]] = OrderedDict()
_NO_REUSE_MAX = 256
_NO_REUSE_LOCK = threading.Lock()


def _no_reuse_key(plan: LogicalPlan, sources: list[Source], ctx, config, cfg) -> tuple | None:
    """The cache key for "this plan has nothing to reuse", or `None` if it cannot be keyed.

    Kyber's own optimizer-memo key carries the plan fingerprint, the config, the hub and every
    learned input; `kind` separates this question from the optimizer's so the two cannot
    collide in each other's namespace.

    Two things it does not carry, and both matter here. A `Scan`'s IR is only its `source_id`,
    while this analysis turns on *which bindings are the same object* — that is the whole job
    of `_one_id_per_source` — so the identity pattern is appended. It is held as `id()` for the
    lookup and as weak references for the check, exactly as `orchestration.prepared` does and
    for the same reason: a strong reference would keep a table alive, and a bare `id()` can be
    recycled onto a different object. And the two `optimizer` values this analysis reads are
    appended too, since a plan-cache key need not distinguish them.
    """
    from batcher.kyber import plan_cache

    try:
        base = plan_cache.cache_key(
            plan.content_key(),
            sources,
            config,
            ctx.hub,
            kind="subplan_reuse",
            source_stats=ctx.source_stats,
        )
    except Exception as exc:  # pragma: no cover - an unkeyable plan simply is not cached
        note_suppressed("api", "key the common-subplan verdict", exc)
        return None
    if base is None:
        return None
    return (base, tuple(id(s) for s in sources), cfg.common_subplan_max_bytes, cfg.row_bytes)


def _known_no_reuse(key: tuple | None, sources: list[Source]) -> bool:
    """Whether `key` is a recorded "nothing to reuse" verdict for *these* source objects."""
    if key is None:
        return False
    with _NO_REUSE_LOCK:
        held = _NO_REUSE.get(key)
        if held is None:
            return False
        if len(held) != len(sources) or any(
            r() is not s for r, s in zip(held, sources, strict=True)
        ):
            # A recycled `id()` landed on a different object: drop the entry rather than
            # serve one plan's verdict for another's.
            del _NO_REUSE[key]
            return False
        _NO_REUSE.move_to_end(key)
    return True


def _record_no_reuse(key: tuple | None, sources: list[Source]) -> None:
    """Record that `key`'s plan has nothing worth reusing. Best-effort."""
    if key is None:
        return
    try:
        held = tuple(weakref.ref(s) for s in sources)
    except TypeError:  # pragma: no cover - a source that cannot be weakly referenced
        return
    with _NO_REUSE_LOCK:
        _NO_REUSE[key] = held
        _NO_REUSE.move_to_end(key)
        while len(_NO_REUSE) > _NO_REUSE_MAX:
            _NO_REUSE.popitem(last=False)


def reuse_common_subplans(
    plan: LogicalPlan, sources: list[Source], ctx
) -> tuple[LogicalPlan, list[Source]]:
    """Rewrite `plan` so each repeated subplan is executed once and scanned thereafter.

    Returns the plan unchanged (and the same `sources` list) whenever nothing repeats,
    which is the common case and costs one walk of the plan. Otherwise each chosen subplan
    is executed now, its result appended to `sources` as an `InMemorySource`, and every
    structurally identical appearance replaced by a `Scan` over it.

    Best-effort by construction: this is an optimization, and a query that would have
    produced an answer must still produce it. Any failure while analyzing or materializing
    is logged and the original plan is returned, so the worst case is the work that was
    already being done twice.

    Args:
        plan: The plan about to run.
        sources: Its bound inputs. Never mutated; a new list is returned when a subplan
            was materialized.
        ctx: The `ExecutionContext` the caller will execute with, reused for the
            materializing runs so they see the same hub and source statistics.

    Returns:
        The rewritten plan and the sources it is bound to.
    """
    try:
        return _reuse(plan, sources, ctx)
    except Exception as exc:  # pragma: no cover - an optimization must never break a query
        note_suppressed("api", "reuse common subplans", exc)
        return plan, sources


def _reuse(plan: LogicalPlan, sources: list[Source], ctx) -> tuple[LogicalPlan, list[Source]]:
    from batcher.config import active_config
    from batcher.io.source import is_bounded
    from batcher.kyber.common_subplan import common_subplans

    config = active_config()
    cfg = config.optimizer
    if cfg.common_subplan_max_bytes <= 0:
        return plan, sources
    # An unbounded source has no finite intermediate to hold, and a plan carrying one is
    # already routed to the streaming path rather than here.
    if not all(is_bounded(s) for s in sources):
        return plan, sources
    # A single scan under the root cannot repeat a subtree — the cheapest possible check,
    # and it is what most plans hit.
    if not children(plan):
        return plan, sources
    key = _no_reuse_key(plan, sources, ctx, config, cfg)
    if _known_no_reuse(key, sources):
        return plan, sources

    from batcher.api.source_stats import build_estimator

    canonical = _one_id_per_source(plan, sources)
    targets = common_subplans(
        canonical,
        lambda: build_estimator(sources, ctx.hub),
        max_bytes=cfg.common_subplan_max_bytes,
        row_bytes=cfg.row_bytes,
    )
    if not targets:
        # Hand back the plan as written. The canonical form is semantically identical, but
        # it is only built to make repeats *visible*, and returning it when nothing repeats
        # would perturb plan identity (the result-cache key, learned-stats signatures) for
        # no gain at all.
        _record_no_reuse(key, sources)
        return plan, sources
    plan = canonical

    srcs = list(sources)
    # The estimator bounded each candidate on its own; this bounds what they hold *together*,
    # which is the quantity that actually competes with the running query for memory. Three
    # candidates each just inside a 256 MiB budget is three quarters of a gigabyte held until
    # the query ends. Charged on the materialized size rather than the estimate, so a target
    # whose estimate was optimistic stops the ones after it instead of compounding.
    held = 0
    for target in targets:
        table = _materialize(target, srcs, ctx)
        if table is None:
            continue
        held += table.nbytes
        if held > cfg.common_subplan_max_bytes:
            log_kv(
                _log,
                logging.DEBUG,
                "subplan reuse budget reached",
                held=held,
                budget=cfg.common_subplan_max_bytes,
            )
            break
        sid = len(srcs)
        # No zone maps and `ephemeral`, for the reasons `staging._stage_source` spells out:
        # this relation lives for one query, so an O(rows) min/max pass over it would be
        # recomputed and discarded on every run, and its identity must not seed a
        # distinct-count sketch that outlives it.
        srcs.append(InMemorySource(_batches(table), zone_maps=False, ephemeral=True))
        plan = _replace_all(plan, target, Scan(sid, SchemaRef.from_arrow(table.schema)))
        log_kv(
            _log,
            logging.DEBUG,
            "subplan reused",
            rows=table.num_rows,
            bytes=table.nbytes,
            op=type(target).__name__,
        )
    return plan, srcs


def _one_id_per_source(plan: LogicalPlan, sources: list[Source]) -> LogicalPlan:
    """Point every binding of one source object at that object's first index.

    Without this the analysis below finds nothing on the shape it exists for. `Dataset.join`
    concatenates its two operands' source lists and renumbers the right-hand side's scans,
    so joining a dataset with something derived from *itself* binds the identical `Source`
    object at two indices. The two subtrees are then not structurally equal — one scans
    source 0 and the other source 1 — even though they read the same bytes and compute the
    same relation. `agg.join(agg.filter(...))`, the canonical shape here, lands exactly
    there and measured as "no repeated subplan" until this ran first.

    Matched on **object identity**, not on `Source.identity()`. The data-stable identity
    would also fold two separately-constructed sources over equal data, which is very
    probably right and is not needed for anything here — the reuse this collapses comes from
    a `Dataset` the caller reused, which is the same object by construction. Identity is
    free to be conservative; being wrong is not.

    Args:
        plan: The plan to canonicalize.
        sources: Its bound inputs, positionally.

    Returns:
        The plan with duplicate source bindings collapsed, or the same object when there
        are none (the common case).
    """
    first: dict[int, int] = {}
    alias = {i: first.setdefault(id(s), i) for i, s in enumerate(sources)}
    if all(i == a for i, a in alias.items()):
        return plan
    return transform_up(
        plan,
        lambda n: (
            dataclasses.replace(n, source_id=alias[n.source_id])
            if isinstance(n, Scan) and alias.get(n.source_id, n.source_id) != n.source_id
            else n
        ),
    )


def _materialize(target: LogicalPlan, sources: list[Source], ctx) -> pa.Table | None:
    """Run one shared subplan, or `None` if it cannot be run on this path.

    The subplan is executed with the caller's own context so it reads the same hub and
    already-collected source statistics — but with the *subplan's* output columns, since
    `ctx.columns` names the root's, and with the result cache off: caching an intermediate
    the user never asked for would spend the cache's budget on something no later query
    asks for by name.
    """
    from batcher.api.orchestration.run import run_relational

    try:
        table, _decisions = run_relational(
            target,
            sources,
            dataclasses.replace(ctx, columns=target.available_columns(), cache=False),
            distributed=False,
        )
    except Exception as exc:  # pragma: no cover - fall back to recomputing in place
        note_suppressed("api", "materialize a shared subplan", exc)
        return None
    return table if isinstance(table, pa.Table) else None


def _batches(table: pa.Table) -> list[pa.RecordBatch]:
    """`table` as batches, never empty — an empty relation still carries its types."""
    return table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]


def _replace_all(plan: LogicalPlan, target: LogicalPlan, repl: LogicalPlan) -> LogicalPlan:
    """Replace *every* subtree computing what `target` computes, not just this object.

    `adaptive.plan_surgery.replace` matches on object identity, which is right for the stage
    loop: it executes one specific node it just located. Here the whole point is the other
    appearances, which are equal subtrees built independently — `agg.join(agg.filter(...))`
    shares the object, a SQL `WITH` referenced twice does not, and both must collapse. So
    the match is on the IR, the exact form the engine would execute.

    Bottom-up, so a rewritten child is in place before its parent is compared; `target`
    itself is by construction not nested in another target (`common_subplans` returns
    non-overlapping subtrees), so no appearance is rewritten twice.
    """
    from batcher.kyber.common_subplan import structural_key

    want = structural_key(target)
    return transform_up(plan, lambda node: repl if structural_key(node) == want else node)
