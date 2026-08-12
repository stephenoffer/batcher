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
from batcher.plan.visitor import children, transform_up, walk

__all__ = ["reuse_common_subplans"]

_log = get_logger("api.subplan_reuse")

#: The analysis's **verdict** for a plan already analyzed, so a re-issued query does not
#: re-derive it. Bounded, and holding only the decision — never a materialized result, which
#: would be a data cache and is `Dataset.cache()`'s job.
#:
#: A verdict is the pre-order **positions** of each chosen subtree's appearances, and the
#: empty list is the (much commoner) "nothing repeats". Positions rather than nodes or keys
#: because the key below carries `plan.content_key()` — the plan's whole lowered IR — so an
#: entry can only be served to a plan with the identical tree, where position `i` of the
#: pre-order walk is the identical node. That makes a hit cost one `walk`, against the
#: canonical rebuild plus a `structural_key` per node plus a `CostModel` pass over the plan
#: that deriving it costs: measured on TPC-DS q80, **404 ms per collect** (337 ms of analysis
#: and 67 ms of canonicalization) against a whole query that runs in 151 ms.
#:
#: Caching the positive verdict is what makes that saving reachable at all. Only rejections
#: were cached before, on the reasoning that a plan with something to reuse pays the analysis
#: once and then the materialization dominates — true when the analysis was believed to be a
#: walk, and false by two orders of magnitude on a snowflake query.
#:
#: The analysis is not the cheap walk its docstring claimed. `_one_id_per_source` rebuilds the
#: plan whenever one source object is bound twice (every self-join, and TPC-H q8's two
#: `nation` bindings), and the fresh nodes defeat `content_key`'s per-instance memo, so every
#: `collect()` re-keyed every node. Measured warm at scale 1: **TPC-H q8 16.2 ms, q5 6.5 ms,
#: q9 7.2 ms — 37%, 18% and 12% of those queries' entire wall time, to conclude "nothing
#: repeats"** each time.
#:
#: The verdict is keyed with Kyber's own key builder but **without its learned fields**
#: (`learned=False`), so it moves with the plan, the config, the hub and the sources and not
#: with the generation counter or the calibration fingerprint. Carrying those was the obvious
#: choice and it is measurably the wrong one: inside a mixed workload every query moves the
#: generation for every other, so the key never repeated and the analysis ran in full on every
#: execution forever. TPC-DS q80 in isolation is 97 ms with the verdict served and 848 ms
#: inside the suite without it.
#:
#: What that trades away is small and stated plainly: a verdict taken under one set of
#: estimates can outlive them, so a subtree that stops being worth materializing keeps being
#: materialized until the plan, config or sources change. That costs a slower query, never a
#: wrong one — and the decision is far less sensitive than a plan, since it asks only whether
#: a subtree repeats and whether materializing it beats an engine round trip.
_VERDICTS: OrderedDict[tuple, tuple[tuple[weakref.ref, ...], tuple[tuple[int, ...], ...]]] = (
    OrderedDict()
)
_VERDICTS_MAX = 256
_VERDICTS_LOCK = threading.Lock()


def _verdict_key(plan: LogicalPlan, sources: list[Source], ctx, config, cfg) -> tuple | None:
    """The cache key for this plan's reuse verdict, or `None` if it cannot be keyed.

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
            learned=False,
        )
    except Exception as exc:  # pragma: no cover - an unkeyable plan simply is not cached
        note_suppressed("api", "key the common-subplan verdict", exc)
        return None
    if base is None:
        return None
    return (base, tuple(id(s) for s in sources), cfg.common_subplan_max_bytes, cfg.row_bytes)


def _known_verdict(key: tuple | None, sources: list[Source]):
    """The recorded verdict for `key` over *these* source objects, or `None` if there is none.

    A verdict is a tuple of per-target appearance-position tuples; the empty tuple is
    "nothing worth reusing", which is why the caller must distinguish it from `None`.
    """
    if key is None:
        return None
    with _VERDICTS_LOCK:
        entry = _VERDICTS.get(key)
        if entry is None:
            return None
        held, verdict = entry
        if len(held) != len(sources) or any(
            r() is not s for r, s in zip(held, sources, strict=True)
        ):
            # A recycled `id()` landed on a different object: drop the entry rather than
            # serve one plan's verdict for another's.
            del _VERDICTS[key]
            return None
        _VERDICTS.move_to_end(key)
    return verdict


def _record_verdict(key: tuple | None, sources: list[Source], verdict) -> None:
    """Record this plan's reuse verdict. Best-effort."""
    if key is None:
        return
    try:
        held = tuple(weakref.ref(s) for s in sources)
    except TypeError:  # pragma: no cover - a source that cannot be weakly referenced
        return
    with _VERDICTS_LOCK:
        _VERDICTS[key] = (held, verdict)
        _VERDICTS.move_to_end(key)
        while len(_VERDICTS) > _VERDICTS_MAX:
            _VERDICTS.popitem(last=False)


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
    key = _verdict_key(plan, sources, ctx, config, cfg)
    verdict = _known_verdict(key, sources)
    if verdict is None:
        verdict = _analyze(plan, sources, ctx, cfg)
        _record_verdict(key, sources, verdict)
    if not verdict:
        # Hand back the plan as written — including when the analysis built a canonical form
        # to look at. That form is semantically identical, but it is only built to make
        # repeats *visible*, and returning it when nothing repeats would perturb plan
        # identity (the result-cache key, learned-stats signatures) for no gain at all.
        return plan, sources
    nodes = list(walk(plan))
    srcs = list(sources)
    # The estimator bounded each candidate on its own; this bounds what they hold *together*,
    # which is the quantity that actually competes with the running query for memory. Three
    # candidates each just inside a 256 MiB budget is three quarters of a gigabyte held until
    # the query ends. Charged on the materialized size rather than the estimate, so a target
    # whose estimate was optimistic stops the ones after it instead of compounding.
    held = 0
    for positions in verdict:
        appearances = [nodes[i] for i in positions]
        table = _materialize(appearances[0], srcs, ctx)
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
        plan = _replace_all(plan, appearances, Scan(sid, SchemaRef.from_arrow(table.schema)))
        log_kv(
            _log,
            logging.DEBUG,
            "subplan reused",
            rows=table.num_rows,
            bytes=table.nbytes,
            op=type(appearances[0]).__name__,
        )
    return plan, srcs


def _analyze(plan: LogicalPlan, sources: list[Source], ctx, cfg) -> tuple[tuple[int, ...], ...]:
    """Which subtrees to materialize, as pre-order positions in `plan`'s own walk.

    The analysis runs over a **canonical** form of the plan, in which every binding of one
    source object points at that object's first index — the whole reason
    `_one_id_per_source` exists, since two subtrees reading the same table through different
    bindings are otherwise not structurally equal.

    That canonical form is an **analysis artefact and must not be executed**. Collapsing the
    bindings is also what makes `bc_interp::streaming_parallelizes` false — that predicate is
    "no source is scanned twice", and a plan failing it is routed to the *materializing*
    executor for its whole length. Returning the canonical plan to be run therefore changed
    the executor of every query with a table bound more than once, which on a snowflake
    schema is most of them: TPC-DS q80 1,010 -> **151 ms**, q77 482 -> **91 ms**, q5
    473 -> **199 ms** once the executed plan keeps its own source ids. The reuse itself was
    never at fault — q14 (5.5x) and q73 (4.9x) keep their wins either way.

    So the appearances are *located* through the canonical tree and reported as positions in
    the original one. `walk` is pre-order and the two trees differ only in the `source_id`
    **field** of their `Scan`s, so the two walks are the same sequence of nodes and position
    `i` names the same subtree in both — an exact correspondence, not a heuristic.

    Args:
        plan: The plan as written.
        sources: Its bound inputs, positionally.
        ctx: The execution context, for the hub the estimator reads.
        cfg: The optimizer config, for the size budget and the row-width fallback.

    Returns:
        One position tuple per chosen subtree, outermost first; empty when nothing repeats.
    """
    from batcher.api.source_stats import build_estimator
    from batcher.kyber.common_subplan import common_subplans, structural_key

    canonical = _one_id_per_source(plan, sources)
    targets = common_subplans(
        canonical,
        lambda: build_estimator(sources, ctx.hub),
        max_bytes=cfg.common_subplan_max_bytes,
        row_bytes=cfg.row_bytes,
    )
    if not targets:
        return ()
    keys = [structural_key(node) for node in walk(canonical)]
    out: list[tuple[int, ...]] = []
    for target in targets:
        want = structural_key(target)
        if want is None:  # pragma: no cover - an opaque subtree is never a candidate
            continue
        positions = tuple(i for i, k in enumerate(keys) if k == want)
        if positions:
            out.append(positions)
    return tuple(out)


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


def _replace_all(plan: LogicalPlan, appearances: list, repl: LogicalPlan) -> LogicalPlan:
    """Replace every node in `appearances` with `repl`.

    Matched on **object identity**, because `appearances` holds the very nodes of `plan`
    that `_analyze` located, read back by position. Identity is what keeps the rewrite exact
    once the
    canonical form is no longer the thing being rewritten: two original subtrees may be
    equal *canonically* and not equal as written (they scan different bindings of one
    table), and it is precisely those that must both be replaced — which the canonical match
    already established and a structural match on the original plan could not.

    Bottom-up, and `transform_up` returns the same object for an unchanged subtree, so a
    node's identity survives until it is either replaced or one of its descendants is. No
    appearance is a descendant of another (`common_subplans` returns non-overlapping
    subtrees), so every identity in `appearances` is still present when it is reached.
    """
    ids = {id(node) for node in appearances}
    return transform_up(plan, lambda node: repl if id(node) in ids else node)
