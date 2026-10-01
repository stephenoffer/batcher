"""Compute a repeated subplan once and read it back (control plane, `api`).

The seam: `kyber.common_subplan` decides *which* subtrees repeat often enough and cheaply
enough to be worth materializing; this module is the half allowed to act on that. It runs
each chosen subplan, wraps the result as an in-memory source, and rewrites every appearance
of the subtree into a `Scan` over it — the same splice `api.adaptive.staging` performs at a
stage boundary, driven by structure rather than by a measured cardinality.

Keeping the two apart is the same split the adaptive loop uses: the decision stays pure and
testable without a query running, and the execution stays in the layer allowed to execute.

**Where it applies.** The single-node and the distributed relational executors. Not the
**adaptive** route, which already materializes at every breaker and splices by object
identity, so a `Dataset` the caller reused is executed once there by construction; a
second materializing pass in front of it would pay twice for the same thing.

On the distributed route this is a matter of correctness as well as speed. A float
reduction is identical across partitionings only up to reassociation, so a subplan run twice
over the cluster can return two different last bits -- and TPC-H q15, which keeps the
supplier whose `sum` *equals* the `max` of the same `sum`, returned no rows at all. Run once,
both appearances read the same values. The shared result is materialized by a distributed
run, collected to the driver under the same `common_subplan_max_bytes` budget, and broadcast
as an in-memory source; a subplan past that budget is still recomputed per appearance, and
keeping it partitioned on the workers instead (`staging`'s `MaterializedSource`) is the gap
that remains.
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
from batcher.plan.expr_ir import Expr
from batcher.plan.logical import Filter, LogicalPlan, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import derivation_key
from batcher.plan.types import logical_bytes, retained_bytes
from batcher.plan.visitor import children, transform_up, walk

__all__ = ["reuse_common_subplans"]

_log = get_logger("api.subplan_reuse")

#: The analysis's verdict for a plan already analyzed, so a re-issued query does not re-derive
#: it. Bounded, and holding only the decision, never a materialized result (that is
#: `Dataset.cache()`'s job). A verdict is the pre-order positions of each chosen subtree's
#: appearances (empty for "nothing repeats"); the key carries `plan.content_key()`, so an entry
#: is only served to an identical tree, where position `i` is the identical node. Both positive
#: and negative verdicts are cached because the analysis is not cheap: a canonical rebuild, a
#: `structural_key` per node and a `CostModel` pass (404 ms per collect on TPC-DS q80, whose
#: query runs in 151 ms). The key uses Kyber's key builder with `learned=False`, so it follows
#: the plan, config, hub and sources but not the generation counter, which every query in a
#: mixed workload moves. The cost is that a verdict can outlive the estimates it was taken
#: under: a slower query at worst, never a wrong one.
#:
#: Each entry also holds the row filter learned for each of its targets (see
#: `_learn_row_filters`), keyed by the target's index in the verdict. It lives here rather than
#: in its own map so the identity check that guards the verdict guards it too.
_VERDICTS: OrderedDict[
    tuple,
    tuple[tuple[weakref.ref, ...], tuple[tuple[int, ...], ...], dict[int, Expr | None]],
] = OrderedDict()
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
    except Exception as exc:  # an unkeyable plan simply is not cached
        note_suppressed("api", "key the common-subplan verdict", exc)
        return None
    if base is None:
        return None
    return (base, tuple(id(s) for s in sources), _budget_bytes(config), cfg.row_bytes)


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
        verdict = entry[1]
        if not _held_by(entry, sources):
            # A recycled `id()` landed on a different object: drop the entry rather than
            # serve one plan's verdict for another's.
            del _VERDICTS[key]
            return None
        _VERDICTS.move_to_end(key)
    return verdict


def _held_by(entry: tuple, sources: list[Source]) -> bool:
    """Whether a verdict entry was recorded over exactly these source objects.

    A recycled `id()` can land on a different object under the same key, so the entry's weak
    references are compared by identity before anything in it is served.
    """
    held = entry[0]
    return len(held) == len(sources) and all(r() is s for r, s in zip(held, sources, strict=True))


def _record_verdict(key: tuple | None, sources: list[Source], verdict) -> None:
    """Record this plan's reuse verdict. Best-effort."""
    if key is None:
        return
    try:
        held = tuple(weakref.ref(s) for s in sources)
    except TypeError:  # a source that cannot be weakly referenced
        return
    with _VERDICTS_LOCK:
        _VERDICTS[key] = (held, verdict, {})
        _VERDICTS.move_to_end(key)
        while len(_VERDICTS) > _VERDICTS_MAX:
            _VERDICTS.popitem(last=False)


def _budget_bytes(config) -> int:
    """The bytes reuse may hold for one query: the fixed cap, or a share of memory if larger.

    `optimizer.common_subplan_max_bytes` alone is a scale threshold: the same repeated
    subtree fits it at one scale factor and not at the next, and past it the subtree is run
    once per appearance. TPC-DS q67's shared ROLLUP aggregate is 57 MB at sf1 and 572 MB at
    sf10; at sf10 it was materialized, found over 256 MiB, and dropped, and each of the nine
    levels then recomputed it -- 27 s. Holding it costs no more than the aggregate's own
    output already did while it was built, so the cap rises with the hard memory budget
    (`common_subplan_memory_fraction` of it). A cap of `0` still turns reuse off.

    The share is rounded down to a power of two. The hard budget follows live free RAM, and
    the budget is part of the verdict's cache key, so an unrounded share would re-key -- and
    re-analyze -- every plan whenever the page cache moved.

    Args:
        config: The active config.

    Returns:
        The budget in bytes, `0` when reuse is off.
    """
    cfg = config.optimizer
    if cfg.common_subplan_max_bytes <= 0:
        return 0
    share = int(config.spill_budget_bytes() * cfg.common_subplan_memory_fraction)
    return max(cfg.common_subplan_max_bytes, 1 << (share.bit_length() - 1) if share > 0 else 0)


def reuse_common_subplans(
    plan: LogicalPlan, sources: list[Source], ctx, *, distributed: bool = False
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
        distributed: Materialize each shared subplan with a distributed run, as the query
            itself will run.

    Returns:
        The rewritten plan and the sources it is bound to.
    """
    try:
        return _reuse(plan, sources, ctx, distributed)
    except Exception as exc:  # an optimization must never break a query
        note_suppressed("api", "reuse common subplans", exc)
        return plan, sources


def _reuse(
    plan: LogicalPlan, sources: list[Source], ctx, distributed: bool
) -> tuple[LogicalPlan, list[Source]]:
    from batcher.config import active_config
    from batcher.io.source import is_bounded

    config = active_config()
    cfg = config.optimizer
    budget = _budget_bytes(config)
    if budget <= 0:
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
        verdict = _analyze(plan, sources, ctx, cfg, budget)
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
    filters = _row_filters(key, sources)
    unlearned: list[tuple[int, int]] = []
    for index, positions in enumerate(verdict):
        appearances = [nodes[i] for i in positions]
        target = _narrowed(plan, appearances, len(srcs))
        if index in filters:
            if filters[index] is not None:
                target = Filter(target, filters[index])
        else:
            unlearned.append((index, len(srcs)))
        table = _materialize(target, srcs, ctx, distributed)
        if table is None:
            continue
        # `retained_bytes`: a cached subplan is *held* for the query's lifetime, so the
        # figure the budget needs is what it pins, not what its rows address. A result cut
        # zero-copy out of a larger table reports a fraction of what it keeps resident, and
        # the budget would then admit several such tables and hold gigabytes under a
        # 256 MiB cap — the OOM the cap exists to prevent, reached through the cap itself.
        held += retained_bytes(table)
        if held > budget:
            log_kv(_log, logging.DEBUG, "subplan reuse budget reached", held=held, budget=budget)
            break
        sid = len(srcs)
        # No zone maps and `ephemeral`, for the reasons `staging._stage_source` spells out:
        # this relation lives for one query, so an O(rows) min/max pass over it would be
        # recomputed and discarded on every run, and its identity must not seed a
        # distinct-count sketch that outlives it.
        srcs.append(
            InMemorySource(
                _batches(table),
                zone_maps=False,
                ephemeral=True,
                derivation=derivation_key(appearances[0], srcs),
            )
        )
        plan = _replace_all(plan, appearances, Scan(sid, SchemaRef.from_arrow(table.schema)))
        log_kv(
            _log,
            logging.DEBUG,
            "subplan reused",
            rows=table.num_rows,
            bytes=logical_bytes(table),
            op=type(appearances[0]).__name__,
        )
    if unlearned:
        _learn_row_filters(plan, srcs, ctx, key, sources, unlearned)
    return plan, srcs


def _row_filters(key: tuple | None, sources: list[Source]) -> dict[int, Expr | None]:
    """The row filters learned for this verdict's targets, by target index; empty if none."""
    if key is None:
        return {}
    with _VERDICTS_LOCK:
        entry = _VERDICTS.get(key)
        return dict(entry[2]) if entry is not None and _held_by(entry, sources) else {}


def _learn_row_filters(
    plan: LogicalPlan,
    srcs: list[Source],
    ctx,
    key: tuple | None,
    sources: list[Source],
    unlearned: list[tuple[int, int]],
) -> None:
    """Record, for each target materialized whole, the rows its consumers can read at all.

    A shared subplan is materialized standalone, so nothing above it filters it: TPC-DS
    q39 aggregates a year of inventory in its CTE and reads two months of it back, in two
    scans, `d_moy = 1` and `d_moy = 2`. Kyber's pushdown already derives what every consumer of
    the materialized source keeps, as the disjunction of their filters
    (`kyber.rules.projections.scan_predicates`), so this asks it once, over the rewritten plan,
    and the next run materializes `Filter(target, predicate)`. `None` records that some
    consumer reads the target unfiltered, so there is nothing to learn.

    Sound because it is keyed by the verdict: the same plan over the same source objects, so
    the predicate is a superset of what every consumer of *this* query reads, including any
    bound Kyber derived from another input. Best-effort: a failure learns nothing and the
    target stays whole.
    """
    if key is None:
        return
    try:
        from batcher import kyber
        from batcher.kyber.rules.projections import scan_predicates

        found = scan_predicates(kyber.optimize_logical(plan, sources=srcs, hub=ctx.hub))
    except Exception as exc:  # learning a filter must never break a query
        note_suppressed("api", "learn a common-subplan row filter", exc)
        return
    with _VERDICTS_LOCK:
        entry = _VERDICTS.get(key)
        if entry is None or not _held_by(entry, sources):
            return
        for index, sid in unlearned:
            entry[2][index] = found.get(sid)


def _narrowed(plan: LogicalPlan, appearances: list[LogicalPlan], sid: int) -> LogicalPlan:
    """The chosen subtree cut down to the columns the rest of the plan still reads.

    Materializing forfeits the *fusion* each appearance had with its parent, and what that
    costs is a width: embedded, a subtree's consumers prune it through projection pushdown,
    and standalone there is no consumer to prune it — so every column it carries is built,
    held for the query, and re-scanned once per appearance. Measured on TPC-DS, the chosen
    subtree against what the plan reads of it: q18 **133 columns against 11**, q86 **84
    against 3**, and q80 5 against 5, q70 1 against 1, q14 6 against 6. Those are exactly
    the two that lost and the three that won — q18 by 640 ms and q86 by 118, against wins
    of 78 ms on q14 and 41 on q70 — so the waste is the whole of the difference.

    The need is not recomputed here. The hypothetical rewrite is built (a `Scan` of the
    subtree's own schema in place of every appearance) and the optimizer's own
    need-propagation is asked what that scan must read, so the one definition of "which
    columns does this plan require" stays in `kyber.rules.projections`. A subtree with no
    static schema, or one the pass cannot narrow, is returned unchanged.

    Args:
        plan: The plan being rewritten, as it currently stands.
        appearances: Every occurrence of the chosen subtree within it.
        sid: The source id the materialized result will take.

    Returns:
        The subtree, wrapped in a `Project` when that drops a column and unchanged otherwise.
    """
    from batcher.kyber.rules.projections import required_columns_per_source
    from batcher.plan.expr_ir import col
    from batcher.plan.logical import Project, Projection

    target = appearances[0]
    try:
        schema = target.available_schema()
        if schema is None:
            return target
        # Deliberately the need of the plan **as written**, not of its optimized form, which
        # is the opposite of what the size gate wants and for a reason that is easy to walk
        # into. Running projection pushdown first reports a *smaller* need -- a column a
        # projection merely passes through stops counting -- but the plan being rewritten
        # still names that column above the appearance, so a `Scan` without it does not
        # validate. Measured: asking the pushed form for a shared SELECT carrying an unread
        # column returned ['k', 'v'] against ['k', 'v', 'unused'], and the narrower scan
        # raised, so `reuse_common_subplans` caught it and declined the reuse altogether --
        # turning a saving into nothing at all. The need as written is exactly the set the
        # surrounding plan references, which is the set that keeps it valid.
        hypothetical = _replace_all(plan, appearances, Scan(sid, schema))
        wanted = required_columns_per_source(hypothetical).get(sid)
        carried = list(target.available_columns())
        if wanted is None or len(wanted) >= len(carried):
            return target
        keep = [c for c in carried if c in set(wanted)]
        if not keep or len(keep) >= len(carried):
            return target
        return Project(target, tuple(Projection(c, col(c)) for c in keep))
    except Exception as exc:  # narrowing must never break a query
        note_suppressed("api", "narrow a common-subplan candidate", exc)
        return target


def _analyze(
    plan: LogicalPlan, sources: list[Source], ctx, cfg, budget: int | None = None
) -> tuple[tuple[int, ...], ...]:
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
        cfg: The optimizer config, for the row-width fallback.
        budget: The size budget; `_budget_bytes` of the active config when omitted.

    Returns:
        One position tuple per chosen subtree, outermost first; empty when nothing repeats.
    """
    from batcher.api.source_stats import build_estimator
    from batcher.kyber.common_subplan import common_subplans, structural_key

    if budget is None:
        from batcher.config import active_config

        budget = _budget_bytes(active_config())
    canonical = _one_id_per_source(plan, sources)
    targets = common_subplans(
        canonical,
        lambda: build_estimator(sources, ctx.hub),
        max_bytes=budget,
        row_bytes=cfg.row_bytes,
        normalize=lambda node: _as_run(node, sources, ctx),
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


def _as_run(node: LogicalPlan, sources: list[Source], ctx) -> LogicalPlan:
    """`node` as the engine will run it, for the analysis's size and cost gates.

    Those two gates ask how big a subtree's result is and what it costs, and the plan as
    written answers neither: a `WHERE` clause sits above the join tree until pushdown moves
    it, so every join under it is estimated over unfiltered inputs and the error compounds
    across a multi-way join. See `common_subplans`' `normalize` argument for the counts and
    the measured sizes -- on TPC-DS q80 they differ by more than ten orders of magnitude,
    which is the difference between "too large to hold" and 66 kilobytes.

    Only the *gates* read this form. The structure being matched, and the positions handed
    back to be rewritten, are still the plan as written.

    Failing to optimize a subtree is not a reason to fall back to the raw estimate and
    admit it: without a trustworthy size there is no bound on what materializing would
    hold, so the raw node is returned and its own (astronomical) estimate declines it --
    the same direction `_fits` takes for a missing estimate.

    Args:
        node: A subtree of the canonical plan, or the plan itself.
        sources: The plan's bound inputs, for the optimizer's statistics.
        ctx: The execution context, for the hub the optimizer reads.

    Returns:
        The optimized logical form, or `node` unchanged if optimizing it failed.
    """
    from batcher.kyber.optimizer.facade import optimize_logical

    try:
        return optimize_logical(node, sources=sources, hub=ctx.hub)
    except Exception as exc:
        note_suppressed("api", "optimize a common-subplan candidate for sizing", exc)
        return node


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


def _materialize(
    target: LogicalPlan, sources: list[Source], ctx, distributed: bool
) -> pa.Table | None:
    """Run one shared subplan, or `None` if it cannot be run on this path.

    The subplan is executed with the caller's own context so it reads the same hub and
    already-collected source statistics — but with the *subplan's* output columns, since
    `ctx.columns` names the root's, and with the result cache off: caching an intermediate
    the user never asked for would spend the cache's budget on something no later query
    asks for by name.

    On the distributed route the subplan goes where ``distributed="auto"`` would send it:
    all it needs is to be computed *once*, and a small one is faster on the driver than
    through a ~1 s cluster round trip (TPC-H q15 at SF1, 1.9 s -> 0.6 s).

    A distributed run that fails is retried single-node rather than given up. Declining
    reuse is harmless on one node, but across the cluster it is not: the appearances would
    each be recomputed, a float reduction can then differ in its last bits between them, and
    TPC-H q15 returned no rows when a cold fleet made the first materialization time out.
    """
    from batcher.api.orchestration.run import run_relational
    from batcher.api.terminal.routing import resolve_distributed

    sub_ctx = dataclasses.replace(ctx, columns=target.available_columns(), cache=False)
    cluster = distributed and resolve_distributed("auto", target, sources)
    for on_cluster in (True, False) if cluster else (False,):
        try:
            table, _decisions = run_relational(target, sources, sub_ctx, distributed=on_cluster)
        except Exception as exc:  # fall back to the next route, then to recomputing in place
            if on_cluster:
                _log.warning(
                    "a shared subplan failed to materialize on the cluster (%s); "
                    "materializing it on the driver instead",
                    exc,
                )
            else:
                note_suppressed("api", "materialize a shared subplan", exc)
            continue
        return table if isinstance(table, pa.Table) else None
    return None


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
