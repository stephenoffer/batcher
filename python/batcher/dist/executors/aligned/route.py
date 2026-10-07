"""Choose the alignment key for a plan and route it to the aligned executor, or decline.

`choose_plan` asks, for each key class the plan's equi-joins induce, which of its sources
can be aligned on it (a splittable Parquet source whose files store the key in order) and
whether the plan splits into per-range cuts with every other source read whole. It also
tries each large source alone, split by file with no key, which is what a single fact table
joined to small filtered dimensions needs. The candidate spreading the most bytes across
the fleet wins.

`aligned_route` is the same question without running anything, for the adaptive gate: a
plan this executor will take is not cut into stages first, because a stage boundary between
two aligned sources is exactly the exchange the layout makes unnecessary.
"""

from __future__ import annotations

import operator
from collections import OrderedDict

import pyarrow as pa

from batcher._internal.logging import get_logger, note_suppressed
from batcher.dist.executors.aligned.analysis import (
    AlignedCut,
    AlignedPlan,
    KeyClass,
    find_plan,
    key_classes,
)
from batcher.dist.executors.aligned.rewrite import (
    colocate_aligned_joins,
    distinct_membership_sides,
    group_broadcast_joins,
)
from batcher.dist.executors.aligned.run import run_plan
from batcher.dist.executors.aligned.units import clustered, projected_bytes, source_key_bounds
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["aligned_route", "choose_plan", "scan_plan", "try_aligned"]


def choose_plan(
    plan: LogicalPlan,
    sources: list[Source],
    *,
    strict: bool = True,
    single_table: bool = False,
    exclude: frozenset[int] = frozenset(),
) -> AlignedPlan | None:
    """The aligned plan to run `plan` with, or None when no key supports one.

    `strict` also requires every broadcast *source* to fit the broadcast budget, which is
    the only size a caller that executes nothing can check. The executor passes False: it
    evaluates each broadcast subtree first and bounds the result, which a filter usually
    makes far smaller than its source. `exclude` names sources no cut may broadcast, because a
    run has already found their broadcast too large to hold.
    """
    from batcher.config import active_config
    from batcher.kyber.rules.agg_pushdown import pre_aggregate_facts

    if not active_config().distributed.aligned_execution:
        return None
    # The plan as optimized, and the same plan with each star's facts aggregated beneath its
    # dimensions: the second is what lets TPC-H q10 align at SF1000, where its `customer`
    # (25 GB) cannot be broadcast. It is kept only when it aligns better; on one node it
    # loses, which is why it is asked for here and not by the optimizer.
    best = _best_plan(plan, sources, strict, single_table, exclude)
    # A semi or anti join needs only which keys its other side holds, so that side can be cut
    # to its distinct keys: TPC-H q22's `NOT EXISTS (... orders ...)` over 1.5B orders at
    # SF1000 is 100M customer keys, which the fleet computes and the units then slice. Only a
    # side too large to broadcast as it is: split by file, every unit returns the distinct
    # keys of its own slice, ~1.4M of SF100's 10M customer keys each, and the driver merges
    # them all -- q22 at SF100 took 10 s that way, and 2.8 s left to the shuffle.
    membership = distinct_membership_sides(plan, lambda side: _unbroadcastable(side, sources))
    variants = [pre_aggregate_facts(plan), membership]
    # A dimension the join order put between two aligned tables keeps them out of one cut:
    # TPC-H q10's `lineitem JOIN (customer JOIN orders)`. With the aligned side moved down to
    # the table it keys on, the pair joins per range, and pre-aggregated, the dimension leaves
    # the cut altogether.
    for aligned in {_aligned_on(key, sources) for key in key_classes(plan)}:
        if len(aligned) < 2:
            continue
        moved = colocate_aligned_joins(plan, aligned)
        if moved is not plan:
            # Pre-aggregated first: on a tie it returns partial groups rather than joined rows.
            variants += [pre_aggregate_facts(moved), moved]
    for rewritten in {id(p): p for p in variants}.values():
        if rewritten is plan:
            continue
        other = _best_plan(rewritten, sources, strict, single_table, exclude)
        if other is not None and (best is None or other[0] > best[0]):
            best = other
    return best[1] if best is not None else None


def _unbroadcastable(node: LogicalPlan, sources: list[Source]) -> bool:
    """Whether what `node` reads, projected, is past what every node can hold, or unsized."""
    from batcher.dist.executors.aligned.run import BROADCAST_BYTES
    from batcher.dist.executors.partition_io import source_pushdown
    from batcher.plan.visitor import scanned_source_ids

    sizes = [
        projected_bytes(sources[sid], source_pushdown(node, sid)[0])
        for sid in scanned_source_ids(node)
        if sid < len(sources)
    ]
    return any(size is None for size in sizes) or sum(s or 0 for s in sizes) > BROADCAST_BYTES


def _best_plan(
    plan: LogicalPlan,
    sources: list[Source],
    strict: bool,
    single_table: bool = False,
    exclude: frozenset[int] = frozenset(),
) -> tuple[int, AlignedPlan] | None:
    """The heaviest aligned plan for `plan` over every candidate key, with its weight."""
    from batcher.plan.visitor import scanned_source_ids

    scanned = scanned_source_ids(plan)
    candidates: list[tuple[KeyClass, frozenset[int]]] = []
    for key in key_classes(plan):
        aligned = _aligned_on(key, sources)
        if aligned:
            candidates.append((key, aligned))
    # Split by file: one table and every other scan of it (a relation the query uses twice),
    # each scan in its own cut, split independently -- nothing aligns rows *between* them.
    identity = {sid: _identity(sources[sid]) for sid in scanned if sid < len(sources)}
    seen: set[str] = set()
    for sid in sorted(identity):
        if identity[sid] in seen or getattr(sources[sid], "key_bounds", None) is None:
            continue
        seen.add(identity[sid])
        twins = frozenset(s for s, ident in identity.items() if ident == identity[sid])
        candidates.append((KeyClass.by_file(sid), twins))
    best: tuple[int, AlignedPlan] | None = None
    for key, aligned in candidates:
        found = find_plan(plan, key, aligned, len(sources), single_table=single_table)
        # A plan broadcasting what no node can hold would only decline once it ran; the same
        # plan with those joins left to the residual is the one that can run.
        big = _oversized_broadcasts(found, sources) if found is not None else frozenset()
        big |= exclude - aligned
        if big:
            found = find_plan(
                plan, key, aligned, len(sources), exclude=big, single_table=single_table
            )
        weight = _weigh(found, sources, strict) if found is not None else None
        get_logger("dist").debug(
            "aligned candidate %s aligned %s: %s cut(s), weight %s",
            "by file" if key.keyless else sorted(key.columns),
            sorted(aligned),
            None if found is None else len(found.cuts),
            weight,
        )
        if weight is not None and (best is None or weight > best[0]):
            best = (weight, found)
    return best


def _oversized_broadcasts(found: AlignedPlan, sources: list[Source]) -> frozenset[int]:
    """Broadcast sources no node could hold whole, read with no filter to shrink them."""
    from batcher.dist.executors.aligned.run import BROADCAST_BYTES

    big = set()
    for cut in found.cuts:
        for sid in cut.broadcast - set(found.placeholders):
            if sid >= len(sources):
                continue
            size = projected_bytes(sources[sid], cut.pushdown(cut.body, sid)[0])
            if (
                size is not None
                and size > BROADCAST_BYTES
                and not _filtered(cut.body, sid, cut.aligned, sources[sid])
            ):
                big.add(sid)
    return frozenset(big)


def scan_plan(node: LogicalPlan, sources: list[Source]) -> AlignedPlan | None:
    """`node` split by file of its largest source, every other input read whole, or None.

    For a broadcast input too large to read quickly on the driver that aligns on no key:
    TPC-H's filtered `customer` (1.5 GB at SF1000) and `part` (2 GB) are single scans, and
    the driver read them one after another, 5-13 s of each query before any unit ran. Split
    by file, the fleet reads them in parallel.

    Taken only for filters, projections and inner joins over scans, where the union of the
    per-file results is the whole result: every row of the split source is in exactly one
    unit, and each unit joins it against the whole of every other input.
    """
    from batcher.plan.logical import Filter, Join, Project, Scan
    from batcher.plan.visitor import scanned_source_ids, walk

    for n in walk(node):
        if isinstance(n, Join) and n.join_type != "inner":
            return None
        if not isinstance(n, (Scan, Filter, Project, Join)):
            return None
    scanned = sorted(s for s in scanned_source_ids(node) if s < len(sources))
    sizes = {s: projected_bytes(sources[s], None) or 0 for s in scanned}
    if not sizes:
        return None
    big = max(sizes, key=sizes.__getitem__)
    schema = node.available_schema()
    if getattr(sources[big], "key_bounds", None) is None or schema is None:
        return None
    cut = AlignedCut(
        body=node,
        aggregate=None,
        key=KeyClass.by_file(big),
        aligned=frozenset({big}),
        broadcast=frozenset(scanned) - {big},
    )
    return AlignedPlan((cut,), Scan(len(sources), schema), (len(sources),))


def _identity(source: Source) -> object:
    """The table `source` reads; `identity()` is optional, and without it only the object."""
    return getattr(source, "identity", lambda: id(source))()


#: Copies of a broadcast input the fleet holds: one per node, on a typical cluster.
_BROADCAST_COPIES = 8


def _weigh(found: AlignedPlan, sources: list[Source], strict: bool) -> int | None:
    """The projected bytes `found` spreads across the fleet, or None when it should not run.

    Declined when a broadcast or residual source is unsized, when one is past what every
    worker (broadcast) or the driver (residual) can hold, or when what is held, counted once
    per node, outweighs what is spread: aligning a small dimension while every worker reads
    the fact table whole spreads nothing.
    """
    from batcher.dist.executors.aligned.run import BROADCAST_BYTES, DRIVER_READ_BYTES
    from batcher.dist.executors.partition_io import source_pushdown
    from batcher.plan.visitor import scanned_source_ids

    # What is spread counts each table once, however many scans read it: a plan that reads
    # `lineitem` in three cuts does three passes of the work one cut aligning its three scans
    # does in one. Summed, TPC-H q21 at SF100 took the three-cut plan for 28 GB spread over
    # the one-cut plan's 22 GB, and ran 132 s against 4.8 s.
    spread_by: dict[object, int] = {}
    held = driver = 0
    for cut in found.cuts:
        # A placeholder is an earlier cut's result, which has no size until it has run; an id
        # past the sources is one of an enclosing plan's (a residual being planned itself).
        for sid in (cut.aligned | cut.broadcast) - set(found.placeholders):
            if sid >= len(sources):
                continue
            size = projected_bytes(sources[sid], cut.pushdown(cut.body, sid)[0])
            if size is None:
                return None
            if sid in cut.aligned:
                table = _identity(sources[sid])
                spread_by[table] = max(spread_by.get(table, 0), size)
            elif _filtered(cut.body, sid, cut.aligned, sources[sid]):
                # What is held is the filtered result, which the fleet reads once, in parallel
                # (`scan_plan`), and which is usually a sliver of its source: TPC-H q20 reads
                # 3 GB of `partsupp` and `part` at SF100 to keep 1% of the parts. Charged at its
                # source size once rather than once per node; a result still too large to
                # hold is declined when it has been evaluated.
                driver += size
            else:
                held += size
                if strict and size > BROADCAST_BYTES:
                    return None
    residual_sources = {s for s in scanned_source_ids(found.residual) if s < len(sources)}
    for sid in residual_sources - set(found.placeholders):
        size = projected_bytes(sources[sid], source_pushdown(found.residual, sid)[0])
        if size is None:
            return None
        if size > DRIVER_READ_BYTES and choose_plan(found.residual, sources, strict=strict) is None:
            # Past what the driver reads, unless the residual aligns on its own: TPC-H q10 at
            # SF1000 joins per-customer sums to `customer` (25 GB), stored in `custkey` order.
            return None
        driver += size
    # A broadcast input is held by every node and joined by every unit, so it costs about
    # its size once per node; charge it that way. Split by file, TPC-H q18 would read
    # `lineitem` in place (96 GB spread) but hold all of `orders` on every node, where
    # aligned on `orderkey` it spreads 120 GB and broadcasts only `customer`. A residual input
    # is read once, by the driver or by a residual that aligns itself, and is charged once.
    weight = sum(spread_by.values()) - _BROADCAST_COPIES * held - driver
    return weight if weight > 0 else None


def _filtered(
    body: LogicalPlan, sid: int, aligned: frozenset[int], source: Source | None = None
) -> bool:
    """Whether the broadcast subtree reading `sid` filters it, so its result can be far
    smaller than the source: what is held is that result, bounded when it is evaluated.

    TPC-H q9 at SF1000 reads 9.3 GB of `part` and 13.5 GB of `partsupp` to broadcast 1.4 GB
    of green parts' suppliers; declining on the sources sent the query to the staged loop.
    A subtree with no predicate but `IS NOT NULL` (which join planning adds everywhere) holds
    its whole source, so it is still judged on the source's size. So does one whose ranges
    the source's own footer bounds already imply: Kyber pushes the other join side's key range
    onto a scan, and at SF1000 TPC-H q18's `customer` carried `c_custkey BETWEEN 1 AND
    149999999`, which is every row. Judged as filtered, 4.2 GiB of it was broadcast, overran
    the bound at run time, and the query fell back to a path that took 205 s.
    """
    from batcher.plan.logical import Aggregate, Filter
    from batcher.plan.visitor import children, scanned_source_ids, walk

    def subtree(n: LogicalPlan) -> LogicalPlan | None:
        scanned = scanned_source_ids(n)
        if sid not in scanned:
            return None
        if not (scanned & aligned):
            return n
        for child in children(n):
            found = subtree(child)
            if found is not None:
                return found
        return None

    columns = _footer_columns(source)

    def real(ir: dict) -> bool:
        if ir.get("e") == "is_not_null":
            return False
        if ir.get("e") == "binary" and ir.get("op") == "and":
            return real(ir["left"]) or real(ir["right"])
        return not _implied_by_bounds(ir, columns)

    # After the broadcast joins are grouped, as they are when the cut runs: q9's filter on
    # `part` reaches `partsupp` only once the rewrite has moved the one onto the other.
    found = subtree(group_broadcast_joins(body, aligned))
    # A grouped aggregate reduces as surely as a filter: q22's distinct customer keys are 100M
    # of `orders`' 1.5B rows.
    return found is not None and any(
        (isinstance(n, Filter) and real(n.predicate.to_ir()))
        or (isinstance(n, Aggregate) and n.group_keys)
        for n in walk(found)
    )


def _footer_columns(source: Source | None) -> dict:
    """The per-column statistics `source` declares, or an empty mapping when it has none."""
    if source is None:
        return {}
    try:
        stats = source.statistics()
    except Exception as exc:
        note_suppressed("dist", "read source statistics for a broadcast filter", exc)
        return {}
    return dict(getattr(stats, "columns", None) or {})


#: `column <op> literal` holds for every row when the column's bound on that side does:
#: (which bound, the comparison it must pass).
_IMPLIED_BY = {
    "ge": ("min", operator.ge),
    "gt": ("min", operator.gt),
    "le": ("max", operator.le),
    "lt": ("max", operator.lt),
}


def _implied_by_bounds(ir: dict, columns: dict) -> bool:
    """Whether a `column <op> literal` range conjunct holds for every row of the source.

    Only an integer literal against an integer bound is judged, so no type coercion is
    guessed at; anything else is taken to filter, which is the old, conservative answer.
    """
    if ir.get("e") != "binary" or ir.get("op") not in _IMPLIED_BY:
        return False
    left, right = ir.get("left", {}), ir.get("right", {})
    if left.get("e") != "col" or right.get("e") != "lit":
        return False
    side, passes = _IMPLIED_BY[ir["op"]]
    bound = getattr(columns.get(left.get("name")), side, None)
    value = (right.get("value") or {}).get("int")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (value, bound)):
        return False
    return passes(bound, value)


def _aligned_on(key: KeyClass, sources: list[Source]) -> frozenset[int]:
    """The sources of `key`'s class whose files store its column in order."""
    return frozenset(
        sid
        for sid in {s for s, _ in key.columns}
        if sid < len(sources) and _clustered_on(sources[sid], key.column_of(sid))
    )


def _clustered_on(source: Source, column: str | None) -> bool:
    """Whether `source`'s files store `column` in order, per their footers."""
    if column is None:
        return False
    bounds = source_key_bounds(source, column)
    return bounds is not None and clustered(bounds)


def aligned_route(plan: LogicalPlan, sources: list[Source]) -> bool:
    """Whether `plan` has an aligned plan, without executing anything."""
    try:
        return choose_plan(plan, sources) is not None
    except Exception as exc:  # a routing probe must never fail the query
        get_logger("dist").debug("aligned route probe failed: %s", exc)
        return False


def try_aligned(
    plan: LogicalPlan, sources: list[Source], workers: int, hub=None, metrics_out=None
) -> pa.Table | None:
    """Run `plan` with the aligned executor, or None to let the dispatcher route it.

    A broadcast that outgrows its bound once evaluated declines only the plan that held it:
    the plan is chosen again with that source's join left to the residual, which the other
    executors would otherwise take over whole. TPC-H q18 at SF1000 broadcast 4.2 GiB of
    `customer`, declined, and fell back to a path that took 205 s.
    """
    key = _overran_key(plan, sources)
    exclude = _OVERRAN.get(key, frozenset()) if key is not None else frozenset()
    while True:
        found = choose_plan(plan, sources, strict=False, exclude=exclude)
        if found is None:
            return None
        oversized: set[int] = set()
        result = _run_found(found, sources, workers, hub, metrics_out, oversized)
        if result is not None or not oversized or oversized <= exclude:
            return result
        exclude |= frozenset(oversized)
        if key is not None:
            # Remembered, so the next run of this plan over these tables starts from the plan
            # that ran rather than paying to evaluate and drop the broadcast again (~7 s).
            _OVERRAN[key] = exclude
            while len(_OVERRAN) > _OVERRAN_ENTRIES:
                _OVERRAN.popitem(last=False)
        get_logger("dist").info("aligned: re-planning without broadcasting %s", sorted(exclude))


#: Per (plan, tables): the sources whose broadcast overran its bound on an earlier run.
_OVERRAN: OrderedDict[tuple, frozenset[int]] = OrderedDict()
_OVERRAN_ENTRIES = 256


def _overran_key(plan: LogicalPlan, sources: list[Source]) -> tuple | None:
    """The memo key for `plan` over `sources`, or None when either cannot be keyed."""
    try:
        return (plan.content_key(), tuple(str(_identity(s)) for s in sources))
    except Exception as exc:  # a memo key must never fail the query
        note_suppressed("dist", "key an aligned plan's overrun broadcasts", exc)
        return None


def _run_found(
    found: AlignedPlan, sources: list[Source], workers: int, hub, metrics_out, oversized: set[int]
) -> pa.Table | None:
    """Log and run one chosen plan, adding to `oversized` any source whose broadcast overran."""
    get_logger("dist").info(
        "aligned: plan of %d cut(s) [%s]; residual %s",
        len(found.cuts),
        "; ".join(
            f"{'agg' if c.aggregate is not None else 'rows'} over {type(c.body).__name__}, "
            f"aligned {sorted(c.aligned)}, broadcast {sorted(c.broadcast)}"
            + (" by file" if c.key.keyless else "")
            for c in found.cuts
        ),
        type(found.residual).__name__,
    )
    return _distributed_aligned(found, sources, workers, hub, metrics_out, oversized)


def _distributed_aligned(
    found: AlignedPlan,
    sources: list[Source],
    workers: int,
    hub=None,
    metrics_out=None,
    oversized: set[int] | None = None,
) -> pa.Table | None:
    """The aligned executor's entry point, named as every distributed executor's is."""
    return run_plan(found, sources, workers, hub, metrics_out, oversized)
