"""Sideways information passing — the filters a join implies about its other side.

A join already *knows* something about the rows its other side can possibly match. Each rule here
turns that into a **superset filter** on the opposite input: a predicate that may only remove rows
which provably cannot match, and is therefore free to sink all the way to the source, where a zone
map or a bloom skips whole row groups. `rules.joins.runtime_join_filter` opens the family (a key's
`[min, max]` range); these rules carry it the rest of the way — the implied `IS NOT NULL`, the
mirrored `IN` list, and the members the other side's range or bloom refutes.

`FILTERABLE_SIDES` is the law for which side each join type may reduce, and every rule that
inserts or narrows a filter routes through it. Everything else is in `evidence.py`; nothing here
re-derives a proof.

These run in **PUSHDOWN** (see `evidence.SIP` for why that is a correctness requirement and not a
preference) — which also means the shipped pushdown rewrites sink the inserted filter to the
`Scan` for free, and the cost model sees it before it picks a join order.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule

# Registering `skipping` first is load-bearing, not cosmetic: within a phase, registration order is
# run order, and `empty_join_from_all_null_key` must read a side's EXACT null count *before*
# `push_is_not_null_from_join_key` wraps that side in a `Filter` — which sets `null_count` to
# unknown and downgrades provenance away from EXACT, erasing the very evidence the proof needs.
from batcher.kyber.rules.extra.runtime_filters import skipping as _skipping  # noqa: F401
from batcher.kyber.rules.extra.runtime_filters.evidence import (
    FILTERABLE_SIDES,
    SIP,
    _add_conjuncts,
    _already_empty,
    _bloom_refutes,
    _filter_chain,
    _may_hold_null,
    _membership_values,
    _out_of_range,
    _provably_true_at_source,
    _real_key_pairs,
    _rebuild,
    _rebuild_membership,
    _value_set,
)
from batcher.plan.expr_ir import Col, Expr, InList, IsNotNull, Lit
from batcher.plan.expr_rewrite import combine_conjuncts, expr_key, split_conjuncts
from batcher.plan.logical import AsofJoin, Filter, Join, Limit, LogicalPlan, Project, Scan
from batcher.plan.stats import ColumnStat

__all__ = [
    "dedup_source_predicates",
    "prune_asof_right_by_on_bound",
    "prune_join_side_in_list_by_other_side_bloom",
    "prune_join_side_in_list_by_other_side_range",
    "push_in_list_across_join_keys",
    "push_is_not_null_from_asof_on_key",
    "push_is_not_null_from_join_key",
]

# A mirrored `IN` list is probed per row on the receiving side, so it must stay small enough that
# the rows it skips repay the probe; beyond this the key's `[min, max]` range is the cheaper
# carrier of the same information.
_MAX_MIRRORED_IN_LIST = 32


@rule(name="push_is_not_null_from_join_key", matches=(Join,), **SIP)
def push_is_not_null_from_join_key(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """Push `key IS NOT NULL` onto every join side the join does not preserve.

    An equi-join never matches a NULL key — `NULL = NULL` is NULL, not TRUE, and the engine honors
    it. So on a side whose *unmatched* rows are not required in the output, a null-keyed row is
    dead on arrival and may be removed before the join even builds: a superset filter by
    construction, the cheapest predicate there is, and — once pushdown sinks it to the `Scan` — one
    a Parquet/lakehouse source answers from its null counts alone.

    `FILTERABLE_SIDES` decides which sides qualify, and it is exactly right here: an anti join's
    null-keyed left rows have *no match* and therefore **must** survive; an outer join's preserved
    side likewise; a `full` join preserves both and gets nothing. A multi-key join implies it for
    *every* key (all keys must be equal, hence all non-null). Skipped for a cartesian pseudo-key
    and for a key whose null count is known to be zero — in both, the filter could never fire, so
    it would be pure per-row cost.
    """
    sides = FILTERABLE_SIDES.get(node.join_type)
    if sides is None or not node.left_keys:
        return None
    left_stats = ctx.estimator.estimate(node.left)
    right_stats = ctx.estimator.estimate(node.right)
    left_preds: list[tuple[str, Expr]] = []
    right_preds: list[tuple[str, Expr]] = []
    for lk, rk in _real_key_pairs(node):
        # `_may_hold_null` is read here, at the join — but a `Filter` between here and the scan
        # has already set `null_count` to unknown, so it says "maybe" for a column the scan
        # proves is null-free. `_provably_true_at_source` asks the same oracle down at the scan,
        # where the predicate would land and where `drop_filter_conjunct_implied_by_zonemap`
        # would then delete it as a tautology — leaving this rule to add it again, forever.
        # Both questions must be asked in the same place or the two rules cannot converge.
        if (
            "left" in sides
            and _scan_rooted(node.left)
            and _may_hold_null(left_stats.column(lk))
            and not _provably_true_at_source(node.left, lk, IsNotNull(Col(lk)), ctx)
        ):
            left_preds.append((lk, IsNotNull(Col(lk))))
        if (
            "right" in sides
            and _scan_rooted(node.right)
            and _may_hold_null(right_stats.column(rk))
            and not _provably_true_at_source(node.right, rk, IsNotNull(Col(rk)), ctx)
        ):
            right_preds.append((rk, IsNotNull(Col(rk))))
    left = _add_conjuncts(node.left, left_preds)
    right = _add_conjuncts(node.right, right_preds)
    return _rebuild(node, left, right, ctx, "is_not_null")


def _scan_rooted(side: LogicalPlan) -> bool:
    """Whether `side` is a plain chain down to one `Scan` — the only side worth filtering.

    The engine already drops null-keyed rows inside the join, so this predicate buys nothing
    *at* the join; its entire value is that pushdown then sinks it into the `Scan`, where a
    Parquet/lakehouse source answers it from its null counts and skips whole row groups. On a
    side that is not scan-rooted — a join, an aggregate, a union — there is no scan to sink
    into, so the filter is pure per-row cost.

    It is also actively harmful there. Wrapping a join's operand in a `Filter` changes that
    operand's structural signature (so the learned cardinality for the shape no longer
    matches) and downgrades its provenance away from EXACT, and the inflated estimate that
    follows was enough to route TPC-H q2 — an ordinary 5-table query — out-of-core: 535 ms
    on the spill path against 24 ms in memory. A rewrite that can only help at a scan should
    only fire at a scan.
    """
    node = side
    while isinstance(node, Filter | Project | Limit):
        node = node.input
    return isinstance(node, Scan)


@rule(name="push_in_list_across_join_keys", matches=(Join,), **SIP)
def push_in_list_across_join_keys(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """Mirror a key's `IN` list across the equi-key correspondence onto the reducible side.

    Every matching pair has equal keys, so a key constrained to a finite set of literals on one
    side must lie in that same set on any *matching* row of the other. Pushing the membership onto
    the opposite input therefore removes only provably non-matching rows — and it is far sharper
    than the `[min, max]` range `runtime_join_filter` mirrors, which keeps every value *between*
    the listed ones.

    This is the `IN`-list half of `infer_join_predicates`, which mirrors only `key OP literal`
    comparisons and only across an **inner** join; `FILTERABLE_SIDES` extends the same inference to
    semi/anti/left/right joins, on the side each may reduce. Capped at `_MAX_MIRRORED_IN_LIST`
    values, and skipped for a singleton (which `infer_join_predicates` already mirrors as an
    equality).
    """
    sides = FILTERABLE_SIDES.get(node.join_type)
    if sides is None or not node.left_keys:
        return None
    new_left, new_right = node.left, node.right
    # `_scan_rooted` for the same reason the null-key rule needs it: this inserts exactly the
    # same kind of `Filter` on exactly the same operands, and on a side that is not scan-rooted
    # there is no `Scan` for it to sink into — so it is pure per-row cost *and* it changes the
    # operand's structural signature and provenance, which is what routed TPC-H q2 out-of-core.
    for lk, rk in _real_key_pairs(node):
        if "right" in sides and _scan_rooted(node.right):
            new_right = _add_conjuncts(new_right, _mirrored_in_list(node.left, lk, rk))
        if "left" in sides and _scan_rooted(node.left):
            new_left = _add_conjuncts(new_left, _mirrored_in_list(node.right, rk, lk))
    return _rebuild(node, new_left, new_right, ctx, "in_list")


def _mirrored_in_list(
    source: LogicalPlan, source_key: str, target_key: str
) -> list[tuple[str, Expr]]:
    """`target_key IN (<source_key's value set>)`, or `[]` when unbounded/singleton/too wide."""
    values = _value_set(source, source_key)
    if values is None or not 2 <= len(values) <= _MAX_MIRRORED_IN_LIST:
        return []
    # `repr` orders any mixed literal set deterministically, so the conjunct is stable across runs
    # and the idempotence check (by structural key) recognises it again.
    return [(target_key, InList(Col(target_key), tuple(sorted(values, key=repr))))]


@rule(name="prune_join_side_in_list_by_other_side_bloom", matches=(Join,), **SIP)
def prune_join_side_in_list_by_other_side_bloom(
    node: Join, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop `IN`-list members on a reducible join side that the other side's key bloom refutes.

    The plan-level form of the runtime bloom filter. A build-side membership index cannot be
    *expressed* as an `Expr` (the IR has no bloom-probe node), but it can still delete key values
    from a probe-side `IN` list: a member absent from the other side's key bloom matches no row
    there, so the probe rows holding it produce nothing and may be removed. Absence is a proof, not
    a guess — and it is the proof that finds a value *inside* `[min, max]`, which the range filter
    cannot.

    Only the sides `FILTERABLE_SIDES` allows are pruned (removing a preserved side's rows would
    delete answers), every probe is domain-guarded, and when *no* member survives, the side becomes
    the canonical empty marker. The general runtime bloom over all build keys lives where it can
    actually be built — the distributed join executor's `runtime_bloom_join`.
    """
    return _prune_by_other_side(node, ctx, _bloom_refutes, "bloom")


@rule(name="prune_join_side_in_list_by_other_side_range", matches=(Join,), **SIP)
def prune_join_side_in_list_by_other_side_range(
    node: Join, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop `IN`-list members on a reducible join side that fall outside the other side's key range.

    The zone-map sibling of the bloom rule, and the sharper form of what `runtime_join_filter`
    does: that rule *adds* a `BETWEEN` conjunct, which still evaluates on every row; this one
    deletes the unreachable members from the membership test the query already has, so the probe
    gets smaller instead of longer. `WHERE f.k IN (1, 5, 900)` joined to a dimension whose key
    tops out at 100 is `f.k IN (1, 5)`.

    Bounds refute at any provenance (a row-shrinking operator only narrows the true range, so a
    value outside the recorded `[min, max]` is absent from the real one too), and an emptied list
    means the side cannot match at all.
    """
    return _prune_by_other_side(node, ctx, _out_of_range, "range")


def _prune_by_other_side(node: Join, ctx: OptimizerContext, refutes, note: str):
    """Narrow each reducible side's key membership by what the *other* side's stats refute."""
    sides = FILTERABLE_SIDES.get(node.join_type)
    if sides is None or not node.left_keys:
        return None
    left_stats = ctx.estimator.estimate(node.left)
    right_stats = ctx.estimator.estimate(node.right)
    new_left, new_right = node.left, node.right
    for lk, rk in _real_key_pairs(node):
        if "left" in sides:
            new_left = _prune_key_members(new_left, lk, right_stats.column(rk), refutes)
        if "right" in sides:
            new_right = _prune_key_members(new_right, rk, left_stats.column(lk), refutes)
    return _rebuild(node, new_left, new_right, ctx, note)


def _prune_key_members(side: LogicalPlan, key: str, other: ColumnStat, refutes) -> LogicalPlan:
    """`side` with every membership constraint on `key` narrowed to what `other` still allows.

    Rewrites the `Filter` chain in place (the shape `split_expensive_filter` may have chosen is
    preserved), keeping each constraint in the spelling it arrived in — an `InList` node or the
    OR-chain of equalities `is_in` lowers to. Monotone: it only ever removes members, so re-running
    it finds nothing left to remove, which is what makes it safe in a fixpoint phase.
    """
    if _already_empty(side):
        return side  # nothing to narrow, and re-wrapping an empty side never converges
    chain = _filter_chain(side)
    if not chain:
        return side
    levels: list[list[Expr]] = []
    changed = False
    for node in chain:
        kept: list[Expr] = []
        for conj in split_conjuncts(node.predicate):
            members = _membership_values(conj, key)
            if members is None:
                kept.append(conj)
                continue
            survivors = tuple(v for v in members if not refutes(other, v))
            if len(survivors) == len(members):
                kept.append(conj)
                continue
            changed = True
            if not survivors:
                return Limit(side, 0)  # no key value can match → this side is dead
            kept.append(_rebuild_membership(conj, key, survivors))
        levels.append(kept)
    if not changed:
        return side
    out = chain[-1].input
    for kept in reversed(levels):
        if kept:
            out = Filter(out, combine_conjuncts(kept))
    return out


@rule(name="push_is_not_null_from_asof_on_key", matches=(AsofJoin,), **SIP)
def push_is_not_null_from_asof_on_key(node: AsofJoin, ctx: OptimizerContext) -> LogicalPlan | None:
    """Push `right_on IS NOT NULL` onto an ASOF join's right (nearest-match) input.

    `bc_runtime::join::asof` skips a right row with a null `on` key outright — it is never a
    candidate for any left row's nearest match — so removing it before the join changes no output
    row. The right side is the *only* one that may be reduced: an ASOF join is left-style (every
    left row is emitted, null-extended when unmatched), so a left row with a null `on` must survive
    even though it can never match.

    The `by` keys are deliberately **not** given this treatment. The ASOF implementation groups them
    by their Arrow row encoding, so a null `by` key *matches* a null `by` key — unlike the
    equi-join's SQL null semantics. Dropping null-`by` right rows would delete real matches.
    """
    # Both oracles, exactly as the equi-join version asks them (see the join rule above):
    # `_may_hold_null` reads the stats *here*, but a `Filter` between this join and the scan sets
    # `null_count` to unknown, so it answers "maybe" for a column the source proves null-free.
    # `_provably_true_at_source` asks down at the scan. Without it the tautological filter is added,
    # sunk, deleted at the scan, and re-added on the next fixpoint iteration — the ping-pong that
    # cost 16 iterations on TPC-H q3, 24 on q5 and 25 on q7 before the join version was guarded.
    if not _may_hold_null(ctx.estimator.estimate(node.right).column(node.right_on)):
        return None
    if _provably_true_at_source(node.right, node.right_on, IsNotNull(Col(node.right_on)), ctx):
        return None
    if not _scan_rooted(node.right):
        return None  # nothing to sink into; see `_scan_rooted`
    new_right = _add_conjuncts(node.right, [(node.right_on, IsNotNull(Col(node.right_on)))])
    if new_right is node.right:
        return None
    return dataclasses.replace(node, right=new_right)


@rule(name="prune_asof_right_by_on_bound", matches=(AsofJoin,), **SIP)
def prune_asof_right_by_on_bound(node: AsofJoin, ctx: OptimizerContext) -> LogicalPlan | None:
    """Bound an ASOF join's right `on` column by the left's — the direction decides which end.

    A *backward* ASOF picks the largest `right.on <= left.on`, so a right row whose `on` exceeds
    the largest `left.on` can never be chosen by any left row: push `right_on <= max(left_on)`. A
    *forward* ASOF picks the smallest `right.on >= left.on`, so the mirror bound applies. Only that
    one end is sound — a right row *below* a backward join's range is still a legitimate nearest
    match, and pushing a lower bound there would delete it.

    Bounds are valid at any provenance (a row-shrinking operator only narrows the true range, so
    the recorded bound still contains every left value), and the filter fires only when the right
    side actually extends past it — otherwise it would be pure per-row cost.

    A ``"nearest"`` join gets **no** bound, and the distinction is load-bearing rather than
    conservative: it may take its match from either side, so a right row above the largest
    `left.on` is still the forward candidate for that left row, and one below the smallest is
    still the backward candidate for that one. Neither end is prunable. Reading the direction as
    "backward, or else forward" silently pushed the forward bound onto a nearest join and deleted
    exactly the rows it should have matched.
    """
    if node.direction not in ("backward", "forward"):
        return None
    left = ctx.estimator.estimate(node.left).column(node.left_on)
    right = ctx.estimator.estimate(node.right).column(node.right_on)
    backward = node.direction == "backward"
    bound = left.max if backward else left.min
    reach = right.max if backward else right.min
    if bound is None or reach is None:
        return None
    try:
        if not (reach > bound if backward else reach < bound):
            return None  # the right side already lies inside the left's reach
    except TypeError:
        return None  # incomparable bound types → leave the join untouched
    col = Col(node.right_on)
    pred = (col <= Lit(bound)) if backward else (col >= Lit(bound))
    new_right = _add_conjuncts(node.right, [(node.right_on, pred)])
    if new_right is node.right:
        return None
    return dataclasses.replace(node, right=new_right)


@rule(name="dedup_source_predicates", matches=(Filter,), **SIP)
def dedup_source_predicates(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a conjunct that another `Filter` on the same stack already enforces (`X AND X ≡ X`).

    Everything stacked over one `Scan` is AND-combined into a single source predicate
    (`required_predicates_per_source`), so a repeated conjunct is handed to the source twice and
    evaluated twice on every row. `merge_adjacent_filters` fuses `.filter(p).filter(p)` into
    `Filter(p AND p)` in this same phase — *after* `remove_duplicate_conjuncts` (NORMALIZE) has run
    for the last time — and the runtime filters here stack onto sides that may already carry the
    same constraint.

    The deepest occurrence is kept (so the constraint still reaches the scan) and the stack's shape
    is otherwise preserved, which is what keeps a cost-based split intact. Only *structurally
    identical* conjuncts are removed, and removing them is monotone, so the rule is idempotent.
    """
    chain = _filter_chain(node)
    seen: set[str] = set()
    levels: list[list[Expr]] = []
    changed = False
    for level in reversed(chain):  # innermost first — keep the occurrence nearest the scan
        kept: list[Expr] = []
        for conj in split_conjuncts(level.predicate):
            key = expr_key(conj)
            if key in seen:
                changed = True
                continue
            seen.add(key)
            kept.append(conj)
        levels.append(kept)
    if not changed:
        return None
    out = chain[-1].input
    for kept in levels:
        if kept:
            out = Filter(out, combine_conjuncts(kept))
    return out
