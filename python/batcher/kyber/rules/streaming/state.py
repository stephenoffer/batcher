"""Streaming rule family: state minimization — shrink what a streaming operator retains.

`watermark.py` pushes a *whole* predicate through the two watermark-bounded operators.
This module is the other half of that lane: the rewrites that make the retained state
itself smaller — a narrower dedup key, fewer rows in a join buffer, a conjunction split so
the part that *can* cross does.

Under a bounded input every rule here saves CPU on rows that were going to be discarded
anyway — a micro-optimization. Under a stream the same rewrites change the *memory
bound*: a `WatermarkDedup` holds one seen-key entry per live key and a
`WatermarkStreamJoin` holds both sides' buffers, those are the working set of a query
that never ends, and `api/terminal/stream/watermark.py::_check_stream_state` fails the
query outright once they outgrow `memory.streaming_state_max_bytes`.

Two properties every rule here holds, because both were nearly violated while writing it:

- **The node type at the root is preserved.** The streaming driver dispatches by
  `isinstance` and falls back to the *unoptimized* plan when a rewrite changes the shape
  it expects (`_optimized_streaming_node`). Hoisting a `Project` above a `WatermarkDedup`
  is perfectly sound and still a net loss, because it discards every other optimization
  along with itself. Rules here rewrite a streaming node into the same streaming node, or
  rewrite a `Filter` above one.
- **Row-set-preserving rewrites are preferred over row-removing ones.** The dedup driver
  advances its watermark from the *max event time of the batch it is handed* and drops
  rows below it, so removing rows below a dedup can lower the watermark and change which
  rows count as late. The four subset-narrowing rules therefore change no row at all —
  they only shrink the key. `split_filter_conjuncts_through_watermark_dedup` does remove
  rows, and carries the same proof and the same watermark caveat as the shipped
  `push_filter_through_watermark_dedup` it generalizes.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.expr_ir import (
    Binary,
    Cast,
    Col,
    Expr,
    InList,
    IsNotNull,
    Lit,
    referenced_columns,
    remap_columns,
)
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts
from batcher.plan.logical import (
    Filter,
    LogicalPlan,
    Project,
    WatermarkDedup,
    WatermarkStreamJoin,
)

__all__ = [
    "add_stream_join_key_null_rejection",
    "deduplicate_stream_join_keys",
    "deduplicate_watermark_dedup_subset",
    "drop_constant_watermark_dedup_key",
    "drop_watermark_dedup_key_determined_by_other_keys",
    "drop_watermark_dedup_key_pinned_by_equality_filter",
    "split_filter_conjuncts_into_stream_join_sides",
    "split_filter_conjuncts_through_watermark_dedup",
]


def _child_projections(node: WatermarkDedup) -> dict[str, Expr] | None:
    """The dedup child's `alias -> expr` map when that child is a `Project`, else None."""
    child = node.input
    if not isinstance(child, Project):
        return None
    return {item.alias: item.expr for item in child.items}


def _is_row_local(expr: Expr) -> bool:
    """Whether `expr`'s value depends on nothing but the row it is evaluated on.

    The proof that a derived column is *functionally determined* by the columns it reads
    needs the derivation to be a function of the row — a window function reads its whole
    partition and an aggregate its whole group, so neither qualifies, and treating one as
    determined would drop a key that genuinely splits rows apart. Rather than enumerate
    what is unsafe (a list that goes stale every time an expression node is added), this
    whitelists the shapes that are provably row-local — a literal, a column, a cast, and a
    binary operator over those — so a new node kind defaults to "not proven".
    """
    if isinstance(expr, (Lit, Col)):
        return True
    if isinstance(expr, Cast):
        return _is_row_local(expr.input)
    if isinstance(expr, Binary):
        return _is_row_local(expr.left) and _is_row_local(expr.right)
    return False


def _pinned_columns(predicate: Expr) -> set[str]:
    """The columns a predicate pins to one non-null constant for every surviving row.

    A conjunct `col = <literal>` admits only rows whose column equals that literal; SQL
    three-valued logic rejects a NULL there (``NULL = 5`` is NULL, not true), so the
    column is a genuine constant — not "one value plus NULLs" — among the surviving rows.
    A single-element `IN` list is the same statement spelled differently.
    """
    pinned: set[str] = set()
    for conjunct in split_conjuncts(predicate):
        if isinstance(conjunct, Binary) and conjunct.op == "eq":
            for a, b in ((conjunct.left, conjunct.right), (conjunct.right, conjunct.left)):
                if isinstance(a, Col) and isinstance(b, Lit):
                    pinned.add(a.name)
        elif (
            isinstance(conjunct, InList)
            and isinstance(conjunct.input, Col)
            and len(conjunct.values) == 1
        ):
            pinned.add(conjunct.input.name)
    return pinned


def _narrowed(node: WatermarkDedup, subset: tuple[str, ...]) -> LogicalPlan | None:
    """The dedup rebuilt on `subset`, or None when that would be a no-op or empty.

    A dedup with no key at all would collapse the whole stream to its first row, so an
    empty result is refused rather than emitted — every narrowing rule shares this guard.
    """
    if not subset or len(subset) >= len(node.subset):
        return None
    return dataclasses.replace(node, subset=subset)


@rule(
    name="deduplicate_watermark_dedup_subset",
    phase=Phase.NORMALIZE,
    matches=(WatermarkDedup,),
    category=RuleCategory.REWRITE,
)
def deduplicate_watermark_dedup_subset(
    node: WatermarkDedup, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Collapse a repeated column name in a dedup `subset` to a single occurrence.

    A key is a *set* of columns: two rows agree on ``(a, a, b)`` exactly when they agree
    on ``(a, b)``, so the partition into key groups — and every row the dedup emits — is
    identical, while the seen-key state stores one column fewer per live key. It needs
    nothing but the subset itself, which is why it runs in NORMALIZE: the three rules that
    reason about *where* a key's values come from are cheaper once repetition is gone.

    Args:
        node: The `WatermarkDedup` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The dedup with a de-duplicated subset, or None when the subset is already a set.
    """
    seen: list[str] = []
    for name in node.subset:
        if name not in seen:
            seen.append(name)
    return _narrowed(node, tuple(seen))


@rule(
    name="drop_constant_watermark_dedup_key",
    phase=Phase.REWRITE,
    matches=(WatermarkDedup,),
    category=RuleCategory.REWRITE,
)
def drop_constant_watermark_dedup_key(
    node: WatermarkDedup, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop a dedup key the child projection defines as a literal.

    ``WatermarkDedup(Project(x, [k := col, tag := lit("eu")]), subset=("k", "tag"))``
    keys on a column that holds the same value in every row. A constant column cannot
    separate two rows, so ``(k, tag)`` and ``(k,)`` induce the identical partition and the
    dedup emits the identical rows — including under NULL, since a literal NULL is equally
    constant and all rows land in the one NULL group.

    The saving is real: the constant is stored once per *live key* in the seen-key table,
    and a partition tag pinned by a source (one Kafka topic, one region) is exactly how
    such a column gets into a key. Row-set-preserving — the `Project` below is untouched,
    so the batch the dedup sees, and the watermark from its max event time, are unchanged.

    Args:
        node: The `WatermarkDedup` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The dedup keyed on its non-constant columns, or None.
    """
    defs = _child_projections(node)
    if defs is None:
        return None
    kept = tuple(name for name in node.subset if not isinstance(defs.get(name), Lit))
    return _narrowed(node, kept)


@rule(
    name="drop_watermark_dedup_key_determined_by_other_keys",
    phase=Phase.REWRITE,
    matches=(WatermarkDedup,),
    category=RuleCategory.REWRITE,
)
def drop_watermark_dedup_key_determined_by_other_keys(
    node: WatermarkDedup, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop a dedup key that is a row-local function of the keys kept beside it.

    If key ``b`` is defined as ``f(a)`` for a row-local ``f`` and ``a`` is itself a key,
    then any two rows agreeing on ``a`` also agree on ``b``, so ``(a, b)`` and ``(a,)``
    induce the same partition and the dedup emits the same first row per group. The common
    shapes are a rename (``b := col("a")``) and a cast or unit conversion
    (``cents := amount * 100``) a user adds for readability and then includes in the key.

    Two guards carry the proof. `_is_row_local` whitelists the derivations that are a
    function of the row alone — a window function reads its whole partition and would not
    be determined by ``a`` at all. And a key is only tested against the keys **already
    kept** as the subset is scanned left to right, never against keys still under
    consideration: without that, ``b := col("a")`` and ``a := col("b")`` would each be
    judged determined by the other and both dropped, leaving a keyless dedup.

    A determining key must be a *bare* column reference: ``k := a + 1`` does determine
    ``b := a + 1``, but proving it needs expression equality rather than column
    containment, and this rule declines rather than reach for it.

    Args:
        node: The `WatermarkDedup` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The dedup keyed on its determining columns, or None.
    """
    defs = _child_projections(node)
    if defs is None:
        return None
    kept: list[str] = []
    determining: set[str] = set()
    for name in node.subset:
        expr = defs.get(name)
        reads = referenced_columns(expr) if expr is not None else None
        # `reads` empty is the constant case, which `drop_constant_watermark_dedup_key`
        # owns; leaving it there keeps each rule's proof to a single idea.
        if expr is not None and reads and _is_row_local(expr) and reads <= determining:
            continue
        kept.append(name)
        if isinstance(expr, Col):
            determining.add(expr.name)
    return _narrowed(node, tuple(kept))


@rule(
    name="drop_watermark_dedup_key_pinned_by_equality_filter",
    phase=Phase.REWRITE,
    matches=(WatermarkDedup,),
    category=RuleCategory.REWRITE,
)
def drop_watermark_dedup_key_pinned_by_equality_filter(
    node: WatermarkDedup, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop a dedup key that a filter directly below has pinned to one constant.

    ``WatermarkDedup(Filter(x, col("region") == lit("eu")), subset=("region", "user"))``
    keys on a column every surviving row shares. Only rows with ``region = 'eu'`` reach
    the dedup — SQL's three-valued logic rejects a NULL region, so this is a true constant
    and not "one value plus NULLs" — hence keying on it splits nothing.

    This is the shape `push_filter_through_watermark_dedup` *creates*: it moves a
    key-constant predicate below the dedup, which is precisely when the key it constrains
    becomes redundant. The two compose — that rule shrinks the number of live keys, this
    one shrinks each key.

    Row-set-preserving: the `Filter` stays where it is, so neither the rows entering the
    dedup nor the watermark derived from them moves. Only the key narrows.

    Args:
        node: The `WatermarkDedup` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The dedup keyed on its unpinned columns, or None.
    """
    child = node.input
    if not isinstance(child, Filter):
        return None
    pinned = _pinned_columns(child.predicate)
    if not pinned:
        return None
    return _narrowed(node, tuple(n for n in node.subset if n not in pinned))


@rule(
    name="split_filter_conjuncts_through_watermark_dedup",
    phase=Phase.PUSHDOWN,
    matches=(Filter,),
    category=RuleCategory.REWRITE,
)
def split_filter_conjuncts_through_watermark_dedup(
    node: Filter, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Push the key-constant conjuncts of an `AND` below a dedup, keeping the rest above.

    `push_filter_through_watermark_dedup` moves a predicate below the dedup only when the
    *whole* predicate reads nothing but `subset` columns. A realistic predicate is a
    conjunction — ``region = 'eu' AND amount > 100`` — where one half qualifies and the
    other does not, and that shape got nothing. This splits it.

    The proof is the shipped rule's, applied conjunct by conjunct. ``AND`` is associative
    and commutative, so ``Filter(p_key AND p_rest)`` is ``Filter(p_rest)`` over
    ``Filter(p_key)``; and ``p_key``, reading only key columns, cannot disagree between
    two rows of one key. For a key satisfying ``p_key`` every row still enters the dedup
    and the same first row is emitted, then filtered by ``p_rest``; for a key failing it,
    the old plan emitted that key's first row and immediately rejected it while the new
    plan admits no rows at all. Same output, and the rejected keys never enter the state.

    Inherited caveat, stated rather than hidden: the driver advances its watermark from
    the max event time of the batch it receives, so removing rows below the dedup can
    lower that watermark and change which rows count as late. That is the same trade the
    shipped whole-predicate rule already makes; this rule does not widen it. Fires only
    when the split is genuine — at least one conjunct pushable and at least one not — so
    that rule keeps its own case and the two cannot ping-pong.

    Args:
        node: The `Filter` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The rewritten plan with the key-constant part below the dedup, or None.
    """
    dedup = node.input
    if not isinstance(dedup, WatermarkDedup):
        return None
    keys = set(dedup.subset)
    conjuncts = split_conjuncts(node.predicate)
    pushable = [c for c in conjuncts if referenced_columns(c) <= keys]
    residual = [c for c in conjuncts if not referenced_columns(c) <= keys]
    if not pushable or not residual:
        return None
    pushed = Filter(dedup.input, combine_conjuncts(pushable))
    return Filter(dataclasses.replace(dedup, input=pushed), combine_conjuncts(residual))


@rule(
    name="split_filter_conjuncts_into_stream_join_sides",
    phase=Phase.PUSHDOWN,
    matches=(Filter,),
    category=RuleCategory.REWRITE,
)
def split_filter_conjuncts_into_stream_join_sides(
    node: Filter, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Push each side-pure conjunct of an `AND` into its stream-join side.

    `push_filter_into_stream_join_sides` requires the entire predicate to resolve to one
    side. A predicate mixing a left-only conjunct, a right-only one, and a cross-side one
    — the ordinary case above a stream-stream join — matched none of that and stayed above
    the join, where every row it rejects has already been buffered for a whole interval.

    Sound by the same argument, per conjunct: the join is an inner join, so removing rows
    from one side can only remove output rows, never add or alter one. Conjuncts naming
    both sides, or a column that is not a join output, stay above — a cross-side predicate
    is a join condition, not a pushable filter. Column identity comes from the
    join's own `output` mapping, and each pushed conjunct is remapped from output aliases
    into the side's pre-rename names before being attached. Fires only when at least one
    conjunct pushes *and* something is left behind, so the whole-predicate rule retains
    its case and neither rule can undo the other.

    Args:
        node: The `Filter` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The rewritten plan with side-pure conjuncts inside the join, or None.
    """
    join = node.input
    if not isinstance(join, WatermarkStreamJoin):
        return None
    by_alias = {o.alias: o for o in join.output}
    per_side: dict[str, list[Expr]] = {"left": [], "right": []}
    residual: list[Expr] = []
    for conjunct in split_conjuncts(node.predicate):
        used = referenced_columns(conjunct)
        cols = [by_alias.get(name) for name in used]
        sides = {c.side for c in cols if c is not None}
        if not used or None in cols or len(sides) != 1:
            residual.append(conjunct)
            continue
        renames = {c.alias: c.name for c in cols if c is not None}
        per_side[sides.pop()].append(remap_columns(conjunct, renames))
    # An outer join preserves one side and supplies nulls for the other. Filtering the
    # null-supplying side before the join replaces matched rows with null-padded ones
    # rather than merely removing rows, so a conjunct destined for it stays above the
    # join. (See `push_filter_into_stream_join_side` for the argument in full.)
    if join.emits_unmatched_right:
        residual.extend(per_side["left"])
        per_side["left"] = []
    if join.emits_unmatched_left:
        residual.extend(per_side["right"])
        per_side["right"] = []
    if not (per_side["left"] or per_side["right"]) or not residual:
        return None
    lefts, rights = per_side["left"], per_side["right"]
    left = Filter(join.left, combine_conjuncts(lefts)) if lefts else join.left
    right = Filter(join.right, combine_conjuncts(rights)) if rights else join.right
    pushed = dataclasses.replace(join, left=left, right=right)
    return Filter(pushed, combine_conjuncts(residual))


@rule(
    name="deduplicate_stream_join_keys",
    phase=Phase.NORMALIZE,
    matches=(WatermarkStreamJoin,),
    category=RuleCategory.REWRITE,
)
def deduplicate_stream_join_keys(
    node: WatermarkStreamJoin, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Collapse a repeated `(left_key, right_key)` pair in a stream-stream join.

    The key lists are positional: ``left_keys=("a", "a")`` with ``right_keys=("b", "b")``
    states ``a = b`` twice. Conjunction is idempotent, so dropping the repetition leaves
    the matched pairs — and the output — identical, while the symmetric hash join the
    driver runs per micro-batch hashes a narrower key tuple on both buffered sides.

    Only an exact *pair* repetition is collapsed. ``left_keys=("a", "b")`` with
    ``right_keys=("c", "c")`` repeats a column but states two different equalities
    (``a = c`` and ``b = c``); dropping either would widen the join into another query.

    Args:
        node: The `WatermarkStreamJoin` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The join with de-duplicated key pairs, or None when they are already distinct.
    """
    pairs: list[tuple[str, str]] = []
    for pair in zip(node.left_keys, node.right_keys, strict=True):
        if pair not in pairs:
            pairs.append(pair)
    if not pairs or len(pairs) >= len(node.left_keys):
        return None
    return dataclasses.replace(
        node,
        left_keys=tuple(p[0] for p in pairs),
        right_keys=tuple(p[1] for p in pairs),
    )


@rule(
    name="add_stream_join_key_null_rejection",
    phase=Phase.PUSHDOWN,
    matches=(WatermarkStreamJoin,),
    category=RuleCategory.REWRITE,
)
def add_stream_join_key_null_rejection(
    node: WatermarkStreamJoin, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Reject null-keyed rows before they enter a stream-join buffer.

    An equi-join never matches a NULL key against anything, including another NULL.
    A row whose join key is null is therefore buffered, compared against every arriving
    row of the opposite side for the whole interval window, and emitted never. Adding
    ``IS NOT NULL`` on each side's keys removes exactly those rows and no others, so the
    output is unchanged by construction while the buffered state drops by the null rate —
    which on a real event stream (an unset ``user_id``, an unparsed device id) is not
    small, and unlike a bounded query is *permanent* occupancy, not a scan cost.

    "Emitted never" is only true on a side the join does not **preserve**. An outer join
    emits its preserved side's unmatched rows null-padded, and a null-keyed row is exactly
    an unmatched one — so removing it there deletes output rather than dead weight. The
    rule therefore skips the preserved side, and a full outer join entirely. This is the
    same distinction `push_is_not_null_from_join_key` draws for the bounded join with its
    `FILTERABLE_SIDES` table.

    The guard that keeps it from firing forever is idempotence: a conjunct already present
    on that side is not added again, which stops the rule matching its own output through
    the fixpoint loop. There is deliberately **no** "skip a non-nullable key" guard —
    `available_schema()` rebuilds its fields with `pa.field(name, type)` at every node,
    which defaults to nullable, so no plan here ever reports a non-nullable column and
    such a guard would be dead code that merely looked careful.

    Args:
        node: The `WatermarkStreamJoin` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The join with null-rejecting filters on its sides, or None when nothing is added.
    """
    # A preserved side's null-keyed rows are its *output*, not its dead weight.
    left = None if node.emits_unmatched_left else _reject_null_keys(node.left, node.left_keys)
    right = None if node.emits_unmatched_right else _reject_null_keys(node.right, node.right_keys)
    if left is None and right is None:
        return None
    return dataclasses.replace(
        node,
        left=node.left if left is None else left,
        right=node.right if right is None else right,
    )


def _reject_null_keys(side: LogicalPlan, keys: tuple[str, ...]) -> LogicalPlan | None:
    """`side` with an `IS NOT NULL` conjunct per unguarded key, or None if none is needed."""
    # Compared as IR dicts: `Expr.__eq__` builds an expression rather than returning a
    # bool, so `==`/`in` over `Expr` is a trap. A list, not a set — the dicts are unhashable.
    existing = (
        [c.to_ir() for c in split_conjuncts(side.predicate)] if isinstance(side, Filter) else []
    )
    fresh = []
    for key in keys:
        guard = IsNotNull(Col(key))
        if guard.to_ir() not in existing:
            fresh.append(guard)
    if not fresh:
        return None
    if isinstance(side, Filter):
        return Filter(side.input, combine_conjuncts(split_conjuncts(side.predicate) + fresh))
    return Filter(side, combine_conjuncts(fresh))
