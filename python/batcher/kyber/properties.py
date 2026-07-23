"""Physical properties — what a plan node *delivers*, and what its parent *requires*.

Cardinality answers "how many rows"; this answers "in what shape". Two shapes matter, and
they are the two every mature optimizer reasons about:

  - **Ordering** — the row order a node hands upward. A `Sort` whose input already delivers
    the ordering it wants is pure waste, and the only way to know that is to propagate the
    property. `RelStats.sorted_by` carries it (a canonical ascending, nulls-last column
    prefix — the one ordering a producer and a consumer can compare unambiguously).
  - **Distribution** — how a relation's rows are spread across workers. A hash join
    co-partitions both sides by the join key, so its *output* is hash-partitioned by that
    key; an aggregate whose group keys are a superset of it therefore computes complete
    groups locally and needs no second shuffle. `dist` already exploits exactly that, but
    it re-derives it inline from the node types instead of asking for the property.

This module is the one place those are computed, so a rule asks a question rather than
re-deriving an answer. It decides; it never executes and never rewrites — `kyber`'s lane.

There is deliberately **no `Exchange` node and no enforcer** here. Inserting one would mean
a new `RelOp` in the JSON IR and a matching operator in the Rust data plane — a two-sided
wire-contract change (`.claude/rules/python-control-plane.md`) — and Batcher's distributed
path does not want it: `dist` schedules the *same* mergeable operators and decides shuffles
itself. So the useful thing a property layer can do here is tell the truth about what is
already delivered, which is what lets redundant work be removed and a shuffle be skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
)
from batcher.plan.stats import RelStats

__all__ = [
    "PhysicalProperties",
    "delivered",
    "hash_partitioned_on",
    "project_ordering",
    "satisfies",
]


@dataclass(frozen=True, slots=True)
class PhysicalProperties:
    """The shape a node hands its parent: row order, and how rows are distributed.

    `ordering` is a canonical **ascending, nulls-last** column prefix — a descending or
    nulls-first key is simply not recorded, because a producer and a consumer can only
    compare orderings they both spell the same way. `hash_partitioned_on` is the key set
    whose equal values are guaranteed to share a partition; empty means "no guarantee",
    which is always the safe answer.
    """

    ordering: tuple[str, ...] = ()
    hash_partitioned_on: tuple[str, ...] = ()


def satisfies(have: PhysicalProperties, want: PhysicalProperties) -> bool:
    """Whether a relation delivering `have` already satisfies a requirement for `want`.

    **Ordering and partitioning contain in opposite directions** — the single most
    error-prone thing here, so it is spelled out:

    * An **ordering** satisfies a requirement when it is a *prefix-extension* of it: rows
      sorted by `(a, b)` are also sorted by `(a)`, so a stronger (longer) order satisfies a
      weaker one, never the reverse.
    * A **partitioning** satisfies a grouping requirement when the delivered keys are a
      *subset* of the required ones — the opposite containment. Rows partitioned by `hash(a)`
      keep every `(a, b)` group whole (equal `(a, b)` implies equal `a`, hence one bucket), so
      partitioning on `(a)` satisfies grouping by `(a, b)`. Partitioning on the *superset*
      `hash(a, b)` does **not** satisfy grouping by `(a)`: two rows sharing `a` but differing
      in `b` hash to different buckets, so the `a`-group straddles them and a reducer that
      skipped the shuffle would emit a partial group — a wrong answer, not a slow one. An
      empty delivered partitioning guarantees nothing and satisfies only an empty requirement.

    Args:
        have: The properties the relation delivers.
        want: The properties the consumer requires.

    Returns:
        True iff no extra sort or shuffle is needed.
    """
    ordered = have.ordering[: len(want.ordering)] == want.ordering
    partitioned = not want.hash_partitioned_on or (
        bool(have.hash_partitioned_on)
        and set(have.hash_partitioned_on) <= set(want.hash_partitioned_on)
    )
    return ordered and partitioned


def project_ordering(items: tuple, child_sorted_by: tuple[str, ...]) -> tuple[str, ...]:
    """The ordering a `Project` delivers, given the ordering of its input.

    A projection selects and renames columns; it reorders nothing, so the input's row order
    survives verbatim — under the *output* names. The estimator used to drop the ordering at
    a `Project` anyway, and a `Project` sits between a sort and its consumer in essentially
    every real query (`SELECT a, b FROM (… ORDER BY a)`), so the delivered order was lost
    exactly where it would have been useful and the redundant-sort rule could never fire
    across a `SELECT`.

    Only a bare-column output can carry an order key: a *computed* output is not the key it
    was computed from. An order key the projection does not carry forward truncates the
    prefix — the columns before it are still delivered in order.

    Args:
        items: The projection's output items.
        child_sorted_by: The canonical ordering the projection's input delivers.

    Returns:
        The canonical ordering the projection delivers, under its output names.
    """
    renamed: dict[str, str] = {}
    for item in items:
        if isinstance(item.expr, Col) and item.expr.name not in renamed:
            renamed[item.expr.name] = item.alias
    out: list[str] = []
    for key in child_sorted_by:
        alias = renamed.get(key)
        if alias is None:
            break  # the order's next key is not projected — the prefix ends here
        out.append(alias)
    return tuple(out)


def hash_partitioned_on(node: LogicalPlan) -> tuple[str, ...]:
    """The keys `node`'s output is guaranteed hash-partitioned by, or `()` if none.

    A hash join co-partitions both inputs by the join key — equal keys hash to the same
    bucket on both sides — so every row of its output carrying that key sits in the bucket
    the key hashes to. The output is therefore hash-partitioned by the join keys, and a
    consumer that groups (or joins again) on a *superset* of them needs no further shuffle.
    That is the property `dist`'s post-join aggregate already relies on; naming it here is
    what lets a rule or a scheduler ask instead of re-deriving.

    An aggregate delivers a partitioning on its group keys for the same reason (its shuffle
    sent equal keys to one reducer). Everything else is left unclaimed — an unclaimed
    partitioning only ever costs an unnecessary shuffle, never a wrong answer.

    Args:
        node: The plan node whose output partitioning is in question.

    Returns:
        The guaranteed hash-partitioning key set, or an empty tuple.
    """
    if isinstance(node, Join):
        # Only an inner/semi-style join guarantees it: an outer join's null-extended rows
        # carry a NULL key that did not come from the hash bucket its row now sits in.
        #
        # NOTE: extending this to `left`/`anti` looks sound on paper — a LEFT join
        # null-extends the *right* columns, so every output row still carries a real left
        # key — and it was tried. It is NOT safe to ship without first pinning the execution
        # contract: `dist._join_sides_are_map_only` documents that each side is "shipped to
        # every worker and re-run against that worker's partition", i.e. a *replicated* join,
        # not a shuffle co-partitioned by the join key. This property is consumed to **skip a
        # shuffle**, so an over-claim is a partial group and a wrong aggregate, not a slow
        # one. Establish how the join actually partitions its output before widening this.
        if node.join_type in ("inner", "semi") and node.left_keys:
            return tuple(node.left_keys)
        return ()
    if isinstance(node, Aggregate) and node.group_keys:
        keys = [k.alias for k in node.group_keys if isinstance(k.expr, Col)]
        return tuple(keys) if len(keys) == len(node.group_keys) else ()
    if isinstance(node, Filter | Limit | Distinct):
        return hash_partitioned_on(node.input)  # row-shrinking: a row never changes bucket
    if isinstance(node, Project):
        # A projection is map-only — it never moves a row between buckets — so the
        # partitioning survives, but *under the output names*: the same rename the ordering
        # gets in `project_ordering`. Every partitioning key must be carried forward as a bare
        # column; if one is dropped or computed the partitioning still physically holds but can
        # no longer be *named*, so it is unclaimed (costing at most a needless shuffle).
        return _rename_keys(hash_partitioned_on(node.input), node.items)
    return ()


def _rename_keys(keys: tuple[str, ...], items: tuple) -> tuple[str, ...]:
    """`keys` under a projection's output aliases, or `()` if any key is not carried through."""
    if not keys:
        return ()
    renamed: dict[str, str] = {}
    for item in items:
        if isinstance(item.expr, Col) and item.expr.name not in renamed:
            renamed[item.expr.name] = item.alias
    out = [renamed.get(k) for k in keys]
    return tuple(o for o in out if o is not None) if all(o is not None for o in out) else ()


def delivered(node: LogicalPlan, stats: RelStats) -> PhysicalProperties:
    """The physical properties `node` hands its parent.

    Args:
        node: The plan node.
        stats: The node's estimated `RelStats` (supplies the propagated ordering).

    Returns:
        The ordering and hash-partitioning the node guarantees.
    """
    return PhysicalProperties(
        ordering=stats.sorted_by,
        hash_partitioned_on=hash_partitioned_on(node),
    )
