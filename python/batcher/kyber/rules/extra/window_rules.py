"""Window-operator rewrites — canonicalize a `Window`'s keys and prune dead output.

A `Window` partitions rows, orders them within each partition, and appends one
column per function (the input columns pass through). These rewrites shrink the
operator's specification without changing a single output value:

- `drop_dead_window` removes window functions whose output no enclosing `Project`
  reads (or the whole `Window`, when none is read) — the case the global column
  pruner (`projection_rewrite`) does *not* cover: that pass prunes the window's
  *child* columns but always keeps every function it finds.
- `dedupe_window_partition_keys` / `dedupe_window_order_keys` drop a partition or
  order key that structurally duplicates an earlier one.
- `drop_constant_partition_key` / `drop_constant_order_key` drop a literal-constant
  key, which — being equal on every row — neither subdivides a partition nor
  distinguishes any two rows in the ordering.

Every rule is semantics-preserving (only the plan shape changes) and returns None
once there is nothing left to do, so the rewrite fixpoint terminates.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Lit, referenced_columns
from batcher.plan.logical import LogicalPlan, Project, Window

__all__ = [
    "dedupe_window_order_keys",
    "dedupe_window_partition_keys",
    "drop_constant_order_key",
    "drop_constant_partition_key",
    "drop_dead_window",
]


@rule(name="drop_dead_window", phase=Phase.PUSHDOWN, matches=(Project,))
def drop_dead_window(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Project(Window(x, …, fns), items)` → drop the `fns` no `items` expression reads.

    A window function computes an appended column; if the enclosing projection never
    references that column, the computation is dead. When *no* function output is
    read the whole `Window` disappears (it is row-preserving, so its input columns
    reach the projection unchanged); otherwise only the unread functions are removed.

    Gated to `rank_limit is None`: a `rank_limit` window also *filters* rows (a
    per-partition top-k), so it is not dead even when its rank column is unused. This
    complements the global column pruner, which prunes the window's input columns but
    never drops a function it finds. Returns None when every function is read, so the
    rule is idempotent.
    """
    win = node.input
    if not isinstance(win, Window) or win.rank_limit is not None:
        return None
    used: set[str] = set()
    for item in node.items:
        used |= referenced_columns(item.expr)
    kept = tuple(fn for fn in win.functions if fn.alias in used)
    if len(kept) == len(win.functions):
        return None
    if not kept:
        # Nothing the window produces is read — it is a pure pass-through of `x`.
        return dataclasses.replace(node, input=win.input)
    new_win = Window(win.input, win.partition_keys, win.order_keys, kept, win.rank_limit)
    return dataclasses.replace(node, input=new_win)


@rule(name="dedupe_window_partition_keys", phase=Phase.NORMALIZE, matches=(Window,))
def dedupe_window_partition_keys(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`PARTITION BY a, a` → `PARTITION BY a` — drop structurally duplicate keys.

    Partitioning groups rows by the *tuple* of key values; a repeated key contributes
    no further subdivision (two rows equal on the first copy are equal on the second,
    NULLs included). Keys are compared by lowered IR (never ``==``, which builds a
    comparison expression). Returns None when the keys are already distinct.
    """
    seen: list[object] = []
    kept = []
    for pk in node.partition_keys:
        sig = pk.to_ir()
        if sig in seen:
            continue
        seen.append(sig)
        kept.append(pk)
    if len(kept) == len(node.partition_keys):
        return None
    return dataclasses.replace(node, partition_keys=tuple(kept))


@rule(name="dedupe_window_order_keys", phase=Phase.NORMALIZE, matches=(Window,))
def dedupe_window_order_keys(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`ORDER BY a ASC, a DESC` → `ORDER BY a ASC` — drop a duplicated order expression.

    Once rows are ordered by an expression, a later order key on the *same* expression
    sees only rows already tied on it (equal values, NULLs included), so it can never
    reorder them and never changes peer-group / frame membership. The first
    occurrence's direction and null placement win; later copies of that expression are
    dropped (compared by lowered IR). Returns None when the order expressions are
    already distinct.
    """
    seen: list[object] = []
    kept = []
    for key in node.order_keys:
        sig = key.expr.to_ir()
        if sig in seen:
            continue
        seen.append(sig)
        kept.append(key)
    if len(kept) == len(node.order_keys):
        return None
    return dataclasses.replace(node, order_keys=tuple(kept))


@rule(name="drop_constant_partition_key", phase=Phase.NORMALIZE, matches=(Window,))
def drop_constant_partition_key(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`PARTITION BY <const>, a` → `PARTITION BY a` — drop literal-constant keys.

    A literal is the same value on every row, so it never splits a partition. Dropping
    it (including the case where every partition key is constant, leaving one partition
    over all rows — exactly what `PARTITION BY <const>` already means) preserves every
    per-partition result. Returns None when no partition key is a literal.
    """
    kept = tuple(pk for pk in node.partition_keys if not isinstance(pk, Lit))
    if len(kept) == len(node.partition_keys):
        return None
    return dataclasses.replace(node, partition_keys=kept)


@rule(name="drop_constant_order_key", phase=Phase.NORMALIZE, matches=(Window,))
def drop_constant_order_key(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`ORDER BY a, <const>, b` → `ORDER BY a, b` — drop literal-constant order keys.

    A literal has the same value on every row, so it distinguishes no two rows: it is a
    pure no-op tie in the ordering and never affects peer-group / frame membership.
    Only fires while at least one non-constant order key remains, so the ordering stays
    well-defined and non-empty (never invalidating a ranking function). Returns None
    when no order key is a literal, or when every order key is a literal (left intact,
    conservatively). Idempotent — after firing, no literal order key remains.
    """
    kept = tuple(key for key in node.order_keys if not isinstance(key.expr, Lit))
    if len(kept) == len(node.order_keys) or not kept:
        return None
    return dataclasses.replace(node, order_keys=kept)
