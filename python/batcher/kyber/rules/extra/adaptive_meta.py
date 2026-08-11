"""Adaptive metadata rules — simplifications a provably-EXACT cardinality unlocks.

Batcher's estimator tags every operator's row count with a `Provenance`: a count is
`EXACT` only when it is provably correct *without executing a row* — a footer/manifest
row count, a global aggregate's single row, `Limit(_, 0)`, or a materialized
intermediate the adaptive loop spliced back as a known-size source. These rules act
only on that EXACT signal (`Provenance.is_exact`), so a mere Selinger estimate can
never drive them — a fast wrong answer is a bug.

- `drop_inert_limit` removes a `LIMIT n` whose input provably already has ≤ n rows: the
  cap never fires, so the operator — a pipeline breaker Carbonite budgets and the
  adaptive loop would split a stage on — is pure overhead.
- `empty_limit_past_offset` turns a `LIMIT … OFFSET k` that provably skips past the last
  row into the canonical empty relation, which `propagate_empty_relation` folds away.
- `fold_exact_empty_input` recognises a provably-empty *input* (an empty source, not
  only the `Limit(_, 0)` marker) under a schema-preserving operator and folds it to that
  marker, collapsing the filter/sort/distinct/sample above an empty scan.

Every rewrite is byte-identical in result (same rows, same order) and idempotent
(returns None once there is nothing left to simplify). They are especially potent inside
the adaptive re-optimization loop, where each finished stage is spliced back as an
EXACT-sized source — so a downstream limit/empty becomes provable mid-query.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.logical import (
    Distinct,
    Filter,
    Limit,
    LogicalPlan,
    Sample,
    Sort,
)

__all__ = [
    "drop_inert_limit",
    "empty_limit_past_offset",
    "fold_exact_empty_input",
]

# Unary operators that pass their input through unchanged in schema and can only shrink
# or reorder rows — so a provably-empty input makes them empty with the *identical*
# schema. Mirrors `zonemap_pruning`'s schema-preserving set; kept local so this module
# owns its own matched types rather than importing a private name across modules.
_SCHEMA_PRESERVING = (Filter, Sort, Distinct, Sample)


def _exact_rows(node: LogicalPlan, ctx: OptimizerContext) -> float | None:
    """The node's EXACT output row count, or None when the estimate is not provable.

    Returns a number only when the estimate's provenance proves it correct without
    execution (`Provenance.is_exact`). Every rule here refuses to act otherwise, so a
    Selinger guess can never drop a `Limit` or fold a subtree.
    """
    if ctx is None:
        return None
    stats = ctx.estimator.estimate(node)
    return stats.rows if stats.provenance.is_exact else None


def _is_empty_marker(node: LogicalPlan) -> bool:
    """Whether `node` is already the canonical empty relation `Limit(_, 0)`."""
    return isinstance(node, Limit) and node.n == 0


@rule(name="drop_inert_limit", phase=Phase.PUSHDOWN, matches=(Limit,))
def drop_inert_limit(node: Limit, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a `LIMIT n` whose input is provably no larger than `n`.

    `Limit(x, n, offset=0)` returns the first ``min(n, |x|)`` rows in input order; when
    ``|x| ≤ n`` is *proven* (an EXACT row count) that is every row of ``x``, unchanged
    and in order — so the limit is dead weight, and it is a pipeline breaker Carbonite
    budgets and the adaptive loop would split a stage on. Restricted to ``offset == 0``
    (a non-zero offset still skips rows) and ``n ≥ 1`` (``n == 0`` is the empty marker,
    left to the empty-propagation rules). EXACT-gated, so an estimate never drops it;
    returns None once no such limit remains (idempotent).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.kyber.optimizer import Optimizer
            >>> from batcher.plan.logical import Limit
            >>> ds = bt.from_pydict({"x": [1, 2, 3]})  # 3 rows, exactly known
            >>> plan = ds.limit(10)._plan  # cap 10 ≥ 3 rows → inert
            >>> out = Optimizer(sources=ds._sources).logical_rewrite(plan)
            >>> isinstance(out, Limit)
            False
    """
    if node.offset != 0 or node.n < 1:
        return None
    rows = _exact_rows(node.input, ctx)
    if rows is not None and rows <= node.n:
        return node.input
    return None


@rule(name="empty_limit_past_offset", phase=Phase.PUSHDOWN, matches=(Limit,))
def empty_limit_past_offset(node: Limit, ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold a `LIMIT … OFFSET k` that provably skips every row into an empty relation.

    `Limit(x, n, offset=k)` yields the rows ``[k : k + n)``; when ``|x| ≤ k`` is *proven*
    (an EXACT row count) the window opens past the last row, so the result is empty.
    Rewrite to the canonical empty marker ``Limit(x, 0)``, which
    `propagate_empty_relation` then folds up through the operators above. Restricted to
    ``offset ≥ 1`` and skips a node already at ``n == 0`` (the marker itself), so the
    rule is idempotent. EXACT-gated — an estimate never empties a query.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.kyber.optimizer import Optimizer
            >>> from batcher.plan.logical import Limit
            >>> ds = bt.from_pydict({"x": [1, 2, 3]})  # 3 rows
            >>> plan = ds.limit(10, offset=3)._plan  # skips all 3 → empty
            >>> out = Optimizer(sources=ds._sources).logical_rewrite(plan)
            >>> isinstance(out, Limit) and out.n == 0
            True
    """
    if node.offset < 1 or node.n == 0:
        return None
    rows = _exact_rows(node.input, ctx)
    if rows is not None and rows <= node.offset:
        return Limit(node.input, 0)
    return None


@rule(name="fold_exact_empty_input", phase=Phase.FUSION, matches=_SCHEMA_PRESERVING)
def fold_exact_empty_input(node: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold a schema-preserving operator over a provably-empty input to the empty marker.

    A `Filter`/`Sort`/`Distinct`/`Sample` over a relation *proven* empty (an EXACT row
    count of 0 — an empty source, or a subtree metadata shows is empty) emits zero rows
    with the input's schema, so it is replaced by the canonical empty marker
    ``Limit(input, 0)``. This complements `propagate_empty_relation`, which starts only
    from an existing ``Limit(_, 0)`` marker: here the emptiness is proven from the
    estimator's EXACT cardinality, catching an empty *source* that carries no marker.
    Runs bottom-up in the SELECTION pass so a chain of such operators collapses in one
    traversal; skips an input already at the marker, so it is idempotent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher import col
            >>> from batcher.kyber.optimizer import Optimizer
            >>> from batcher.plan.logical import Limit
            >>> ds = bt.from_pydict({"x": []})  # provably empty source
            >>> plan = ds.filter(col("x") > 1)._plan
            >>> out = Optimizer(sources=ds._sources).logical_rewrite(plan)
            >>> isinstance(out, Limit) and out.n == 0
            True
    """
    child = node.input
    if _is_empty_marker(child):
        return None  # already the marker — leave it to propagate_empty_relation
    if _exact_rows(child, ctx) == 0:
        return Limit(child, 0)
    return None
