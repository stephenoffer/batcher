"""Regression + self-maintaining guard for the `expr_ir.walk` traversals.

`referenced_columns` collects the columns an expression reads; the projection /
pushdown / fusion rules use it to prune columns. A node type it forgets to descend
into reads columns *invisibly*, so the optimizer prunes a column the query needs
and execution fails with "unknown column" (the B17-class bug — `IsInf` was one such
omission; the transparent `Aliased` wrapper was another).

`test_referenced_columns_descends_into_every_child_node` makes the walk
self-maintaining: it reflects over the authoritative "has sub-expressions" table
(`expr_rewrite.traverse._EXPR_KIDS`) and proves `referenced_columns` finds a probe
column planted in each such node. A future node added to the tree that forgets the
walk turns this red.
"""

from __future__ import annotations

import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

import batcher as bt
from batcher.plan.expr_ir.core import Aliased
from batcher.plan.expr_ir.walk import referenced_columns, remap_columns

_PROBE = "__probe_col__"


@pytest.mark.unit
def test_aliased_is_transparent_to_referenced_columns() -> None:
    # A mid-expression alias reads whatever it wraps; missing this pruned the column.
    assert referenced_columns(Aliased(bt.col("x"), "y")) == {"x"}
    assert referenced_columns(bt.col("a").alias("z") + bt.lit(1)) == {"a"}


@pytest.mark.unit
def test_aliased_is_transparent_to_remap_columns() -> None:
    out = remap_columns(Aliased(bt.col("x"), "y"), {"x": "renamed"})
    assert isinstance(out, Aliased)
    assert referenced_columns(out) == {"renamed"}


@pytest.mark.unit
def test_nested_alias_column_is_not_pruned() -> None:
    # End-to-end: an alias buried inside a projection expression must survive
    # projection pushdown (previously failed with "unknown column").
    ds = bt.from_pydict({"a": [1, 2, 3], "b": [10, 20, 30]})
    inner = ds.select((bt.col("a") + bt.col("b")).alias("t"), bt.col("b"))
    out = inner.select((bt.col("t").alias("z") + bt.lit(1)).alias("r"))
    assert out.collect().to_pydict() == {"r": [12, 23, 34]}


@pytest.mark.unit
def test_referenced_columns_descends_into_every_child_node() -> None:
    """Every node type with sub-expressions must be visible to `referenced_columns`.

    Uses the `_EXPR_KIDS` dispatch table (the single source of "what are a node's
    children") and the IR snapshot's representative instances to rebuild each node
    with a probe column planted in one child, then asserts the walk finds it.
    """
    from tests.unit.test_ir_snapshot import _representatives

    from batcher.plan.expr_rewrite.traverse import _EXPR_KIDS, _EXPR_REBUILD

    # One representative instance per node type (the snapshot builds one of each).
    by_type: dict[type, object] = {}
    for node in _representatives().values():
        by_type.setdefault(type(node), node)

    probe = bt.col(_PROBE)
    missing: list[str] = []
    for node_type, kids_of in _EXPR_KIDS.items():
        rep = by_type.get(node_type)
        if rep is None:
            continue  # no representative to plant a probe in; snapshot test covers tags
        n_kids = len(kids_of(rep))
        if n_kids == 0:
            continue
        rebuilt = _EXPR_REBUILD[node_type](rep, tuple([probe] * n_kids))
        if _PROBE not in referenced_columns(rebuilt):
            missing.append(node_type.__name__)

    assert not missing, (
        "referenced_columns does not descend into these node types (a pruning bug "
        f"waiting to happen): {sorted(missing)}. Add an arm in plan/expr_ir/walk.py."
    )


# --- the memo is shared, so it must not be mutable ------------------------------------


def test_computing_a_parent_does_not_pollute_a_childs_cached_columns() -> None:
    """Asking a `CASE` what it reads must not change the answer for its `ELSE` branch.

    `referenced_columns` memoizes onto the (immutable) expression node and hands the
    *same* object back on every call. `_referenced_columns_impl` used to seed its
    accumulator from a child's cached answer and then union the siblings into it in
    place, so computing the parent silently rewrote the child's cache to the parent's
    answer -- permanently, for the rest of the process.

    The consequence was not a wrong estimate. `Project.__post_init__` validates its items
    with this function, so a projection could be reported as reading a column it does not
    reference; TPC-DS q80 failed to plan with ``projection 'id' references unknown
    column(s) ['store_id']`` for exactly that reason.
    """
    branch, otherwise = bt.col("a"), bt.col("b")
    case = bt.when(bt.col("c") > bt.lit(1)).then(branch).otherwise(otherwise)

    assert referenced_columns(otherwise) == {"b"}
    assert referenced_columns(case) == {"a", "b", "c"}
    # The child is unchanged by the parent having been asked.
    assert referenced_columns(otherwise) == {"b"}
    assert referenced_columns(branch) == {"a"}


def test_the_cached_column_set_cannot_be_mutated_by_a_caller() -> None:
    """The memo is handed out by reference, so it is immutable rather than trusted.

    The previous contract was a comment asserting that no caller mutates the returned
    set. That was not true even inside the module that wrote it, and nothing checked it.
    A `frozenset` makes the guarantee mechanical: an in-place update is an
    `AttributeError` at the call site instead of silent corruption everywhere else.
    """
    cols = referenced_columns(bt.col("a") + bt.col("b"))
    assert cols == {"a", "b"}
    assert isinstance(cols, frozenset)
    with pytest.raises(AttributeError):
        cols.add("c")  # type: ignore[attr-defined]
