"""Module-level helpers for `Dataset`: argument coercion and join wiring.

These are pure functions shared by the fluent builder in `dataset.py`. They live
here to keep `dataset.py` focused on the public `Dataset` surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, Expr
from batcher.plan.expr_ir.core import _wrap
from batcher.plan.logical import JoinOutputCol, empty_result_schema
from batcher.plan.schema import placeholder_schema

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

__all__ = [
    "_as_expr",
    "_as_key_expr",
    "_broadcast",
    "_empty_result_schema",
    "_empty_schema",
    "_join_output",
    "_resolve_join_keys",
]


def _as_expr(value: Any) -> Expr:
    """Coerce a `select`/`with_columns` value to an `Expr`, lifting a scalar to `Lit`.

    Delegates to `plan.expr_ir`'s `_wrap` rather than re-deciding, because the two
    disagreed: `_wrap` passes an `AggExpr` through (it is not an `Expr`, but it is a
    legal leaf of one), while this lifted it into ``Lit(AggExpr)``. A bare
    ``select(s=col("x").sum())`` therefore skipped both aggregate-misuse guards and
    failed much later inside the optimizer with ``TypeError: unsupported literal type:
    AggExpr``. Passing it through reaches `AggExpr.to_ir`, which names the real mistake.
    """
    return _wrap(value)


def _as_key_expr(value: str | Expr) -> Expr:
    """A sort/group key is either a column name or an expression."""
    if isinstance(value, str):
        return Col(value)
    if isinstance(value, Expr):
        return value
    raise PlanError(f"expected a column name or expression, got {type(value).__name__}")


def _resolve_join_keys(
    on: str | list[str] | None,
    left_on: str | list[str] | None,
    right_on: str | list[str] | None,
) -> tuple[list[str], list[str]]:
    if on is not None:
        if left_on is not None or right_on is not None:
            raise PlanError("pass either `on` or `left_on`/`right_on`, not both")
        keys = [on] if isinstance(on, str) else list(on)
        return keys, keys
    if left_on is None or right_on is None:
        raise PlanError("join requires `on`, or both `left_on` and `right_on`")
    lk = [left_on] if isinstance(left_on, str) else list(left_on)
    rk = [right_on] if isinstance(right_on, str) else list(right_on)
    if len(lk) != len(rk):
        raise PlanError("left_on and right_on must have the same length")
    return lk, rk


def _join_output(
    left_cols: list[str],
    right_cols: list[str],
    left_keys: list[str],
    right_keys: list[str],
    how: str,
    suffix: str,
) -> list[JoinOutputCol]:
    """Compute the join output column list (key cols, then left, then right)."""
    # Semi/anti joins return only the left relation's columns.
    if how in {"semi", "anti"}:
        return [JoinOutputCol("left", c, c) for c in left_cols]

    out: list[JoinOutputCol] = []
    used: set[str] = set()
    # The two "is this column a join key?" tests below run once per column of each side,
    # against key lists — a linear scan each, so a multi-key join over a wide relation
    # paid columns x keys comparisons to build a column list.
    left_key_set = set(left_keys)
    right_key_set = set(right_keys)

    if how == "full":
        # In a full outer join a key is null on whichever side didn't match, so
        # neither side alone carries it. Emit *both* sides as temp columns; the
        # caller coalesces them into the final key.
        for i, (lk, rk) in enumerate(zip(left_keys, right_keys, strict=True)):
            out.append(JoinOutputCol("left", lk, f"__fk_l_{i}"))
            out.append(JoinOutputCol("right", rk, f"__fk_r_{i}"))
    else:
        # The key value is carried by the side always present for kept rows: the
        # right side for a right join, otherwise the left.
        key_side = "right" if how == "right" else "left"
        key_src = right_keys if key_side == "right" else left_keys
        for out_name, src in zip(left_keys, key_src, strict=True):
            out.append(JoinOutputCol(key_side, src, out_name))
            used.add(out_name)

    for c in left_cols:
        if c in left_key_set:
            continue
        out.append(JoinOutputCol("left", c, c))
        used.add(c)

    for c in right_cols:
        if c in right_key_set:
            continue
        alias = c if c not in used else f"{c}{suffix}"
        out.append(JoinOutputCol("right", c, alias))
        used.add(alias)
    return out


def _asof_output(
    left_cols: list[str],
    right_cols: list[str],
    right_on: str,
    right_by: list[str],
    suffix: str,
) -> list[JoinOutputCol]:
    """Output spec for an ASOF join (left-style): all left columns, then the right's
    columns minus the match keys (`right_on`/`right_by`), suffixed on a name clash."""
    out = [JoinOutputCol("left", c, c) for c in left_cols]
    used = set(left_cols)
    drop = {right_on, *right_by}
    for c in right_cols:
        if c in drop:
            continue
        alias = c if c not in used else f"{c}{suffix}"
        out.append(JoinOutputCol("right", c, alias))
        used.add(alias)
    return out


def _as_str_list(value: str | list[str] | None) -> list[str]:
    """Normalize a key argument (``None`` / a single name / a list) to a list."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _broadcast(flag: bool | list[bool], n: int, name: str) -> list[bool]:
    """Expand a single bool to `n`, or validate a list of the right length."""
    if isinstance(flag, bool):
        return [flag] * n
    if len(flag) != n:
        raise PlanError(f"{name} list has {len(flag)} entries but there are {n} keys")
    return list(flag)


# Both empty-result helpers now live in neutral `plan`, so `api`, `dist`, and `core` share
# one spelling instead of three that disagreed on an empty result's column types. Re-exported
# here under their original private names to keep this module's callers unchanged.
_empty_schema = placeholder_schema
_empty_result_schema = empty_result_schema


def range_semi_join(
    left: Dataset,
    right: Dataset,
    conditions: list[tuple[str, str, str]],
    *,
    negate: bool,
) -> Dataset:
    """Semi (or anti) join two datasets on a single **inequality**.

    `SELECT * FROM a WHERE EXISTS (SELECT 1 FROM b WHERE a.x < b.y)` is exactly
    ``a SEMI JOIN b ON a.x < b.y``, and `NOT EXISTS` is the anti join. Before this the SQL
    front-end raised `NotImplementedError: correlated subqueries not supported` for the
    shape — only *equality*-correlated `EXISTS` decorrelated — so a query DuckDB answers had
    no plan at all here.

    It is a thin wrapper because the work was already done elsewhere: `RangeJoin` carries a
    `join_type`, and `bc_runtime::join::range_join_indices` implements `Semi` and `Anti` and
    is fuzzed against a brute-force cross-product oracle for both. This is the first caller
    to emit a `RangeJoin` that is not an inner join.

    Args:
        left: The outer relation, whose rows are kept or dropped.
        right: The subquery relation, used only as a membership test.
        conditions: One or two `(left_key, right_key, op)` inequalities, each oriented
            ``left_key OP right_key``. Two is the engine's ceiling (IEJoin sorts on two
            axes) and covers the interval shape, ``a.lo < b.y AND b.y < a.hi``.
        negate: `True` for `NOT EXISTS` (an anti join), `False` for `EXISTS`.

    Returns:
        A new `Dataset` carrying only `left`'s columns.
    """
    from batcher.api.dataset.frame import Dataset
    from batcher.plan.logical import RangeCondition, RangeJoin, remap_sources

    offset = len(left._sources)
    right_plan = remap_sources(right._plan, offset)
    output = tuple(JoinOutputCol("left", c, c) for c in left.columns)
    node = RangeJoin(
        left._plan,
        right_plan,
        tuple(RangeCondition(lk, rk, op) for lk, rk, op in conditions),
        "anti" if negate else "semi",
        output,
    )
    return Dataset(node, left._sources + right._sources)
