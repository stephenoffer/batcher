"""Module-level helpers for `Dataset`: argument coercion and join wiring.

These are pure functions shared by the fluent builder in `dataset.py`. They live
here to keep `dataset.py` focused on the public `Dataset` surface.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, Expr, Lit
from batcher.plan.logical import JoinOutputCol

__all__ = [
    "_as_expr",
    "_as_key_expr",
    "_broadcast",
    "_empty_schema",
    "_join_output",
    "_resolve_join_keys",
]


def _as_expr(value: Any) -> Expr:
    return value if isinstance(value, Expr) else Lit(value)


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
        if c in left_keys:
            continue
        out.append(JoinOutputCol("left", c, c))
        used.add(c)

    for c in right_cols:
        if c in right_keys:
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


def _empty_schema(names: list[str]) -> pa.Schema:
    # When a query yields zero batches we still need a schema; use null-typed
    # placeholders until the optimizer tracks derived-column types.
    return pa.schema([pa.field(name, pa.null()) for name in names])
