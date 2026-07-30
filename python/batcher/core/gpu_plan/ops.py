"""Relational `RelOp` IR → dataframe operations, for the GPU (cuDF) and pandas backends.

One function per operator, each written to match the CPU engine exactly rather than to match
the dataframe library's default. Three of those defaults differ in ways that produce a wrong
answer instead of an error, so they are handled explicitly here:

* a **filter whose predicate is null** drops the row (SQL's three-valued `WHERE`), where
  indexing a frame with a null-bearing boolean mask either raises or keeps the row;
* a **sort** places nulls where the key's `nulls_first` flag says, not where the library
  happens to put them;
* a **limit with an offset** slices by position, which is only the same thing as the
  library's positional slice once the index has been reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.core.gpu_plan.aggs import aggregate, supported_aggregate
from batcher.core.gpu_plan.backend import Unsupported
from batcher.core.gpu_plan.exprs import eval_expr
from batcher.core.gpu_plan.windows import supported_window, window

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend

__all__ = ["SUPPORTED_OPS", "apply_op", "supported_op"]

SUPPORTED_OPS = ("filter", "project", "aggregate", "sort", "distinct", "limit", "window")


def supported_op(ir: dict) -> bool:
    """Whether one `RelOp` IR node is translatable to the dataframe backends.

    This is a *shape* check only — an expression inside the node may still turn out to be
    untranslatable, which surfaces as `Unsupported` at execution and falls back the same way.

    Args:
        ir: The operator's JSON IR node.

    Returns:
        True when the node's operator (and its aggregate/window vocabulary) is translatable.
    """
    op = ir.get("op")
    if op not in SUPPORTED_OPS:
        return False
    if op == "aggregate":
        return supported_aggregate(ir)
    if op == "window":
        return supported_window(ir)
    return True


def apply_op(df, ir: dict, be: DfBackend):
    """Apply one `RelOp` IR node to `df`.

    Args:
        df: The dataframe to transform.
        ir: The operator's JSON IR node.
        be: The dataframe backend to compute on.

    Returns:
        The transformed dataframe.

    Raises:
        Unsupported: For an operator or expression outside the translated subset.
    """
    handler = _HANDLERS.get(ir["op"])
    if handler is None:
        raise Unsupported(ir["op"])
    return handler(df, ir, be)


def _filter(df, ir: dict, be: DfBackend):
    # `fillna(False)`: a null predicate is not a match. SQL's `WHERE` keeps only rows the
    # predicate proves true, and a null-bearing mask is otherwise either an error (pandas)
    # or a kept row (which would add rows the CPU engine drops).
    mask = be.column(eval_expr(ir["predicate"], df, be), df).fillna(False)
    return df[mask].reset_index(drop=True)


def _project(df, ir: dict, be: DfBackend):
    cols = {p["alias"]: be.column(eval_expr(p["expr"], df, be), df) for p in ir["exprs"]}
    return be.lib.DataFrame(cols).reset_index(drop=True)


def _sort(df, ir: dict, _be: DfBackend):
    keys = ir["keys"]
    if any(k["expr"].get("e") != "col" for k in keys):
        raise Unsupported("sort on a computed key")
    # Both backends take one `na_position` for the whole sort, so keys that disagree on
    # null placement cannot be expressed and must fall back rather than be approximated.
    positions = {bool(k.get("nulls_first")) for k in keys}
    if len(positions) > 1:
        raise Unsupported("sort with mixed null placement")
    out = df.sort_values(
        [k["expr"]["name"] for k in keys],
        ascending=[not k["descending"] for k in keys],
        na_position="first" if positions.pop() else "last",
        kind="stable",
    )
    if ir.get("limit"):
        out = out.head(ir["limit"])
    return out.reset_index(drop=True)


def _distinct(df, _ir: dict, _be: DfBackend):
    return df.drop_duplicates().reset_index(drop=True)


def _limit(df, ir: dict, _be: DfBackend):
    offset = ir.get("offset", 0)
    return df.iloc[offset : offset + ir["n"]].reset_index(drop=True)


_HANDLERS = {
    "filter": _filter,
    "project": _project,
    "aggregate": aggregate,
    "sort": _sort,
    "distinct": _distinct,
    "limit": _limit,
    "window": window,
}
