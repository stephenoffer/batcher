"""Relational `RelOp` IR → dataframe operations, for the GPU (cuDF) and pandas backends.

One function per operator, each written to match the CPU engine exactly rather than to match
the dataframe library's default. Four of those defaults differ in ways that produce a wrong
answer instead of an error, so they are handled explicitly here:

* a **filter whose predicate is null** drops the row (SQL's three-valued `WHERE`), where
  indexing a frame with a null-bearing boolean mask either raises or keeps the row;
* a **sort** places nulls where each key's own `nulls_first` flag says, not where the library
  happens to put them — and the libraries take one null position for the whole sort, so a
  per-key one is carried by an indicator column rather than declined;
* a **distinct** treats `-0.0` and `0.0` as one value, as IEEE and SQL do, where the libraries
  deduplicate on a hash of the bits and keep both;
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

__all__ = ["SUPPORTED_OPS", "apply_op", "distinct_rows", "fold_zero", "supported_op"]

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


def _sort(df, ir: dict, be: DfBackend):
    keys = ir["keys"]
    names, ascending = _sort_columns(df, keys, be)
    out = df.sort_values(
        names,
        ascending=ascending,
        # Every key's nulls are already separated by its own indicator column below, so the
        # frame-wide position no longer decides anything and is pinned only for determinism.
        na_position="last",
        kind="stable",
    )
    if ir.get("limit"):
        out = out.head(ir["limit"])
    return out.reset_index(drop=True)[list(df.columns)]


def _sort_columns(df, keys: list[dict], be: DfBackend) -> tuple[list[str], list[bool]]:
    """The columns to sort by and their directions, materializing what the frame lacks.

    Two shapes the dataframe libraries cannot express directly are handled by adding a column
    rather than by falling back, because both are ordinary in real queries and each one
    otherwise drops the *whole* plan to the CPU engine:

    * a **computed key** (``sort("a", bt.col("b").str.lower())``) is evaluated into a private
      column and sorted on that;
    * **per-key null placement** is carried by a private boolean indicator sorted immediately
      before its key. Both backends take one `na_position` for the entire sort, so a query
      whose keys disagree could not be expressed at all; an indicator separates each key's
      nulls under that key's own rule, and since every null shares one key value the indicator
      orders them completely.
    """
    names: list[str] = []
    ascending: list[bool] = []
    for i, key in enumerate(keys):
        expr = key["expr"]
        if expr.get("e") == "col":
            name = expr["name"]
        else:
            name = f"__bt_sk{i}"
            df[name] = be.column(eval_expr(expr, df, be), df)
        indicator = f"__bt_sn{i}"
        df[indicator] = be.column(df[name].isna(), df)
        # True sorts above False, so ascending on the indicator puts the non-nulls first.
        names.extend((indicator, name))
        ascending.extend((not bool(key.get("nulls_first")), not key["descending"]))
    return names, ascending


def fold_zero(series, be: DfBackend):
    """`series` with negative zero folded onto zero, for a float column; unchanged otherwise.

    `-0.0` and `0.0` are one value to IEEE, to SQL and to the engine, and two to a dataframe
    library, which compares them by a hash of their bits. Every place a float takes part in an
    *identity* — a group key, a distinct row, a join key's membership test — needs the fold, or
    one value silently becomes two.

    Adding zero is the fold and nothing else: `-0.0 + 0.0` is `+0.0`, and `x + 0.0` is `x` for
    every other value, the infinities and `NaN` included.

    Args:
        series: The column to fold.
        be: The dataframe backend it belongs to.

    Returns:
        The folded column, or `series` itself when it is not a float column.
    """
    return series + 0.0 if be.is_float(series) else series


def distinct_rows(df, be: DfBackend):
    """`df` with duplicate rows dropped, under the engine's idea of which rows are duplicates.

    Deduplicating on a folded *copy* while emitting the original row is what keeps the
    surviving row's value the one the engine keeps: the first occurrence, negative zero
    included.

    Args:
        df: The frame to deduplicate.
        be: The dataframe backend to compute on.

    Returns:
        The frame with duplicate rows removed, index reset.
    """
    folded = {name: fold_zero(df[name], be) for name in df.columns}
    if all(folded[name] is df[name] for name in df.columns):
        return df.drop_duplicates().reset_index(drop=True)
    return df[~be.lib.DataFrame(folded).duplicated()].reset_index(drop=True)


def _distinct(df, _ir: dict, be: DfBackend):
    # DISTINCT is a group-by over every column (`Distinct.as_aggregate`), so it inherits the
    # group-key problem the fold above exists for.
    return distinct_rows(df, be)


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
