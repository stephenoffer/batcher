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

SUPPORTED_OPS = (
    "filter",
    "project",
    "aggregate",
    "sort",
    "distinct",
    "limit",
    "window",
    "unnest",
    "unpivot",
    "row_id",
)


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
    import pyarrow as pa

    from batcher.core.gpu_plan.vocab.dates import date_typed

    cols = {}
    for p in ir["exprs"]:
        alias = p["alias"]
        cols[alias] = be.column(eval_expr(p["expr"], df, be), df)
        # Neither library has a calendar-day type, so a projection that *computes* a date has
        # to say so — the column itself cannot. Recorded or cleared per alias, so the last
        # projection to produce a name decides what that name is.
        if date_typed(p["expr"], be):
            be.remember_date_alias(alias, pa.date32())
        else:
            be.forget_date_alias(alias)
    return be.lib.DataFrame(cols).reset_index(drop=True)


def _sort(df, ir: dict, be: DfBackend):
    # The output columns are the ones the frame arrived with. Read *before* the private sort
    # keys and null indicators are added, because `_sort_columns` adds them to this same frame:
    # asking the mutated frame what its columns are returned `__bt_sn0` and `__bt_sk0` as part
    # of the answer, so every GPU query whose last operator was a sort came back with columns
    # the CPU engine does not produce.
    output = list(df.columns)
    # Shallow: the copy shares every column's buffer, so this costs a Python object and not a
    # device allocation. What it buys is that the caller's frame is not mutated — which matters
    # in a plan tree, where one leaf's frame can be an input to two operators.
    df = df.copy(deep=False)
    names, ascending = _sort_columns(df, ir["keys"], be)
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
    return out.reset_index(drop=True)[output]


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


#: Private columns `unnest` carries across the explode: each pre-explode row's position, so an
#: element can be numbered within its own list, and whether that row's list held no elements at
#: all, which decides whether the row the explode invented has a position.
_ROW = "__bt_unnest_row"
_EMPTY = "__bt_unnest_empty"


def _unnest(df, ir: dict, be: DfBackend):
    """`UNNEST` — one row per element of a list column.

    Both libraries' `explode` implements the *outer* form: a row whose list is null or empty
    comes back once, carrying a null element. SQL's `UNNEST` and DuckDB's default do the
    opposite and drop such a row, so the plan's `outer` flag decides whether those rows are
    filtered back out. Getting this backwards is invisible row loss rather than an error — a
    document that chunked to nothing would silently take its id and metadata with it.

    The exploded column stays in the position it already occupied, because that is what the
    plan's own `available_columns` promises; only the optional element index is appended.
    """
    column, alias = ir["column"], ir["alias"]
    index_alias = ir.get("index_alias")
    outer = bool(ir.get("outer", False))
    order = [alias if c == column else c for c in df.columns]
    df = _marked(df, column, be, number=index_alias is not None)
    out = df.explode(column).reset_index(drop=True)
    if not outer:
        # A row the explode invented for an empty or null list is the row the default
        # semantics drop. It is identified by the marker, NEVER by the element being null: a
        # list may legitimately *contain* a null, and that element is a row the engine keeps.
        out = out[~out[_EMPTY]].reset_index(drop=True)
    out = (
        _with_element_index(out, index_alias, be)
        if index_alias is not None
        else out.drop(columns=[_EMPTY])
    )
    if index_alias is not None:
        order = [*order, index_alias]
    return out.rename(columns={column: alias})[order]


def _marked(df, column: str, be: DfBackend, *, number: bool):
    """`df` with the private columns the explode needs carried across it.

    The emptiness test is taken *before* the explode, because afterwards it cannot be taken at
    all: a row invented for an empty list and a row carrying a list's own null element are the
    same row by then, and they need opposite answers. The first is not a row at all under the
    default semantics; the second is a row whose value happens to be null.
    """
    df = df.copy()
    lengths = df[column].list.len()
    # `fillna(True)`: a null list has no length, and no elements either, so it is empty.
    df[_EMPTY] = be.column((lengths.isna() | (lengths == 0)).fillna(True), df)
    if number:
        df[_ROW] = range(len(df))
    return df


def _with_element_index(out, index_alias: str, be: DfBackend):
    """Number each exploded element within its own list, 0-based.

    A row kept only by `outer` has no element and so has no position. The engine reports null
    there rather than zero, which would read as "the first element" of a list that has none.
    """
    import pyarrow as pa

    out = out.copy()
    counted = out.groupby(_ROW).cumcount()
    numbered = be.column(counted, out).astype(be.dtype(pa.int64()))
    out[index_alias] = numbered.where(~out[_EMPTY], None)
    return out.drop(columns=[_ROW, _EMPTY])


def _row_id(df, ir: dict, be: DfBackend):
    """`with_row_index` — a sequential index column, **prepended**.

    Prepended rather than appended, because that is what the plan's own `available_columns`
    promises and what Polars' `with_row_index` does. Appending it would put every column of
    every result in the wrong place, which a comparison by column *name* would never notice.

    The numbering is over the whole relation, which is exactly why a chain carrying this
    operator must not be split across devices: each shard would restart at the offset. Nothing
    here enforces that, and nothing needs to — `row_id` has no mergeable form, so the shard
    planner declines the chain and it runs on one device (`plan.distribution._split_reducer`).
    """
    import pyarrow as pa

    offset = int(ir.get("offset", 0))
    out = df.copy()
    out[ir["alias"]] = be.series(range(offset, offset + len(out))).astype(be.dtype(pa.int64()))
    return out[[ir["alias"], *df.columns]]


def _unpivot(df, ir: dict, be: DfBackend):
    """`UNPIVOT` — wide to long, the `melt` both libraries spell the same way.

    Every column outside `index` and `on` is dropped, which is the operator's contract rather
    than an accident of the library: the plan's own `available_columns` is exactly
    `[*index, variable_name, value_name]`.
    """
    return be.lib.melt(
        df,
        id_vars=list(ir["index"]),
        value_vars=list(ir["on"]),
        var_name=ir["variable_name"],
        value_name=ir["value_name"],
    ).reset_index(drop=True)


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
    "unnest": _unnest,
    "unpivot": _unpivot,
    "row_id": _row_id,
}
