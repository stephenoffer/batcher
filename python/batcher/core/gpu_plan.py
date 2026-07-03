"""Translate a linear Batcher plan to a GPU dataframe execution (cuDF) — many ops, not one.

Extends the GPU backend from a single group-by to a chain of relational ops — filter, project /
with_columns, group-by aggregate, sort, distinct, limit — by walking the plan's RelOp IR and its
Expr IR and replaying each on a cuDF DataFrame (the same approach Polars-GPU takes to its cuDF
engine). The executor is dataframe-library-parameterized: it runs on **cuDF** on a GPU worker
(the accelerated backend) and on **pandas** for the head-runnable correctness test against the
native CPU engine — the GPU is only *where* it runs.

`gpu_plan_ops(plan)` returns the ordered op-IR list for a supported linear single-source plan (or
`None` so the caller falls back to the CPU engine); `execute_cudf_plan(table, ops)` replays them.
Any unsupported op or expression raises `_Unsupported`, caught into a fallback.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.plan.logical import LogicalPlan

__all__ = ["execute_cudf_plan", "gpu_plan_ops"]

_SUPPORTED_OPS = ("filter", "project", "aggregate", "sort", "distinct", "limit")
_AGG_FUNCS = {"sum", "count", "mean", "min", "max"}
_BINOPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "mod": operator.mod,
    "gt": operator.gt,
    "lt": operator.lt,
    "ge": operator.ge,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "and": operator.and_,
    "or": operator.or_,
}


class _Unsupported(Exception):
    """An op or expression the cuDF translator does not handle — triggers CPU fallback."""


def gpu_plan_ops(plan: LogicalPlan):
    """`(scan, [op_ir, ...])` for a linear chain of supported ops over a single scan, bottom-up,
    else `None`. A non-supported / branching node (join, union, map_batches, window) makes it
    ineligible — the caller then uses the CPU engine."""
    from batcher.plan.logical import Scan

    ops: list[dict] = []
    node: Any = plan
    while node is not None and not isinstance(node, Scan):
        try:
            ir = node.to_ir()
        except Exception:
            return None  # e.g. map_batches — Python-only, not lowered to the engine IR
        if ir.get("op") not in _SUPPORTED_OPS:
            return None
        # group keys / agg inputs must be plain columns for the fast cuDF path.
        if ir["op"] == "aggregate" and not _agg_is_supported(ir):
            return None
        ops.append(ir)
        node = getattr(node, "input", None)
    if not isinstance(node, Scan) or not ops:
        return None
    ops.reverse()  # bottom-up: apply nearest-the-scan first
    return node, ops


def _agg_is_supported(ir: dict) -> bool:
    if any(gk["expr"].get("e") != "col" for gk in ir["group_keys"]):
        return False
    return all(
        a.get("func") in _AGG_FUNCS and a.get("input", {}).get("e") == "col"
        for a in ir["aggregates"]
    )


def _lit(value: dict):
    return next(iter(value.values()))


def _eval(ir: dict, df):
    """Evaluate an Expr IR against a dataframe `df` (cuDF or pandas), returning a Series/scalar."""
    e = ir.get("e")
    if e == "col":
        return df[ir["name"]]
    if e == "lit":
        return _lit(ir["value"])
    if e == "binary":
        op = ir["op"]
        if op not in _BINOPS:
            raise _Unsupported(f"binary op {op}")
        return _BINOPS[op](_eval(ir["left"], df), _eval(ir["right"], df))
    if e == "math":
        x = _eval(ir["input"], df)
        fn = ir["fn"]
        method = getattr(x, fn, None)  # Series.abs / .sqrt / .log / .exp / ...
        if method is None:
            raise _Unsupported(f"math fn {fn}")
        return method()
    raise _Unsupported(f"expr {e}")


def _apply(df, op: dict, lib):
    """Apply one RelOp IR to `df` (a cuDF/pandas DataFrame from module `lib`)."""
    kind = op["op"]
    if kind == "filter":
        return df[_eval(op["predicate"], df)]
    if kind == "project":
        cols = {}
        for p in op["exprs"]:
            v = _eval(p["expr"], df)
            cols[p["alias"]] = v if hasattr(v, "reset_index") else _broadcast(v, df, lib)
        return lib.DataFrame(cols)
    if kind == "aggregate":
        return _aggregate(df, op, lib)
    if kind == "sort":
        by = [k["expr"]["name"] for k in op["keys"]]
        asc = [not k["descending"] for k in op["keys"]]
        out = df.sort_values(by, ascending=asc)
        if op.get("limit"):
            out = out.head(op["limit"])
        return out.reset_index(drop=True)
    if kind == "distinct":
        return df.drop_duplicates().reset_index(drop=True)
    if kind == "limit":
        off = op.get("offset", 0)
        return df.iloc[off : off + op["n"]].reset_index(drop=True)
    raise _Unsupported(kind)


def _broadcast(scalar, df, lib):
    return lib.Series([scalar] * len(df), index=df.index)


def _aggregate(df, op: dict, lib):
    keys = [gk["expr"]["name"] for gk in op["group_keys"]]
    key_aliases = [gk["alias"] for gk in op["group_keys"]]
    col_funcs: dict[str, list[str]] = {}
    for a in op["aggregates"]:
        col_funcs.setdefault(a["input"]["name"], [])
        if a["func"] not in col_funcs[a["input"]["name"]]:
            col_funcs[a["input"]["name"]].append(a["func"])
    grouped = df.groupby(keys, sort=False).agg(col_funcs)
    out = lib.DataFrame(
        {
            alias: grouped.index.get_level_values(k)
            for alias, k in zip(key_aliases, keys, strict=True)
        }
    )
    out = out.reset_index(drop=True)
    gvals = grouped.reset_index(drop=True)
    for a in op["aggregates"]:
        out[a["alias"]] = gvals[(a["input"]["name"], a["func"])]
    return out


def _execute_df_plan(table: pa.Table, ops: list[dict], lib):
    """Replay `ops` on a dataframe built from `table` using module `lib` (cuDF or pandas)."""
    if hasattr(lib.DataFrame, "from_arrow"):
        df = lib.DataFrame.from_arrow(table)
    else:
        df = table.to_pandas()
    for op in ops:
        df = _apply(df, op, lib)
    return df


def execute_cudf_plan(table: pa.Table, ops: list[dict]) -> pa.Table:
    """Replay `ops` on the GPU via cuDF, returning Arrow. Raises `_Unsupported` (→ CPU fallback)
    for any op/expression outside the translated subset."""
    import cudf

    out = _execute_df_plan(table, ops, cudf)
    return out.to_arrow()
