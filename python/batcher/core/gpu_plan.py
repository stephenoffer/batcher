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

__all__ = ["execute_cudf_join", "execute_cudf_plan", "gpu_join_spec", "gpu_plan_ops"]

_JOIN_HOW = {"inner": "inner", "left": "left", "right": "right", "outer": "outer", "full": "outer"}

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


def gpu_join_spec(plan: LogicalPlan):
    """`(left_scan, right_scan, join_ir, [op_ir, ...])` for a `[supported ops] over Join(scan,
    scan)` plan, else `None`. Enables an equi-join plus a supported op chain above it on the GPU;
    both join sides must be plain scans (a chain under a join → CPU fallback for now)."""
    from batcher.plan.logical import Join, Scan

    ops: list[dict] = []
    node: Any = plan
    while node is not None and not isinstance(node, (Scan, Join)):
        try:
            ir = node.to_ir()
        except Exception:
            return None
        if ir.get("op") not in _SUPPORTED_OPS or (
            ir["op"] == "aggregate" and not _agg_is_supported(ir)
        ):
            return None
        ops.append(ir)
        node = getattr(node, "input", None)
    if not isinstance(node, Join):
        return None
    if not isinstance(node.left, Scan) or not isinstance(node.right, Scan):
        return None
    join_ir = node.to_ir()
    if join_ir.get("join_type") not in _JOIN_HOW:
        return None
    ops.reverse()
    return node.left, node.right, join_ir, ops


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
    df = _df_from_arrow(table, lib)
    for op in ops:
        df = _apply(df, op, lib)
    return df


def execute_cudf_plan(table: pa.Table, ops: list[dict]) -> pa.Table:
    """Replay `ops` on the GPU via cuDF, returning Arrow. Raises `_Unsupported` (→ CPU fallback)
    for any op/expression outside the translated subset."""
    import cudf

    out = _execute_df_plan(table, ops, cudf)
    return out.to_arrow()


def _df_from_arrow(table, lib):
    if hasattr(lib.DataFrame, "from_arrow"):
        return lib.DataFrame.from_arrow(table)
    return table.to_pandas()


def _execute_join_plan(left_t, right_t, join_ir: dict, ops: list[dict], lib):
    """Equi-join two tables then replay `ops`. Each side's columns are prefixed (L__/R__) before
    the merge so same-named columns never collide; the join's `output` spec then picks the right
    side by name and aliases it — the exact columns and order the CPU engine would produce."""
    lg = _df_from_arrow(left_t, lib).add_prefix("L__")
    rg = _df_from_arrow(right_t, lib).add_prefix("R__")
    merged = lg.merge(
        rg,
        left_on=[f"L__{k}" for k in join_ir["left_keys"]],
        right_on=[f"R__{k}" for k in join_ir["right_keys"]],
        how=_JOIN_HOW[join_ir["join_type"]],
    )
    cols = {}
    for o in join_ir["output"]:
        src = ("L__" if o["side"] == "left" else "R__") + o["name"]
        cols[o["alias"]] = merged[src].reset_index(drop=True)
    out = lib.DataFrame(cols)
    for op in ops:
        out = _apply(out, op, lib)
    return out


def execute_cudf_join(
    left_t: pa.Table, right_t: pa.Table, join_ir: dict, ops: list[dict]
) -> pa.Table:
    """Run an equi-join + op chain on the GPU via cuDF, returning Arrow."""
    import cudf

    return _execute_join_plan(left_t, right_t, join_ir, ops, cudf).to_arrow()
