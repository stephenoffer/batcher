"""Group-by aggregation on a dataframe backend, matching the CPU engine's null semantics.

Aggregation is where a translated GPU plan is easiest to get subtly wrong, because the
dataframe libraries' defaults disagree with Arrow/SQL on three points that never show up in
a smoke test and each produce a *wrong answer* rather than an error:

* a **null group key** is a group. `groupby` drops it by default, so a query whose key column
  had nulls silently lost rows from its result;
* the **sum of an all-null group is null**, not `0.0`. `groupby.sum()` returns `0.0`, which
  reads as a real measurement;
* **variance and standard deviation are the sample** forms (`ddof=1`), which is the libraries'
  default but not the one a "population" reading would pick.

Each aggregate is computed off one shared `GroupBy`, so the grouping is built once on the
device and every reduction reuses it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.core.gpu_plan.backend import Unsupported
from batcher.core.gpu_plan.exprs import eval_expr

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend

__all__ = ["aggregate", "supported_aggregate"]

# Reductions that are a method of the same name on both backends' `GroupBy`, taking no
# argument and already agreeing with the engine on nulls (nulls are skipped; an all-null
# group yields null, except `count`, which counts non-nulls and so yields 0).
_PLAIN = {
    "count": "count",
    "min": "min",
    "max": "max",
    "mean": "mean",
    "median": "median",
    "count_distinct": "nunique",
    "bool_and": "all",
    "bool_or": "any",
}

# Reductions needing `min_count=1` so an all-null (or empty) group yields null rather than
# the operator's identity element — `sum` of nothing is not `0`, and `product` is not `1`.
_MIN_COUNT = {"sum": "sum", "product": "prod"}

# Sample-moment reductions (`ddof=1`): a one-row group has no sample variance, so both the
# engine and the libraries return null for it.
_SAMPLE_MOMENT = {"var": "var", "stddev": "std"}

_SUPPORTED = (
    frozenset(_PLAIN)
    | frozenset(_MIN_COUNT)
    | frozenset(_SAMPLE_MOMENT)
    | {
        "count_star",
        "quantile",
    }
)


def supported_aggregate(ir: dict) -> bool:
    """Whether one `aggregate` RelOp node is translatable to the dataframe backends.

    Args:
        ir: The `aggregate` node's JSON IR.

    Returns:
        True when every group key and every reduction in the node is translatable.
    """
    return all(a.get("func") in _SUPPORTED for a in ir["aggregates"])


def _key_columns(df, ir: dict, be: DfBackend) -> tuple[list[str], list[str]]:
    """Materialize the group keys into `df`, returning their column names and output aliases.

    A key that is already a plain column is used in place; a computed key (``group_by(x2=col
    ("x") + 1)``) is evaluated into a private column first. Supporting the computed form here
    is what keeps a `GROUP BY <expression>` on the device instead of dropping the whole chain
    to the CPU engine.
    """
    names: list[str] = []
    aliases: list[str] = []
    for i, gk in enumerate(ir["group_keys"]):
        expr = gk["expr"]
        if expr.get("e") == "col":
            names.append(expr["name"])
        else:
            tmp = f"__bt_gk{i}"
            df[tmp] = be.column(eval_expr(expr, df, be), df)
            names.append(tmp)
        aliases.append(gk["alias"])
    return names, aliases


def _input_column(df, spec: dict, be: DfBackend, slot: int) -> str:
    """The column name a reduction consumes, materializing a computed input if needed.

    Declines a `NaN`-bearing float input. The engine orders `NaN` above every number, so a
    `NaN` in a group makes its `max`, `sum` and `mean` all `NaN`; both dataframe libraries
    instead treat it as missing and reduce the remaining values, which returns a plausible
    number where the engine returns `NaN`. Rather than reconcile that per reduction, the
    stage falls back to the CPU engine — `NaN` in a measure column is rare, and a wrong
    aggregate is not worth the coverage.
    """
    expr = spec["input"]
    if expr.get("e") == "col":
        name = expr["name"]
    else:
        name = f"__bt_ag{slot}"
        df[name] = be.column(eval_expr(expr, df, be), df)
    if be.has_nan(df[name]):
        raise Unsupported(f"aggregate over NaN-bearing column {name!r}")
    return name


def _reduce(grouped, spec: dict, column: str):
    """One reduction over the shared `GroupBy`, as a Series indexed by the group key."""
    func = spec["func"]
    if func == "count_star":
        return grouped.size()
    series = grouped[column]
    if func in _PLAIN:
        return _call(series, _PLAIN[func])
    if func in _MIN_COUNT:
        return _call(series, _MIN_COUNT[func], min_count=1)
    if func in _SAMPLE_MOMENT:
        return _call(series, _SAMPLE_MOMENT[func])
    if func == "quantile":
        return _call(series, "quantile", float(spec["param"]))
    raise Unsupported(f"aggregate {func}")


def _call(series, name: str, *args, **kwargs):
    """Invoke `name` on a grouped column, declining rather than guessing when it is absent.

    cuDF's `GroupBy` surface is a subset of pandas', so a reduction pandas offers may not
    exist on the device. An `AttributeError` here would escape as a crash; `Unsupported`
    routes the stage to the CPU engine, which is the contract every other case follows.
    """
    method = getattr(series, name, None)
    if method is None:
        raise Unsupported(f"grouped {name}")
    try:
        return method(*args, **kwargs)
    except (TypeError, NotImplementedError) as exc:
        raise Unsupported(f"grouped {name}: {exc}") from exc


def aggregate(df, ir: dict, be: DfBackend):
    """Apply one `aggregate` RelOp to `df`, returning the grouped result.

    Args:
        df: The dataframe to reduce.
        ir: The `aggregate` node's JSON IR.
        be: The dataframe backend to compute on.

    Returns:
        A dataframe of one row per group, carrying the key aliases then the aggregate aliases.

    Raises:
        Unsupported: For a reduction outside the translated subset.
    """
    keys, aliases = _key_columns(df, ir, be)
    if not keys:
        return _global(df, ir, be)
    # `dropna=False`: a null key is a group, exactly as it is in the engine and in SQL.
    # The libraries drop it by default, which silently deletes rows from the answer.
    grouped = df.groupby(keys, sort=False, dropna=False)
    columns = {}
    for slot, spec in enumerate(ir["aggregates"]):
        column = None if spec["func"] == "count_star" else _input_column(df, spec, be, slot)
        columns[spec["alias"]] = _reduce(grouped, spec, column)
    out = be.lib.DataFrame(columns) if columns else be.lib.DataFrame(index=grouped.size().index)
    out = out.reset_index()
    # `reset_index` restores the key columns under their *source* names; rename to the
    # aliases the plan asked for, then order as the plan does (keys first).
    renames = {src: alias for src, alias in zip(keys, aliases, strict=True) if src != alias}
    if renames:
        out = out.rename(columns=renames)
    return out[[*aliases, *(a["alias"] for a in ir["aggregates"])]]


def _global(df, ir: dict, be: DfBackend):
    """A keyless aggregate — one row over the whole frame.

    Reached by `agg()` with no `group_by`, and by every distributed *combine* step, so it
    cannot be left to the CPU engine without giving up the whole multi-GPU reduce path.
    Modeled as a single constant group so one code path serves both.
    """
    key = "__bt_all"
    df = df.copy()
    df[key] = 0
    grouped = df.groupby([key], sort=False, dropna=False)
    columns = {}
    for slot, spec in enumerate(ir["aggregates"]):
        column = None if spec["func"] == "count_star" else _input_column(df, spec, be, slot)
        columns[spec["alias"]] = _reduce(grouped, spec, column)
    out = be.lib.DataFrame(columns).reset_index(drop=True)
    return out[[a["alias"] for a in ir["aggregates"]]]
