"""Window-function handling for the SQL translator.

Groups SELECT-list window items by their (partition, order) spec and maps each
window function to a `ds.window(...)` call. Functions take the translator
instance (`tr`) as their first argument.
"""

from __future__ import annotations

from batcher._sql.parser.core_utils import _alias_of, _unwrap_alias
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import lit


def _is_window(p) -> bool:
    from sqlglot import expressions as exp

    return isinstance(_unwrap_alias(p), exp.Window)


def _inline_named_windows(node) -> None:
    """Copy `WINDOW w AS (PARTITION BY … ORDER BY …)` specs onto `OVER w` refs.

    A named-window reference parses as a `Window` whose `alias` is the window
    name and whose own spec is empty; resolve it from the SELECT's `windows`.
    """
    from sqlglot import expressions as exp

    named = {w.this.name: w for w in (node.args.get("windows") or [])}
    if not named:
        return
    for w in node.find_all(exp.Window):
        ref = w.alias
        if not ref or ref not in named or w.args.get("partition_by"):
            continue
        src = named[ref]
        if src.args.get("partition_by"):
            w.set("partition_by", [c.copy() for c in src.args["partition_by"]])
        if src.args.get("order"):
            w.set("order", src.args["order"].copy())


def _window(ds: Dataset, projections) -> Dataset:
    """Apply window functions from the SELECT list, appending output columns.

    Window items are grouped by their (partition_by, order_by) spec; each
    distinct spec becomes one chained `ds.window(...)` call.
    """

    # Group window items by their (partition, order) spec, preserving order.
    groups: list[tuple[tuple, tuple, dict]] = []
    for p in projections:
        if not _is_window(p):
            continue
        win = _unwrap_alias(p)
        alias = _alias_of(p)
        part = _window_partition(win)
        order = _window_order(win)
        func = _window_func(win, order)

        key = (part, order)
        for gpart, gorder, funcs in groups:
            if (gpart, gorder) == key:
                funcs[alias] = func
                break
        else:
            groups.append((part, order, {alias: func}))

    for part, order, funcs in groups:
        ds = ds.window(
            partition_by=list(part),
            order_by=list(order),
            functions=funcs,
        )
    return ds


def _window_partition(win) -> tuple:
    from sqlglot import expressions as exp

    cols = win.args.get("partition_by") or []
    keys = []
    for c in cols:
        if not isinstance(c, exp.Column):
            raise NotImplementedError("window PARTITION BY supports plain columns only")
        keys.append(c.name)
    return tuple(keys)


def _window_order(win) -> tuple:
    from sqlglot import expressions as exp

    order = win.args.get("order")
    if order is None:
        return ()
    specs = []
    for o in order.expressions:
        target = o.this
        if not isinstance(target, exp.Column):
            raise NotImplementedError("window ORDER BY supports plain columns only")
        specs.append((target.name, bool(o.args.get("desc"))))
    return tuple(specs)


def _window_func(win, order):
    """Map a window function node to a `ds.window` functions-value."""
    from sqlglot import expressions as exp

    fn = win.this
    name = type(fn).__name__.lower()

    ranking = {"rownumber": "row_number", "rank": "rank", "denserank": "dense_rank"}
    if name in ranking:
        if not order:
            raise NotImplementedError(f"window ranking function {name!r} requires ORDER BY")
        return ranking[name]

    aggregates = {"sum": "sum", "avg": "avg", "min": "min", "max": "max", "count": "count"}
    if name in aggregates:
        # No ORDER BY → whole-partition aggregate; ORDER BY present → running
        # (cumulative) aggregate over the ordered partition (RANGE frame).
        arg = fn.this
        # COUNT(*) OVER (...) → count of a non-null constant = count of rows.
        if name == "count" and (arg is None or isinstance(arg, exp.Star)):
            return ("count", lit(1))
        if not isinstance(arg, exp.Column):
            raise NotImplementedError(
                "window aggregate supports a single plain column argument only"
            )
        return (aggregates[name], arg.name)

    value = {
        "lag": "lag",
        "lead": "lead",
        "firstvalue": "first_value",
        "lastvalue": "last_value",
    }
    if name in value:
        if not order:
            raise NotImplementedError(f"window function {name!r} requires ORDER BY")
        arg = fn.this
        if not isinstance(arg, exp.Column):
            raise NotImplementedError(f"window {name} supports a plain column argument only")
        if name in ("lag", "lead"):
            off = fn.args.get("offset")
            return (value[name], arg.name, int(off.this) if off is not None else 1)
        return (value[name], arg.name)

    raise NotImplementedError(f"unsupported window function: {name}")
