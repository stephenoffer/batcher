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


def _own_windows(p) -> list:
    """Every window in projection `p` that belongs to *this* query's scope.

    A window inside a scalar subquery belongs to the inner query and is evaluated
    when that subquery is translated, so it is excluded — the same scoping rule
    `_has_aggregate` applies to aggregates.
    """
    from sqlglot import expressions as exp

    return [
        w for w in _unwrap_alias(p).find_all(exp.Window) if w.find_ancestor(exp.Subquery) is None
    ]


def _has_window(p) -> bool:
    """Whether `p` contains a window function anywhere, not only as the whole item.

    `_is_window` only sees a projection that *is* a window (`sum(x) OVER () AS r`).
    A window nested inside a larger expression (`sum(x) OVER () + 1`, or TPC-DS q98's
    `sum(x) * 100 / sum(sum(x)) OVER (...)`) is just as much a window query, and
    treating it as an ordinary scalar is what made those fail.
    """
    return bool(_own_windows(p))


def hoist_nested_windows(projections) -> tuple[list, list]:
    """Replace windows nested inside larger projections with `__bc_win<n>` columns.

    A window that *is* the whole projection is left alone — `_window` already names its
    output column with the user's alias and `_projection_map` reads it back. A window
    nested in a bigger expression cannot be handled that way, because the surrounding
    arithmetic has to be evaluated *after* the window column exists. Each such window
    becomes a synthetic `__bc_win<n>` item computed by `_window`, and the projection
    keeps only a reference to it. The `__bc_` prefix keeps it out of `SELECT *`.

    Args:
        projections: The SELECT list.

    Returns:
        The rewritten projections and the synthetic window items to materialize.
    """
    from sqlglot import expressions as exp

    synthetic: list = []
    out: list = []
    for p in projections:
        if _is_window(p):  # a whole-item window keeps the existing direct path
            out.append(p)
            continue
        nested = _own_windows(p)
        if not nested:
            out.append(p)
            continue
        p = p.copy()
        for win in _own_windows(p):
            alias = f"__bc_win{len(synthetic)}"
            # Copy before replacing: `synthetic` keeps the window expression itself,
            # the projection keeps only a reference to its output column.
            synthetic.append(exp.alias_(win.copy(), alias))
            win.replace(exp.column(alias))
        out.append(p)
    return out, synthetic


def rewrite_aggs_in_windows(tr, items) -> None:
    """Point aggregates inside window items at the columns the GROUP BY produced.

    In an aggregate query a window runs *after* grouping, so `sum(sum(x)) OVER ()` is a
    window sum over the already-grouped `sum(x)`. The inner aggregate was registered by
    `_aggregate` and materialized as a column; replacing it in place with a reference to
    that column is what lets the window pass see a plain column argument (which is all
    `_window_func` / `_window_order` / `_window_partition` accept).

    The outer aggregate is deliberately not in `_agg_map` — `_aggregate` skips any
    aggregate whose parent is a `Window` — so only the inner one is rewritten.

    Args:
        tr: The translator, read for its live `_agg_map`.
        items: The window projection items to rewrite, in place.
    """
    from sqlglot import expressions as exp

    if not tr._agg_map:
        return
    for item in items:
        for agg in list(_unwrap_alias(item).find_all(exp.AggFunc)):
            entry = tr._agg_map.get(agg.sql())
            if entry is not None:
                agg.replace(exp.column(entry[0]))


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
        # The named window may also carry a frame (`WINDOW w AS (... ROWS ...)`);
        # copy it too, else `OVER w` silently loses the frame and runs the default.
        if src.args.get("spec") is not None and w.args.get("spec") is None:
            w.set("spec", src.args["spec"].copy())


def _window(ds: Dataset, projections) -> Dataset:
    """Apply window functions from the SELECT list, appending output columns.

    Window items are grouped by their (partition_by, order_by, frame) spec; each
    distinct spec becomes one chained `ds.window(...)` call (one call carries a
    single frame, so functions with different frames go to different calls).
    """

    # Group window items by their (partition, order, frame) spec, preserving order.
    groups: list[tuple[tuple, tuple, tuple | None, dict]] = []
    for p in projections:
        if not _is_window(p):
            continue
        win = _unwrap_alias(p)
        alias = _alias_of(p)
        part = _window_partition(win)
        order = _window_order(win)
        func = _window_func(win, order)
        frame = _resolve_frame(win)

        key = (part, order, frame)
        for gpart, gorder, gframe, funcs in groups:
            if (gpart, gorder, gframe) == key:
                funcs[alias] = func
                break
        else:
            groups.append((part, order, frame, {alias: func}))

    for part, order, frame, funcs in groups:
        ds = ds.window(
            partition_by=list(part),
            order_by=list(order),
            functions=funcs,
            frame=frame,
        )
    return ds


def _is_agg_window(win) -> bool:
    """Whether the window function is an aggregate (frames apply to these only)."""
    return type(win.this).__name__.lower() in {"sum", "avg", "min", "max", "count"}


# Positional value functions that pick a frame's first / last / nth row.
_FRAMED_VALUE = {"firstvalue", "lastvalue", "nthvalue"}


def _resolve_frame(win) -> tuple | None:
    """The window frame this function should run under.

    Aggregates and the positional value functions (`first_value`/`last_value`/
    `nth_value`) honour an explicit frame. When a value function has *no* explicit
    frame but *does* have an ORDER BY, SQL's default frame is
    ``RANGE UNBOUNDED PRECEDING TO CURRENT ROW`` — which makes `last_value` /
    `nth_value` the *running* value (the current peer's / null-until-the-nth-peer)
    rather than the whole-partition value. `first_value` is the same either way, so it
    stays frameless. Ranking / LAG / LEAD ignore frames (as SQL does)."""
    name = type(win.this).__name__.lower()
    explicit = _window_frame(win)
    if _is_agg_window(win):
        return explicit
    if name in _FRAMED_VALUE:
        if explicit is not None:
            return explicit
        # Emit the default running frame only for last_value / nth_value (first_value's
        # frame start is unbounded-preceding, so its result is frame-independent).
        if win.args.get("order") is not None and name in {"lastvalue", "nthvalue"}:
            return (None, 0, "range")
        return None
    return None  # ranking / LAG / LEAD: the frame is meaningless, ignore it


def _window_frame(win) -> tuple[int | None, int | None] | None:
    """Translate an explicit ``ROWS BETWEEN …`` frame to a ``(start, end)`` offset pair.

    Returns ``None`` when there is no explicit frame (the default cumulative/whole
    partition behavior is unchanged). Only ROWS frames are supported; an explicit
    numeric RANGE/GROUPS frame raises, since the engine frame is row-indexed.
    """
    spec = win.args.get("spec")
    if spec is None:
        return None
    if spec.args.get("exclude") is not None:
        # A frame EXCLUDE (TIES / GROUP / CURRENT ROW / NO OTHERS) changes which peer
        # rows the frame drops; the engine has no such option, so honoring the frame
        # while dropping EXCLUDE would silently give the wrong answer. Reject it.
        raise NotImplementedError("window frame EXCLUDE (TIES/GROUP/CURRENT ROW) is not supported")
    kind = (spec.args.get("kind") or "").upper()
    if kind != "ROWS":
        if spec.args.get("start") is not None or spec.args.get("end") is not None:
            raise NotImplementedError("only ROWS window frames are supported (not RANGE/GROUPS)")
        return None
    start = spec.args.get("start")
    end = spec.args.get("end")
    # A single-bound frame — ``ROWS N PRECEDING`` (no BETWEEN) — is shorthand for
    # ``ROWS BETWEEN N PRECEDING AND CURRENT ROW``. sqlglot leaves ``end`` unset in
    # that case; defaulting it to CURRENT ROW (0) avoids treating it as UNBOUNDED
    # FOLLOWING (a silently wrong whole-tail sum).
    if end is None and start is not None:
        end_bound: int | None = 0
    else:
        end_bound = _frame_bound(end, spec.args.get("end_side"))
    return (_frame_bound(start, spec.args.get("start_side")), end_bound)


def _frame_bound(value, side) -> int | None:
    """A frame bound → signed row offset: UNBOUNDED→None, CURRENT ROW→0, n P/F→∓n."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.upper()
        if text == "UNBOUNDED":
            return None
        if text == "CURRENT ROW":
            return 0
        raise NotImplementedError(f"unsupported window frame bound {value!r}")
    n = int(value.this)  # a numeric Literal node
    sided = (side or "").upper()
    if sided == "PRECEDING":
        return -n
    if sided == "FOLLOWING":
        return n
    raise NotImplementedError(f"window frame bound needs PRECEDING/FOLLOWING, got {side!r}")


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


def _const_int(node, ctx: str) -> int:
    """Evaluate a constant integer argument, handling negatives.

    sqlglot parses a negative literal (``-1``) as a ``Neg`` wrapping a ``Literal``, so a
    naive ``int(node.this)`` reads the inner node and raises ``TypeError``. ``to_py()``
    folds ``Neg``/``Literal`` to a Python value; a non-constant argument is rejected.
    """
    from sqlglot import expressions as exp

    if isinstance(node, (exp.Literal, exp.Neg)):
        try:
            return int(node.to_py())
        except (TypeError, ValueError):
            pass  # a non-constant frame bound -> fall through and reject it below
    raise NotImplementedError(f"window function {ctx!r} requires a constant integer argument")


def _ignore_nulls_func(win, fn, order):
    """Map `<value fn>(x IGNORE NULLS) OVER (...)` onto the runtime's fill primitives.

    `IGNORE NULLS` makes a value function skip nulls when picking its answer. Two shapes
    are exactly the engine's existing fills, so they need no new operator:

    * ``last_value(x IGNORE NULLS)`` over the default frame (everything up to the current
      row) is "the most recent non-null so far" — a **forward fill**.
    * ``first_value(x IGNORE NULLS)`` over ``CURRENT ROW AND UNBOUNDED FOLLOWING`` is "the
      next non-null from here" — a **backward fill**.

    Other combinations (`lag`/`lead`/`nth_value` with IGNORE NULLS, or a value function
    over some other frame) need per-row null-skipping the runtime does not have, and are
    rejected rather than silently answered with the null-*respecting* result — which would
    be a wrong answer, not a slower one.

    Args:
        win: The `Window` node, read for its frame spec.
        fn: The inner function node that `IgnoreNulls` wraps.
        order: The window's ORDER BY, required by every value function.

    Returns:
        The `ds.window` functions-value — a `(fill, column)` pair.
    """
    from sqlglot import expressions as exp

    name = type(fn).__name__.lower()
    if not order:
        raise NotImplementedError(f"window function {name!r} requires ORDER BY")
    arg = fn.this
    if not isinstance(arg, exp.Column):
        raise NotImplementedError("IGNORE NULLS supports a single plain column argument only")

    def bound(key: str) -> str:
        """A frame bound as an upper-case keyword, or "" when it is an offset literal.

        An offset bound (`1 PRECEDING`) parses as a `Literal`, not a keyword string, so it
        must not be coerced — it simply is not one of the shapes handled here.
        """
        if spec is None:
            return ""
        v = spec.args.get(key)
        return v.upper() if isinstance(v, str) else ""

    spec = win.args.get("spec")
    kind, start, end = bound("kind"), bound("start"), bound("end")
    # The default frame (no spec) runs from the partition start to the current row, which
    # is what makes `last_value` a forward fill.
    trailing = spec is None or (start == "UNBOUNDED" and end == "CURRENT ROW")
    leading = kind in {"ROWS", "RANGE"} and start == "CURRENT ROW" and end == "UNBOUNDED"

    if name == "lastvalue" and trailing:
        return ("forward_fill", arg.name)
    if name == "firstvalue" and leading:
        return ("backward_fill", arg.name)
    raise NotImplementedError(
        f"{name}(x IGNORE NULLS) over this frame is not supported. Supported: "
        "last_value(x IGNORE NULLS) over the default frame (a forward fill) and "
        "first_value(x IGNORE NULLS) OVER (... ROWS BETWEEN CURRENT ROW AND UNBOUNDED "
        "FOLLOWING) (a backward fill)"
    )


def _window_func(win, order):
    """Map a window function node to a `ds.window` functions-value."""
    from sqlglot import expressions as exp

    fn = win.this
    if type(fn).__name__.lower() == "ignorenulls":
        return _ignore_nulls_func(win, fn.this, order)
    name = type(fn).__name__.lower()

    # Ranking family (no input; needs ORDER BY). `percent_rank`/`cume_dist` produce
    # a fraction; the runtime supports all of these.
    ranking = {
        "rownumber": "row_number",
        "rank": "rank",
        "denserank": "dense_rank",
        "percentrank": "percent_rank",
        "cumedist": "cume_dist",
    }
    if name in ranking:
        if not order:
            raise NotImplementedError(f"window ranking function {name!r} requires ORDER BY")
        return ranking[name]

    # NTILE(n): a no-input ranking function whose bucket count is a constant.
    if name == "ntile":
        if not order:
            raise NotImplementedError("window function 'ntile' requires ORDER BY")
        n = fn.this
        if n is None or getattr(n, "this", None) is None:
            raise NotImplementedError("ntile(n) requires a constant bucket count")
        # Via `_const_int`, not `int(n.this)`: a negative literal is a `Neg` wrapping a
        # `Literal`, so the naive read raised a bare `TypeError` from inside sqlglot.
        buckets = _const_int(n, "ntile")
        # The engine's `ntile` clamps `buckets.max(1)`, so an unvalidated 0 or negative
        # silently put every row in bucket 1 instead of failing. `bt.ntile()` already
        # rejects `n < 1`; the SQL front-end must agree (DuckDB errors here too).
        if buckets < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(f"ntile(n) requires n >= 1, got {buckets}")
        return ("ntile", buckets)

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
        "nthvalue": "nth_value",
    }
    if name in value:
        if not order:
            raise NotImplementedError(f"window function {name!r} requires ORDER BY")
        arg = fn.this
        if not isinstance(arg, exp.Column):
            raise NotImplementedError(f"window {name} supports a plain column argument only")
        if name in ("lag", "lead"):
            if fn.args.get("default") is not None:
                # A default value fills the out-of-range rows; the engine has no
                # such parameter, so honoring the offset while dropping the default
                # would silently return NULL where SQL returns the default. Reject.
                raise NotImplementedError(
                    f"{name}(expr, offset, default) with a default value is not supported yet"
                )
            off = fn.args.get("offset")
            # A negative offset (`lag(x, -1)`) flips direction (== `lead(x, 1)`), which the
            # engine supports; sqlglot wraps it in a `Neg` node, so `int(off.this)` would
            # read the inner Literal and crash. `_const_int` evaluates the constant.
            return (value[name], arg.name, _const_int(off, name) if off is not None else 1)
        if name == "nthvalue":
            n = fn.args.get("offset")
            if n is None:
                raise NotImplementedError("nth_value(expr, n) requires a constant N")
            # The N rides in the offset slot of the (func, column, offset) spec. A
            # non-positive N yields all-NULL (matching DuckDB), so it is not rejected here.
            return ("nth_value", arg.name, _const_int(n, "nth_value"))
        return (value[name], arg.name)

    raise NotImplementedError(f"unsupported window function: {name}")
