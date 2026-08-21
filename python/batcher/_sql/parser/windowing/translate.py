"""Window-function handling for the SQL translator.

Groups SELECT-list window items by their (partition, order) spec and maps each
window function to a `ds.window(...)` call. Functions take the translator
instance (`tr`) as their first argument. The spec-reading half — which frame,
which partition keys, which order — lives in `frame.py`.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.core_utils import _alias_of, _unwrap_alias
from batcher._sql.parser.windowing.frame import (
    _WINDOW_AGGS,
    _const_int,
    _resolve_frame,
    _window_order,
    _window_partition,
)
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import lit


def _is_window(p) -> bool:
    return isinstance(_unwrap_alias(p), exp.Window)


def _own_windows(p) -> list:
    """Every window in projection `p` that belongs to *this* query's scope.

    A window inside a scalar subquery belongs to the inner query and is evaluated
    when that subquery is translated, so it is excluded — the same scoping rule
    `_has_aggregate` applies to aggregates.
    """
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


def rewrite_offset_defaults(projections) -> None:
    """Rewrite ``lag(x, n, d)`` / ``lead(x, n, d)`` into constructs the engine has.

    The third argument fills the rows whose offset falls outside the partition — the first
    `n` rows for LAG, the last `n` for LEAD. The window operator has no default parameter,
    and honouring the offset while dropping the default would return NULL where SQL
    returns `d`, so it was refused outright. But "outside the partition" is expressible
    with window functions that *are* supported, so the whole thing is an AST rewrite:

        lag(x, n, d)  OVER w  ->  CASE WHEN row_number() OVER w <= n
                                       THEN d ELSE lag(x, n) OVER w END
        lead(x, n, d) OVER w  ->  CASE WHEN row_number() OVER w > count(*) OVER p - n
                                       THEN d ELSE lead(x, n) OVER w END

    ``COALESCE(lag(x, n), d)`` is the tempting one-liner and it is wrong: it would also
    replace a NULL that `x` genuinely holds *inside* the partition, where SQL keeps NULL.
    The row's position is what decides, not the value.

    `p` is `w` without its ORDER BY, because `count(*)` over an *ordered* window is a
    running count rather than the partition's size.

    Args:
        projections: The SELECT list; window nodes are rewritten in place.
    """
    for p in projections:
        for win in _own_windows(p):
            fn = win.this
            name = type(fn).__name__.lower()
            if name not in ("lag", "lead"):
                continue
            default = fn.args.get("default")
            if default is None:
                continue
            offset = fn.args.get("offset")
            n = _const_int(offset, name) if offset is not None else 1
            fn.set("default", None)
            position = exp.Window(
                this=exp.RowNumber(),
                partition_by=[c.copy() for c in (win.args.get("partition_by") or [])],
                order=win.args["order"].copy() if win.args.get("order") else None,
            )
            if name == "lag":
                out_of_range = exp.LTE(this=position, expression=exp.Literal.number(n))
            else:
                size = exp.Window(
                    this=exp.Count(this=exp.Star()),
                    partition_by=[c.copy() for c in (win.args.get("partition_by") or [])],
                )
                out_of_range = exp.GT(
                    this=position,
                    expression=exp.Sub(this=size, expression=exp.Literal.number(n)),
                )
            win.replace(
                exp.Case(
                    ifs=[exp.If(this=out_of_range, true=default.copy())],
                    default=win.copy(),
                )
            )


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
    if not tr._agg_map:
        return
    for item in items:
        for agg in list(_unwrap_alias(item).find_all(exp.AggFunc)):
            entry = tr._agg_map.get(agg.sql())
            if entry is not None:
                agg.replace(exp.column(entry[0]))


def rewrite_group_keys_in_windows(items, group_expr_alias: dict[str, str]) -> None:
    """Point a window's keys at the columns a *derived* GROUP BY key materialized.

    A window runs after grouping, over a relation whose columns are the group keys and the
    aggregates. A key that was a bare column keeps its name there, but a *derived* key
    (`extract(year FROM d)`, or the `nullif(a, a)` a rolled-up grouping level projects) was
    materialized under an internal `__gk<n>` name — so a `PARTITION BY` naming the original
    expression referenced a column the grouped relation does not have. TPC-DS q70 and q86
    partition on a `CASE` over a rolled-up key and failed exactly there.

    Rewriting is outermost-match-first, so a key that *is* the whole group expression
    becomes one column reference rather than being rebuilt from parts that are gone.
    Aggregates are left alone; `rewrite_aggs_in_windows` owns those and matches them by
    their own SQL text, which this must not disturb.

    Args:
        items: The window projection items to rewrite, in place.
        group_expr_alias: GROUP BY expression SQL text -> its materialized column name.
    """
    if not group_expr_alias:
        return
    for item in items:
        for win in _own_windows(item):
            for key in list(win.args.get("partition_by") or []):
                _sub_group_keys(key, group_expr_alias)
            order = win.args.get("order")
            for ordered in list(order.expressions) if order is not None else []:
                _sub_group_keys(ordered.this, group_expr_alias)


def _sub_group_keys(node, group_expr_alias: dict[str, str]) -> None:
    """Replace, in place, any sub-expression of `node` that is a materialized group key."""
    from batcher._sql.parser.expressions.aggregates import is_agg_node

    alias = group_expr_alias.get(node.sql())
    if alias is not None:
        node.replace(exp.column(alias))
        return
    for child in list(node.iter_expressions()):
        if not is_agg_node(child):
            _sub_group_keys(child, group_expr_alias)


def _inline_named_windows(node) -> None:
    """Copy `WINDOW w AS (PARTITION BY … ORDER BY …)` specs onto `OVER w` refs.

    A named-window reference parses as a `Window` whose `alias` is the window
    name and whose own spec is empty; resolve it from the SELECT's `windows`.
    """
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


#: Ranking functions whose value is a constant when the window has no ORDER BY: with no
#: ordering every row of a partition is a peer of every other, so they all rank together.
#: `row_number` is deliberately absent — it numbers peers rather than tying them.
_UNORDERED_RANK_CONSTANTS = {
    "rank": 1,
    "denserank": 1,
    "cumedist": 1.0,
    "percentrank": 0.0,
}


def _unordered_ranking(tr, ds: Dataset, taken: set[str], projections) -> tuple[Dataset, set[str]]:
    """Compute the ranking windows that need no ORDER BY, which `ds.window` cannot take.

    The window operator requires an ordering for every ranking function, so the translator
    rejected the whole (valid) query. Two shapes do not need one:

    - ``row_number() OVER ()`` numbers the relation, which is exactly `with_row_index`
      — a single counter, so the sequential and parallel paths agree on an order.
    - ``rank()``, ``dense_rank()``, ``cume_dist()`` and ``percent_rank()`` over an
      *unordered* window are constants, because every row is a peer of every other. That
      holds with or without a PARTITION BY, so the partitioning does not matter here.

    ``row_number() OVER (PARTITION BY g)`` is not one of them: it has to number within
    each partition, which needs the operator. It keeps raising.

    Args:
        ds: The relation the windows are computed over.
        projections: The SELECT list.

    Returns:
        `ds` with the resolved columns appended, and the aliases now satisfied.
    """
    handled: set[str] = set()
    for p in projections:
        if not _is_window(p):
            continue
        win = _unwrap_alias(p)
        if _window_order(win):
            continue
        name = type(win.this).__name__.lower()
        alias = _alias_of(p)
        constant = _UNORDERED_RANK_CONSTANTS.get(name)
        if constant is None and not (name == "rownumber" and not _window_partition(win)):
            continue
        # The output name a SQL window carries is a *select-list* alias, so it may repeat
        # a source column's name (`SELECT g, sum(i) OVER (...) AS s` over a table that
        # already has an `s`). The relational operator underneath cannot: its output is
        # appended to the input's columns, so it refuses a collision. Compute into a
        # hidden name and record it, so the projection can read it back under the alias
        # the user wrote. Without this every such query failed outright.
        out = window_output_name(tr, taken, alias)
        if constant is not None:
            ds = ds.with_columns(**{out: lit(constant)})
        else:
            ds = ds.with_row_index(out, offset=1)
        handled.add(alias)
    return ds, handled


def window_output_name(tr, taken: set[str], alias: str) -> str:
    """The physical column a window item is materialized as, recording any rename.

    A select-list alias may legitimately shadow a source column; a relational window
    output may not, because it is appended to the input relation. When the two clash the
    window is computed under a `__bc_wout<n>` name and `tr._win_physical` remembers the
    mapping, which `_projection_map` (and a QUALIFY predicate) consult to read it back.

    Args:
        tr: The translator, for its rename map and counter.
        taken: Names already claimed by the input relation or an earlier window item.
        alias: The select-list alias this window item is written under.

    Returns:
        The column name to materialize the window under.
    """
    if alias not in taken:
        taken.add(alias)
        return alias
    out = f"__bc_wout{tr._win_out_n}"
    tr._win_out_n += 1
    tr._win_physical[alias] = out
    taken.add(out)
    return out


def _window(tr, ds: Dataset, projections) -> Dataset:
    """Apply window functions from the SELECT list, appending output columns.

    Window items are grouped by their (partition_by, order_by, frame) spec; each
    distinct spec becomes one chained `ds.window(...)` call (one call carries a
    single frame, so functions with different frames go to different calls).
    """

    taken = set(ds.columns)
    # Scoped to this call: a subquery translated earlier may have recorded an alias of
    # the same name, and reading its hidden column here would answer with the wrong one.
    tr._win_physical.clear()
    ds, handled = _unordered_ranking(tr, ds, taken, projections)

    # Group window items by their (partition, order, frame) spec, preserving order.
    groups: list[tuple[tuple, tuple, tuple | None, dict]] = []
    for p in projections:
        if not _is_window(p):
            continue
        win = _unwrap_alias(p)
        alias = _alias_of(p)
        if alias in handled:
            continue
        part = _window_partition(win)
        order = _window_order(win)
        func = _window_func(win, order)
        frame = _resolve_frame(win)
        out = window_output_name(tr, taken, alias)

        key = (part, order, frame)
        for gpart, gorder, gframe, funcs in groups:
            if (gpart, gorder, gframe) == key:
                funcs[out] = func
                break
        else:
            groups.append((part, order, frame, {out: func}))

    for part, order, frame, funcs in groups:
        ds = ds.window(
            partition_by=list(part),
            order_by=list(order),
            functions=funcs,
            frame=frame,
        )
    return ds


def _is_boolean_column(tr, node) -> bool:
    """True when `node` is a column the plan says is boolean."""
    import pyarrow as pa

    t = tr.column_type(node)
    return t is not None and pa.types.is_boolean(t)


def _reshaped_window_argument(tr, item, fn, arg):
    """Rewrite a window aggregate's argument into a type the operator's kernels read.

    Two shapes SQL admits and the window kernels do not, both of which the *grouped* form
    of the same aggregate already answers (`grouping._numeric_reduction`) — so the windowed
    spelling failed on a query the `GROUP BY` one does:

    * ``sum(flag) OVER (…)`` over a boolean: SQL sums the TRUEs.
    * ``avg(ts) OVER (…)`` over a DATE or TIMESTAMP: the mean *instant*, which DuckDB
      returns as a TIMESTAMP. The mean is taken over the microsecond count, so the output
      has to be rebuilt into a timestamp — recorded in `tr._win_rewrap`, which the
      projection applies when it reads the window column back.

    Args:
        tr: The translator, for the in-scope column types and the rewrap map.
        item: The projection this window belongs to, for its output alias.
        fn: The aggregate node.
        arg: Its argument node.

    Returns:
        The replacement argument node, or None to leave the argument alone.
    """
    import pyarrow as pa

    from batcher.plan.functions.temporal import from_epoch

    name = type(fn).__name__.lower()
    if name not in ("sum", "avg") or not isinstance(arg, exp.Column):
        return None
    dtype = tr.column_type(arg)
    if dtype is None:
        return None
    if pa.types.is_boolean(dtype):
        return exp.cast(arg.copy(), "BIGINT")
    if name == "avg" and (pa.types.is_date(dtype) or pa.types.is_timestamp(dtype)):
        tr._win_rewrap[_alias_of(item)] = lambda value: from_epoch(value.cast("int64"), "us")
        return exp.cast(exp.cast(arg.copy(), "TIMESTAMP"), "BIGINT")
    return None


def _any_value_func(fn, order):
    """Refuse `any_value(x) OVER (…)`, naming what it means and what to write instead.

    DuckDB implements the windowed `any_value` as *the first non-null value in the frame*,
    which is neither of the fills the runtime has: a forward fill is the **last** non-null
    at or before the row, not the first, so answering with one returns a different column
    on any partition holding more than one distinct value. DuckDB documents the chosen row
    as unspecified besides, so there is nothing to conform to.

    This is a refusal rather than a wrong answer on purpose; it exists as its own branch
    because sqlglot parks `any_value(x)` under an `IgnoreNulls` wrapper, so the message
    used to name an ``IGNORE NULLS`` clause the query never contained.

    Args:
        fn: The `AnyValue` node.
        order: The window's ORDER BY, or an empty tuple.

    Raises:
        NotImplementedError: Always.
    """
    del fn, order
    raise NotImplementedError(
        "any_value(x) OVER (…) is not supported: DuckDB answers it with the first "
        "non-null value in the frame, which is not one of the runtime's window "
        "primitives, and the row it names is documented as unspecified. Use "
        "first_value(x) / min(x) / max(x) OVER (…), whose answers are defined"
    )


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


#: Window functions whose first argument is a *value* the engine reads per row. Each takes
#: a materialized column, so an argument that is any other expression has to be computed
#: into one first.
_VALUE_ARG_FUNCS = frozenset(
    {
        "sum",
        "avg",
        "min",
        "max",
        "count",
        "lag",
        "lead",
        "firstvalue",
        "lastvalue",
        "nthvalue",
    }
)


def hoist_window_args(tr, ds: Dataset, projections) -> Dataset:
    """Materialize any window argument, `PARTITION BY` key or `ORDER BY` key that is not
    already a plain column.

    `ds.window` binds each function and key to a column name, so the translator required
    each to be written as one and rejected everything else with *"supports a plain column
    only"*. That refused ordinary analytics SQL — ``sum(price * qty) OVER (...)``, the
    conditional running total ``sum(CASE WHEN … THEN … END) OVER (...)``, and the monthly
    ranking ``rank() OVER (PARTITION BY date_trunc('month', ts) ORDER BY x)`` — none of
    which is a corner case.

    All three are ordinary scalar expressions, so computing each into a hidden column
    before the window runs is enough; nothing about the window operator has to change. It
    already accepts an `Expr` key, which is why this is a translator gap rather than an
    engine one. The hidden columns use the `__bc_` prefix the projection builder already
    hides from ``SELECT *``.

    A **key** is deduplicated by its SQL text while an *argument* is not, and that
    difference is load-bearing: `_window` groups window items by their
    ``(partition, order, frame)`` spec so equal specs share one `Window` operator, and two
    windows partitioned on the same expression must therefore land on the same hoisted
    column name or they would split into two operators computing the same partitioning
    twice.

    Args:
        tr: The translator, for lowering the expressions.
        ds: The relation the windows are computed over.
        projections: The SELECT list (plus any hoisted nested-window items).

    Returns:
        `ds` with one added column per hoisted expression, or `ds` unchanged.
    """
    computed: dict[str, object] = {}
    # SQL text -> hoisted column name, so one expression is computed once per query.
    keys_seen: dict[str, str] = {}

    def hoist_key(node) -> None:
        """Replace a computed key node in place with a reference to a hidden column."""
        sql = node.sql()
        name = keys_seen.get(sql)
        if name is None:
            name = f"__bc_wkey{tr._win_arg_n}"
            tr._win_arg_n += 1
            keys_seen[sql] = name
            computed[name] = tr._scalar(node)
        node.replace(exp.column(name))

    for p in projections:
        for win in _own_windows(p):
            fn = win.this
            if type(fn).__name__.lower() == "ignorenulls":
                fn = fn.this
            if type(fn).__name__.lower() in _VALUE_ARG_FUNCS:
                arg = fn.this
                # `sum(flag) OVER (…)` over a boolean column: SQL sums the TRUEs, the
                # window kernels take numbers. Widening it here is the same rule the
                # GROUP BY path applies (`grouping._numeric_reduction`); without it the
                # windowed spelling failed on a query the grouped one answers.
                reshaped = _reshaped_window_argument(tr, p, fn, arg)
                if reshaped is not None:
                    arg = reshaped
                    fn.set("this", arg)
                # `count(*)`/`count()` has no value argument, and a plain column is
                # already what the window operator wants.
                if arg is not None and not isinstance(arg, (exp.Column, exp.Star)):
                    # A prefix of its own: `hoist_nested_windows` already mints
                    # `__bc_win<n>` for a nested window's *output*, and sharing the
                    # counter-free prefix made the two collide on `__bc_win0` whenever a
                    # query had both.
                    name = f"__bc_warg{tr._win_arg_n}"
                    tr._win_arg_n += 1
                    computed[name] = tr._scalar(arg)
                    fn.set("this", exp.column(name))
            for key in list(win.args.get("partition_by") or []):
                if not isinstance(key, exp.Column):
                    hoist_key(key)
            order = win.args.get("order")
            for ordered in list(order.expressions) if order is not None else []:
                if not isinstance(ordered.this, exp.Column):
                    hoist_key(ordered.this)
    return ds.with_columns(**computed) if computed else ds


def _window_func(win, order):
    """Map a window function node to a `ds.window` functions-value."""
    fn = win.this
    if type(fn).__name__.lower() == "ignorenulls":
        # sqlglot parks `any_value(x)` under `IgnoreNulls` even with no such clause
        # written, so the IGNORE-NULLS handler answered a plain `any_value(x) OVER (…)`
        # with an error naming a clause the query never used.
        if type(fn.this).__name__.lower() == "anyvalue":
            return _any_value_func(fn.this, order)
        return _ignore_nulls_func(win, fn.this, order)
    name = type(fn).__name__.lower()
    if name == "anyvalue":
        return _any_value_func(fn, order)

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

    if name in _WINDOW_AGGS:
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
        return (_WINDOW_AGGS[name], arg.name)

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
