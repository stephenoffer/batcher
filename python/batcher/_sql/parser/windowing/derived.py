"""Window forms answered by rewriting them into windows the engine already computes.

Layer: `_sql` (surface). Two SQL window features have no operator of their own, and each is
*exactly* expressible with ones that do, so they are AST rewrites run before the window
pass — the same approach `rewrite_offset_defaults` takes for ``lag(x, n, d)``:

- ``lag``/``lead`` with ``IGNORE NULLS`` become ``nth_value(... IGNORE NULLS)`` (or
  ``last_value``) over a frame that starts just past the current row
  (`rewrite_ignore_nulls_navigation`).
- A frame ``EXCLUDE`` becomes the same aggregate over the frame's pieces either side of
  what is excluded, combined null-safely (`rewrite_frame_exclusions`).

Neither introduces state: each output is a combination of ordinary window aggregates, which
already run partition-parallel, distributed and spilled, so the rewrites inherit all three.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.windowing.frame import _const_int, _window_frame, window_agg

__all__ = [
    "SHARED_FRAME",
    "apply_shared_frames",
    "rewrite_frame_exclusions",
    "rewrite_ignore_nulls_navigation",
]

#: `Window.meta` key marking the pieces of one excluded frame. `_window` computes every
#: window carrying the same token in **one** operator, each under its own frame, because the
#: pieces of a ``ROWS`` frame are only complementary when they read one physical row order:
#: over tied (or absent) ORDER BY keys two separately sorted operators may order the ties
#: differently, and the pieces would then miss or double-count a peer.
SHARED_FRAME = "bc_shared_frame"


def _own_windows(p) -> list:
    """The windows of projection `p` in this query's scope (not inside a subquery)."""
    return [w for w in p.find_all(exp.Window) if w.find_ancestor(exp.Subquery) is None]


def rewrite_ignore_nulls_navigation(projections) -> None:
    """Rewrite ``lag``/``lead`` with ``IGNORE NULLS`` into framed value functions.

    ``lead(x, k) IGNORE NULLS`` is the k-th non-null `x` *after* the current row, which is
    ``nth_value(x, k) IGNORE NULLS`` over ``ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED
    FOLLOWING``. ``lag(x, 1) IGNORE NULLS`` is the last non-null before it, ``last_value(x)
    IGNORE NULLS`` over ``ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING``; a larger lag
    counts back from the row, so it reads the same ``nth_value`` over the *reversed*
    ordering, where the rows before the current one are the rows after it. A negative offset
    swaps the direction, as it does without ``IGNORE NULLS``.

    A default fills the rows with fewer than `k` non-null neighbours, and with nulls skipped
    that is exactly the rows the frame answers NULL for, so it is a ``COALESCE`` here, unlike
    the null-respecting form, where a genuine NULL inside the partition must survive.

    Args:
        projections: The SELECT list; window nodes are rewritten in place.
    """
    for p in projections:
        for win in _own_windows(p):
            wrapped = win.this
            if not isinstance(wrapped, exp.IgnoreNulls):
                continue
            fn = wrapped.this
            name = type(fn).__name__.lower()
            if name not in ("lag", "lead"):
                continue
            offset = fn.args.get("offset")
            k = _const_int(offset, name) if offset is not None else 1
            forward = (name == "lead") == (k >= 0)
            k = abs(k)
            # `lag(x, 0)` is the row itself, nulls included, and never out of range.
            replacement = fn.this.copy() if k == 0 else _navigation_window(win, fn.this, k, forward)
            default = fn.args.get("default")
            if default is not None and k != 0:
                replacement = exp.Coalesce(this=replacement, expressions=[default.copy()])
            win.replace(replacement)


def _navigation_window(win, value, k: int, forward: bool):
    """The framed value window reading the k-th non-null row after (or before) this one."""
    order = win.args.get("order")
    if order is None:
        raise NotImplementedError("lag/lead IGNORE NULLS requires ORDER BY")
    if not forward and k == 1:
        fn, order_node = exp.LastValue(this=value.copy()), order.copy()
        spec = _spec("ROWS", None, -1)
    else:
        fn = exp.NthValue(this=value.copy(), offset=exp.Literal.number(k))
        order_node = order.copy() if forward else _reversed(order)
        spec = _spec("ROWS", 1, None)
    return exp.Window(
        this=exp.IgnoreNulls(this=fn),
        partition_by=[c.copy() for c in (win.args.get("partition_by") or [])],
        order=order_node,
        spec=spec,
    )


def _reversed(order):
    """`order` with every key's direction and null placement flipped."""
    flipped = order.copy()
    for key in flipped.expressions:
        key.set("desc", not key.args.get("desc"))
        key.set("nulls_first", not key.args.get("nulls_first"))
    return flipped


def _spec(kind: str, start: int | None, end: int | None):
    """A ``<kind> BETWEEN start AND end`` spec from signed offsets (None = unbounded)."""

    def bound(offset: int | None, unbounded_side: str) -> dict:
        if offset is None:
            return {"": "UNBOUNDED", "_side": unbounded_side}
        if offset == 0:
            return {"": "CURRENT ROW"}
        side = "PRECEDING" if offset < 0 else "FOLLOWING"
        return {"": exp.Literal.number(abs(offset)), "_side": side}

    args: dict = {"kind": kind}
    for edge, offset, side in (("start", start, "PRECEDING"), ("end", end, "FOLLOWING")):
        for suffix, value in bound(offset, side).items():
            args[edge + suffix] = value
    return exp.WindowSpec(**args)


#: How a piece of an excluded frame is combined with the next, per aggregate. Every fold
#: skips a NULL piece — an empty piece, or one holding only NULLs — which is what the
#: aggregate over the union of the pieces does.
def _null_skipping(op):
    """``coalesce(a op b, a, b)``: `op` of two pieces, or whichever one is not NULL."""
    return lambda a, b: exp.Coalesce(
        this=op(this=a, expression=b), expressions=[a.copy(), b.copy()]
    )


_FOLDS = {
    "sum": _null_skipping(exp.Add),
    "min": lambda a, b: exp.Least(this=a, expressions=[b]),
    "max": lambda a, b: exp.Greatest(this=a, expressions=[b]),
    "bool_and": _null_skipping(exp.And),
    "bool_or": _null_skipping(exp.Or),
    "count": lambda a, b: exp.Add(this=a, expression=b),
}


def rewrite_frame_exclusions(projections) -> None:
    """Rewrite windows whose frame carries an ``EXCLUDE`` clause.

    A frame ``[s, e]`` around the current row, minus the current row (``ROWS``) or the
    current peer group (``GROUPS``), is the two frames ``[s, -1]`` and ``[1, e]`` in the same
    units. ``EXCLUDE TIES`` is ``EXCLUDE GROUP`` with the current row put back. So the
    aggregate over the excluded frame is the aggregate's own fold over those pieces: a
    null-skipping sum, `least`/`greatest`, a count sum, and an average rebuilt from the sum
    and count. Subtracting the current row from the unexcluded result, the tempting
    shortcut, is wrong on a NULL row, an empty remainder and a non-finite float.

    ``EXCLUDE NO OTHERS`` is the default and is dropped. A frame that does not reach the
    excluded rows is unchanged by the clause.

    Args:
        projections: The SELECT list; window nodes are rewritten in place.

    Raises:
        NotImplementedError: For an exclusion no split expresses exactly (``GROUP``/``TIES``
            under ``ROWS``, anything under a value-offset ``RANGE``), for an aggregate with
            no fold here, and for a frame the exclusion leaves empty.
    """
    token = 0
    for p in projections:
        for win in _own_windows(p):
            spec = win.args.get("spec")
            excluded = spec.args.get("exclude") if spec is not None else None
            if excluded is None:
                continue
            mode = excluded.name.upper()
            plain = win.copy()
            plain.args["spec"].set("exclude", None)
            if mode == "NO OTHERS":
                win.replace(plain)
                continue
            start, end, units = _frame_of(plain)
            if (start is not None and start > 0) or (end is not None and end < 0):
                win.replace(plain)  # the excluded rows lie outside the frame
                continue
            if (start, end) == (None, None):
                units = "groups"  # the whole partition, in any units
            agg = window_agg(win.this)
            value = agg[1] if agg is not None else None
            pieces = _pieces(win, mode, start, end, units, f"x{token}", value)
            win.replace(_combined(win, pieces, mode == "TIES"))
            token += 1


def _pieces(win, mode: str, start, end, units: str, token: str, value) -> list[tuple]:
    """The ``(partition, order, spec, token)`` windows the excluded frame splits into.

    ``[s, e]`` minus the current row (``ROWS``) or its peer group (``GROUPS``) is
    ``[s, -1]`` and ``[1, e]`` in the same units. Under ``GROUPS``, ``EXCLUDE CURRENT ROW``
    keeps the rest of the peer group as well, and "every peer but this row" is itself a
    ``ROWS`` split: the rows before and after this one within a partition of the peers.
    """
    splits = {"CURRENT ROW": ("rows", "groups"), "GROUP": ("groups",), "TIES": ("groups",)}
    if units not in splits[mode]:
        raise NotImplementedError(
            f"window frame EXCLUDE {mode} is not supported over a {units.upper()} frame with "
            "these bounds: EXCLUDE CURRENT ROW takes a ROWS or GROUPS frame, EXCLUDE GROUP "
            "and TIES a GROUPS frame, and a RANGE frame only when its bounds are UNBOUNDED "
            "or CURRENT ROW"
        )
    partition = list(win.args.get("partition_by") or [])
    order = win.args.get("order")
    # With no ORDER BY every row is a peer of every other: there are no neighbouring
    # groups, and the one peer group is the whole partition.
    edges = [(start, -1)] if order is not None and (start is None or start < 0) else []
    edges += [(1, end)] if order is not None and (end is None or end > 0) else []
    out = [(partition, order, _spec(units.upper(), a, b), token) for a, b in edges]
    if mode == "CURRENT ROW" and units == "groups":
        # A ROWS frame needs an ordering. Within one peer group the ORDER BY keys are
        # constant, so any order serves, provided both halves read the same one — which
        # the shared operator guarantees. Unordered, the value itself is ordered on.
        keys = [key.this for key in order.expressions] if order is not None else []
        if order is None and value is None:
            raise NotImplementedError(
                "count(*) OVER (... EXCLUDE CURRENT ROW) without ORDER BY is count(*) - 1; "
                "write it that way"
            )
        by = order if order is not None else exp.Order(expressions=[exp.Ordered(this=value.copy())])
        peers = partition + keys
        out += [(peers, by, _spec("ROWS", a, b), f"{token}p") for a, b in ((None, -1), (1, None))]
    return out


def _frame_of(win) -> tuple[int | None, int | None, str]:
    """`win`'s frame, reading a peer-bounded ``RANGE`` frame as the ``GROUPS`` frame it is.

    With no explicit frame an ordered window runs over ``RANGE UNBOUNDED PRECEDING TO
    CURRENT ROW`` and an unordered one over the whole partition. A ``RANGE`` frame whose
    bounds are only UNBOUNDED or CURRENT ROW counts peers, not key values, so it is the
    ``GROUPS`` frame with the same bounds, and splits the way that one does.
    """
    frame = _window_frame(win)
    if frame is None:
        return (None, 0, "groups") if win.args.get("order") else (None, None, "rows")
    start, end, units = frame
    if units == "range" and start in (None, 0) and end in (None, 0):
        return start, end, "groups"
    return frame


def apply_shared_frames(ds, frames: dict[str, tuple]):
    """Give each function of `ds`'s top `Window` named in `frames` that frame.

    The pieces of an excluded frame are computed by one operator (see `SHARED_FRAME`), which
    `ds.window` cannot express because it takes one frame per call.

    Args:
        ds: A Dataset whose plan is the `Window` holding the pieces.
        frames: Output name -> the ``(start, end, units)`` frame it runs under.

    Returns:
        `ds` with those functions' frames set.
    """
    import dataclasses

    from batcher.plan.logical import Window
    from batcher.plan.logical.window import WindowFrame

    node = ds._plan
    if not isinstance(node, Window):
        raise NotImplementedError("window frame EXCLUDE needs the window to be its own operator")
    functions = tuple(
        dataclasses.replace(f, frame=WindowFrame(*frames[f.alias])) if f.alias in frames else f
        for f in node.functions
    )
    return ds._derive(dataclasses.replace(node, functions=functions))


def _combined(win, pieces: list[tuple], with_self: bool):
    """The aggregate of `win` over `pieces` (plus the row itself when `with_self`)."""
    agg = window_agg(win.this)
    tag = agg[0] if agg is not None else None
    if tag not in {*_FOLDS, "avg"}:
        raise NotImplementedError(
            f"window frame EXCLUDE is supported for sum, count, avg, min, max, bool_and and "
            f"bool_or, not {win.this.sql()}"
        )
    if not pieces and not with_self:
        raise NotImplementedError("window frame EXCLUDE leaves this frame empty")
    value = agg[1]
    if tag == "avg":
        total = _fold("sum", pieces, with_self, value)
        count = _fold("count", pieces, with_self, value)
        return exp.Div(
            this=exp.cast(total, "DOUBLE"),
            expression=exp.Nullif(this=count, expression=exp.Literal.number(0)),
        )
    return _fold(tag, pieces, with_self, value)


def _fold(tag: str, pieces: list[tuple], with_self: bool, value):
    """Fold `tag` windows over each piece, and the row's own contribution, into one value."""
    parts = []
    for partition, order, spec, token in pieces:
        part = exp.Window(
            this=_agg_call(tag, value),
            partition_by=[c.copy() for c in partition],
            order=order.copy() if order is not None else None,
            spec=spec.copy(),
        )
        part.meta[SHARED_FRAME] = token
        parts.append(part)
    if with_self:
        parts.append(_own_contribution(tag, value))
    out = parts[0]
    for nxt in parts[1:]:
        out = _FOLDS[tag](out, nxt)
    return exp.Paren(this=out)


def _agg_call(tag: str, value):
    """``tag(value)`` as a sqlglot aggregate node; ``count(*)`` when `value` is None."""
    if tag == "count":
        return exp.Count(this=exp.Star() if value is None else value.copy())
    node = {
        "sum": exp.Sum,
        "min": exp.Min,
        "max": exp.Max,
        "bool_and": exp.LogicalAnd,
        "bool_or": exp.LogicalOr,
    }[tag]
    return node(this=value.copy())


def _own_contribution(tag: str, value):
    """What the current row adds to `tag` when ``EXCLUDE TIES`` puts it back."""
    if tag != "count":
        return value.copy()
    if value is None:
        return exp.Literal.number(1)
    return exp.Case(
        ifs=[
            exp.If(
                this=exp.Is(this=value.copy(), expression=exp.Null()), true=exp.Literal.number(0)
            )
        ],
        default=exp.Literal.number(1),
    )
