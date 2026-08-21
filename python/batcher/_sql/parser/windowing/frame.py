"""Resolving a SQL window spec into the engine's frame, partition and order triple.

Layer: `_sql` (surface). This is the pure-translation half of window handling — it
reads a sqlglot `Window` node and answers what frame the function runs under, which
columns partition it, and how it is ordered. It computes nothing and touches no
`Dataset`; `translate.py` applies what it decides.
"""

from __future__ import annotations

from sqlglot import expressions as exp

#: sqlglot's node name for a window aggregate → the engine's `WindowFn` tag. One table, read
#: by both `_is_agg_window` and `_window_func`, which each carried their own five-name copy.
#: sqlglot names a typed aggregate node `bitwiseoragg`, not `bit_or`, and only the *aggregate*
#: front-end translated that — so SQL rejected six window functions the engine computes.
#: Population variants are excluded deliberately, not by omission: the engine's window layer
#: has only the sample forms, so mapping `stddevpop` here would answer a different question.
#: `tests/differential/test_diff_window_agg_vocabulary.py` pins the whole story.
_WINDOW_AGGS: dict[str, str] = {
    "sum": "sum",
    "avg": "avg",
    "min": "min",
    "max": "max",
    "count": "count",
    "bitwiseandagg": "bit_and",
    "bitwiseoragg": "bit_or",
    "bitwisexoragg": "bit_xor",
    "logicaland": "bool_and",
    "logicalor": "bool_or",
    "stddev": "stddev",
    "stddevsamp": "stddev",
    "variance": "var",
    "median": "median",
}


def _is_agg_window(win) -> bool:
    """Whether the window function is an aggregate (frames apply to these only)."""
    return type(win.this).__name__.lower() in _WINDOW_AGGS


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


#: SQL frame kinds → the engine's frame units. `RANGE` counts in the ORDER BY key's own
#: values, `GROUPS` in peer groups, `ROWS` in physical rows — three different relations,
#: all three now executable, so the translator carries the kind through rather than
#: rejecting two of them.
_FRAME_KINDS = {"ROWS": "rows", "RANGE": "range", "GROUPS": "groups"}


def _window_frame(win) -> tuple[int | None, int | None, str] | None:
    """Translate an explicit ``<kind> BETWEEN …`` frame to a ``(start, end, units)`` triple.

    Returns ``None`` when there is no explicit frame (the default cumulative/whole
    partition behaviour is unchanged). A ``RANGE`` bound may be an interval
    (``RANGE BETWEEN INTERVAL '5' MINUTE PRECEDING``), which becomes microseconds — the
    unit the engine normalizes every temporal order key to.
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
    units = _FRAME_KINDS.get(kind)
    if units is None:
        if spec.args.get("start") is not None or spec.args.get("end") is not None:
            raise NotImplementedError(f"unsupported window frame kind {kind!r}")
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
    return (_frame_bound(start, spec.args.get("start_side")), end_bound, units)


def _frame_bound(value, side) -> int | None:
    """A frame bound → signed offset: UNBOUNDED→None, CURRENT ROW→0, n P/F→∓n.

    The offset is counted in the frame's own units, so an ``INTERVAL`` bound resolves to
    microseconds and a plain number stays a plain number.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.upper()
        if text == "UNBOUNDED":
            return None
        if text == "CURRENT ROW":
            return 0
        raise NotImplementedError(f"unsupported window frame bound {value!r}")
    n = _bound_magnitude(value)
    sided = (side or "").upper()
    if sided == "PRECEDING":
        return -n
    if sided == "FOLLOWING":
        return n
    raise NotImplementedError(f"window frame bound needs PRECEDING/FOLLOWING, got {side!r}")


def _bound_magnitude(value) -> int:
    """The non-negative size of a frame bound: a row/group count, or an interval's
    microseconds.

    ``RANGE BETWEEN INTERVAL '5' MINUTE PRECEDING`` is how SQL spells a five-minute
    trailing window, and it is the shape a time-series query actually reaches for. The
    conversion reuses `plan.functions.temporal`'s duration parser, so the SQL spelling and
    ``bt.window('5m')`` cannot disagree about how long a minute is.
    """
    from batcher.plan.functions.temporal import _duration_micros

    if isinstance(value, exp.Interval):
        unit = value.args.get("unit")
        unit_name = (unit.name if hasattr(unit, "name") else str(unit or "")).strip().lower()
        count = value.this.name if hasattr(value.this, "name") else str(value.this)
        if not unit_name:
            raise NotImplementedError("an INTERVAL window frame bound needs a unit")
        return _duration_micros(f"{count} {unit_name}", arg="window frame bound")
    return int(value.this)


def _window_partition(win) -> tuple:
    cols = win.args.get("partition_by") or []
    keys = []
    for c in cols:
        if not isinstance(c, exp.Column):
            # `hoist_window_args` has already replaced every computed key with a hidden
            # column, so anything left here is a shape it could not reach.
            raise NotImplementedError(f"unsupported window PARTITION BY key: {c.sql()}")
        keys.append(c.name)
    return tuple(keys)


def _window_order(win) -> tuple:
    order = win.args.get("order")
    if order is None:
        return ()
    specs = []
    for o in order.expressions:
        target = o.this
        if not isinstance(target, exp.Column):
            # As above: a computed key is hoisted before this runs.
            raise NotImplementedError(f"unsupported window ORDER BY key: {target.sql()}")
        # `NULLS FIRST` rides as a third element. Reading only `desc` dropped the clause, so
        # every `NULLS FIRST` window returned the `NULLS LAST` ranks — and, under a running
        # frame, different sums, since null placement decides what the frame contains.
        specs.append((target.name, bool(o.args.get("desc")), bool(o.args.get("nulls_first"))))
    return tuple(specs)


def _const_int(node, ctx: str) -> int:
    """Evaluate a constant integer argument, handling negatives.

    sqlglot parses a negative literal (``-1``) as a ``Neg`` wrapping a ``Literal``, so a
    naive ``int(node.this)`` reads the inner node and raises ``TypeError``. ``to_py()``
    folds ``Neg``/``Literal`` to a Python value; a non-constant argument is rejected.
    """
    if isinstance(node, (exp.Literal, exp.Neg)):
        try:
            return int(node.to_py())
        except (TypeError, ValueError):
            pass  # a non-constant frame bound -> fall through and reject it below
    raise NotImplementedError(f"window function {ctx!r} requires a constant integer argument")
