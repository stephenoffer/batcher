"""Window functions on a dataframe backend — ranking, value, and partition/running aggregates.

The engine's `Window` operator keeps every input row and **preserves the input row order**,
adding one column per function. Both halves of that matter. A translation that returns the
rows sorted by the window's own ordering computes the right values and still hands back a
different table than the CPU engine, which an order-independent test cannot see; so the frame
here is sorted to evaluate, then restored to the order it arrived in.

Partitioning runs off a dense integer *partition id* derived from the sorted frame rather than
off the key columns themselves. That is what makes a null partition key work: a group-by on
the raw keys must be told not to drop nulls, and any later join back onto them would drop the
null group silently, because null does not equal null.

Three shapes of function live here, distinguished by what they read:

* **ranking** (`row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile`) reads
  only the ordering, through a tie-group flag computed across all order keys at once;
* **value** (`lag`, `lead`, `first_value`, `last_value`, `nth_value`, and the fills) reads one
  row of the partition, selected by offset or by nullness;
* **aggregate** reads the frame. With no `ORDER BY` the frame is the whole partition; with an
  `ORDER BY` and no explicit frame it is *running* (unbounded preceding through current row),
  which is SQL's default and emphatically not the same answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.core.gpu_plan.backend import Unsupported

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend

__all__ = ["supported_window", "window"]

_RANKING = frozenset({"row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile"})
_VALUE = frozenset({"lag", "lead", "first_value", "last_value", "nth_value"})
_FILL = frozenset({"forward_fill", "backward_fill"})
# Whole-partition reductions, as the backend `GroupBy` method that computes them.
_PARTITION_AGG = {
    "sum": "sum", "avg": "mean", "min": "min", "max": "max", "count": "count",
    "product": "prod", "var": "var", "stddev": "std", "bool_and": "all", "bool_or": "any",
    "count_distinct": "nunique",
}  # fmt: skip
# Reductions with an O(1)-per-row running form (used when an ORDER BY makes the frame
# running). `avg` is derived from the running sum and count rather than listed here.
_RUNNING = frozenset({"sum", "min", "max", "count", "avg"})
# Reductions over a fixed-width moving window. `sum`/`count`/`avg` are computed as a difference
# of running totals, which uses only operations both backends have and needs no windowing
# primitive; `min`/`max` have no such closed form and go through the backend's own `rolling`.
_ROLLING = frozenset({"sum", "min", "max", "count", "avg"})

_POS = "__bt_wpos"
_PID = "__bt_wpid"
_ROW = "__bt_wrow"


def supported_window(ir: dict) -> bool:
    """Whether one `window` RelOp node is translatable to the dataframe backends.

    Args:
        ir: The `window` node's JSON IR.

    Returns:
        True when every partition key, order key and function in the node is translatable.
    """
    # A computed partition or order key is NOT rejected here. It is evaluated into a private
    # column at execution, exactly as a computed group key is (`aggs._key_columns`), because
    # `PARTITION BY date_trunc('month', ts)` is an ordinary way to write a monthly ranking and
    # rejecting it dropped the whole chain to the CPU engine over the shape of one key.
    if ir.get("rank_limit") is not None:
        return False  # a per-partition top-N pushdown, with its own row-elimination semantics
    if len({bool(k.get("nulls_first")) for k in ir["order_keys"]}) > 1:
        return False  # both backends place nulls per sort, not per key
    if not ir["functions"]:
        return False
    ordered = bool(ir["order_keys"])
    return all(_supported_function(f, ordered=ordered) for f in ir["functions"])


def _supported_function(f: dict, *, ordered: bool) -> bool:
    func = f["func"]
    if func in _RANKING:
        return ordered or func == "ntile"
    if func in _VALUE or func in _FILL:
        return ordered and f.get("input", {}).get("e") == "col"
    if func not in _PARTITION_AGG or f.get("input", {}).get("e") != "col":
        return False
    frame = f.get("frame")
    if frame is None:
        # Unframed: whole partition when unordered, running when ordered.
        return not ordered or func in _RUNNING
    if _is_running_frame(frame):
        return func in _RUNNING
    return _rolling_width(frame) is not None and func in _ROLLING


def _is_running_frame(frame: dict) -> bool:
    """Whether an explicit frame is exactly `ROWS UNBOUNDED PRECEDING → CURRENT ROW`."""
    return (
        frame.get("units") == "rows"
        and frame.get("start", {}).get("kind") == "unbounded_preceding"
        and frame.get("end", {}).get("kind") == "current_row"
    )


def _rolling_width(frame: dict) -> int | None:
    """The row count of a `ROWS n PRECEDING → CURRENT ROW` frame, else `None`.

    This is the moving-window shape — a rolling sum, a moving average — which is most of what
    a time series is ever asked for, and which the translator used to decline outright. Frames
    that look *forward* (`FOLLOWING`) are not covered: they are the same idea reflected, but
    each needs its own verification against the engine and an unverified window function is a
    wrong number rather than a slow one.
    """
    if frame.get("units") != "rows":
        return None
    start, end = frame.get("start", {}), frame.get("end", {})
    if end.get("kind") != "current_row":
        return None
    # A one-row window lowers as `CURRENT ROW -> CURRENT ROW` rather than as `0 PRECEDING`,
    # so matching only the `preceding` spelling declined the narrowest window there is.
    if start.get("kind") == "current_row":
        return 1
    if start.get("kind") != "preceding":
        return None
    n = start.get("n")
    return int(n) + 1 if isinstance(n, int) and n >= 0 else None


def window(df, ir: dict, be: DfBackend):
    """Apply one `window` RelOp to `df`, returning every input row in its original order.

    Args:
        df: The dataframe to add window columns to.
        ir: The `window` node's JSON IR.
        be: The dataframe backend to compute on.

    Returns:
        `df` with one added column per window function, in the input's row order.

    Raises:
        Unsupported: For a function or frame outside the translated subset.
    """
    ascending = [not k["descending"] for k in ir["order_keys"]]
    nulls_first = bool(ir["order_keys"] and ir["order_keys"][0].get("nulls_first"))

    out = df.copy()
    out[_ROW] = _arange(be, len(out))
    computed: list[str] = []
    part = _key_names(out, ir["partition_keys"], be, computed, kind="p")
    order = _key_names(out, [k["expr"] for k in ir["order_keys"]], be, computed, kind="o")
    if part or order:
        out = out.sort_values(
            part + order,
            ascending=[True] * len(part) + ascending,
            na_position="first" if nulls_first else "last",
            kind="stable",
        ).reset_index(drop=True)
    grp = out.groupby(_PID, sort=False) if _add_partition_id(out, part) else None
    if grp is None:
        raise Unsupported("window partitioning")
    out[_POS] = grp.cumcount()
    size = _partition_sizes(out)

    for f in ir["functions"]:
        out[f["alias"]] = _evaluate(out, f, be, order=order, size=size)

    # Restore the arrival order, then drop the private columns: the engine's Window keeps the
    # input's row order, and a sorted result would differ from it row-for-row.
    out = out.sort_values(_ROW, kind="stable").reset_index(drop=True)
    return out.drop(columns=[_ROW, _POS, _PID, *computed])


def _key_names(out, keys: list[dict], be: DfBackend, computed: list[str], *, kind: str):
    """The column names to partition or order by, materializing the ones that are expressions.

    A key that is already a plain column is used in place; anything else is evaluated into a
    private column, whose name is recorded in `computed` so the caller drops it again. The
    window operator's contract is that it *adds* one column per function to the input's
    columns, so a materialized key that survived would be an extra output column.
    """
    from batcher.core.gpu_plan.exprs import eval_expr

    names: list[str] = []
    for i, key in enumerate(keys):
        if key.get("e") == "col":
            names.append(key["name"])
            continue
        name = f"__bt_wk{kind}{i}"
        out[name] = be.column(eval_expr(key, out, be), out)
        computed.append(name)
        names.append(name)
    return names


def _arange(be: DfBackend, n: int):
    """`0..n-1` as an Int64 column, built in the library's own layer."""
    import numpy as np

    return be.series(np.arange(n, dtype="int64"))


def _add_partition_id(out, part: list[str]) -> bool:
    """Attach a dense Int64 partition id to the (already sorted) frame.

    An integer id is used in place of the raw key columns so that a **null partition key**
    behaves: it groups and joins like any other value, where a null key would be dropped by
    both a default `groupby` and every subsequent merge.
    """
    if not part:
        out[_PID] = 0
        return True
    changed = None
    for name in part:
        cur = out[name]
        prev = cur.shift(1)
        same = (cur == prev).fillna(False) | (cur.isna() & prev.isna())
        changed = ~same if changed is None else (changed | ~same)
    out[_PID] = changed.astype("int64").cumsum()
    return True


def _partition_sizes(out):
    """Each row's partition size, aligned to the sorted frame."""
    counts = out.groupby(_PID, sort=False).size()
    return out[_PID].map(counts) if hasattr(out[_PID], "map") else counts[out[_PID]]


def _tie_flag(out, order: list[str]):
    """`True` where a row starts a new tie group — a new partition, or a changed order key.

    The partition's first row must always start a group. Deriving that from `shift(1)` alone
    gets it wrong when the first row's order value is null, because the shifted value is null
    too and "both null" reads as a tie.
    """
    first = out[_POS] == 0
    if not order:
        return first
    same = None
    grp = out.groupby(_PID, sort=False)
    for name in order:
        cur = out[name]
        prev = grp[name].shift(1)
        eq = (cur == prev).fillna(False) | (cur.isna() & prev.isna())
        same = eq if same is None else (same & eq)
    return first | ~same


def _evaluate(out, f: dict, be: DfBackend, *, order, size):
    func = f["func"]
    if func in _RANKING:
        return _ranking(out, f, order=order, size=size)
    if func in _FILL:
        grp = out.groupby(_PID, sort=False)[f["input"]["name"]]
        return grp.ffill() if func == "forward_fill" else grp.bfill()
    if func in _VALUE:
        return _value(out, f, size=size)
    frame = f.get("frame")
    if frame is not None and not _is_running_frame(frame):
        width = _rolling_width(frame)
        if width is None:
            raise Unsupported(f"window frame {frame}")
        return _rolling(out, f, be, width)
    running = bool(order) if frame is None else True
    return _running(out, f, be) if running else _partition_agg(out, f, be)


def _ranking(out, f: dict, *, order, size):
    func = f["func"]
    pos = out[_POS]
    if func == "row_number":
        return pos + int(f.get("offset", 1))
    if func == "ntile":
        n = int(f.get("offset", 1))
        if n < 1:
            raise Unsupported("ntile with a non-positive bucket count")
        # SQL's distribution: the remainder rows go to the earliest buckets, which is exactly
        # what the floor form gives.
        return (pos * n // size) + 1
    tie = _tie_flag(out, order)
    if func == "dense_rank":
        return tie.astype("int64").groupby(out[_PID], sort=False).cumsum()
    # `rank`, and the two derived from it, share the min-rank of each tie group: the position
    # of the group's first row, carried forward across the ties.
    start = (pos + 1).where(tie, None)
    rank = start.groupby(out[_PID], sort=False).ffill().astype("int64")
    if func == "rank":
        return rank
    if func == "percent_rank":
        # A single-row partition has no spread, and SQL defines its percent_rank as 0.
        denom = size - 1
        return ((rank - 1) / denom).where(denom > 0, 0.0)
    # cume_dist: the fraction of the partition at or before the current tie group, so it
    # reads the group's LAST position rather than its first.
    end = (pos + 1).where(tie.shift(-1).fillna(True), None)
    last = end.groupby(out[_PID], sort=False).bfill()
    return last / size


def _value(out, f: dict, *, size):
    func = f["func"]
    name = f["input"]["name"]
    grp = out.groupby(_PID, sort=False)[name]
    offset = int(f.get("offset", 1))
    if func == "lag":
        return grp.shift(offset)
    if func == "lead":
        return grp.shift(-offset)
    # first/last/nth select one row of the partition and broadcast it. Masking every other
    # row to null and filling is what keeps a *genuinely null* selected value null: the mask
    # leaves the whole partition null, and a fill of nothing is still nothing.
    target = {"first_value": 0, "last_value": None, "nth_value": offset - 1}[func]
    picked = out[_POS] == (size - 1 if target is None else target)
    masked = out[name].where(picked, None)
    filled = masked.groupby(out[_PID], sort=False)
    return filled.bfill() if target is None else filled.ffill()


def _agg_input(out, f: dict, be: DfBackend) -> str:
    """The column a windowed reduction reads, declining a `NaN`-bearing one.

    Same reason as the grouped case in `aggs`: the engine orders `NaN` above every number,
    both backends treat it as missing, and the difference shows up as a plausible-looking
    number rather than an error.
    """
    name = f["input"]["name"]
    if be.has_nan(out[name]):
        raise Unsupported(f"windowed aggregate over NaN-bearing column {name!r}")
    return name


def _partition_agg(out, f: dict, be: DfBackend):
    """A whole-partition reduction, broadcast to every row of the partition."""
    name = _agg_input(out, f, be)
    method = _PARTITION_AGG[f["func"]]
    kwargs = {"min_count": 1} if f["func"] in ("sum", "product") else {}
    grouped = out.groupby(_PID, sort=False)[name]
    reducer = getattr(grouped, method, None)
    if reducer is None:
        raise Unsupported(f"windowed {f['func']}")
    per_partition = reducer(**kwargs)
    return out[_PID].map(per_partition)


def _running(out, f: dict, be: DfBackend):
    """A running (unbounded preceding → current row) reduction over the sorted partition.

    Every form is expressed so that nulls are *skipped*, and so that a prefix containing no
    non-null value yields null rather than the operator's identity — `sum` over nothing is
    null, not `0`.
    """
    name = _agg_input(out, f, be)
    func = f["func"]
    values = out[name]
    grp = values.groupby(out[_PID], sort=False)
    seen = values.notna().astype("int64").groupby(out[_PID], sort=False).cumsum()
    if func == "count":
        return seen
    if func in ("min", "max"):
        running = grp.cummin() if func == "min" else grp.cummax()
        return running.where(seen > 0, None)
    total = values.fillna(0).groupby(out[_PID], sort=False).cumsum().where(seen > 0, None)
    if func == "sum":
        return total
    if func == "avg":
        return total / seen
    raise Unsupported(f"running {func}")


def _rolling(out, f: dict, be: DfBackend, width: int):
    """A reduction over the `width` rows ending at the current one.

    `sum`, `count` and `avg` are differences of running totals: the total through this row minus
    the total through the row that has just left the window. That is exact, uses only operations
    both backends have, and keeps the null handling identical to the running case — a window
    with no non-null value yields null rather than the operator's identity.

    `min` and `max` have no such closed form (a value leaving the window can be the one that was
    the extreme), so they go through the backend's own `rolling`, and decline if it is absent.
    """
    name = _agg_input(out, f, be)
    func = f["func"]
    values = out[name]
    pid = out[_PID]
    total = values.fillna(0).groupby(pid, sort=False).cumsum()
    seen = values.notna().astype("int64").groupby(pid, sort=False).cumsum()
    # What the window has already passed: zero at a partition's start, where the shift is null.
    gone_total = total.groupby(pid, sort=False).shift(width).fillna(0)
    gone_seen = seen.groupby(pid, sort=False).shift(width).fillna(0)
    count = seen - gone_seen
    if func == "count":
        return count
    if func in ("sum", "avg"):
        window_sum = total - gone_total
        return (window_sum if func == "sum" else window_sum / count).where(count > 0, None)
    grouped = values.groupby(pid, sort=False)
    roller = getattr(grouped, "rolling", None)
    if roller is None:
        raise Unsupported(f"rolling {func}")
    try:
        rolled = getattr(roller(width, min_periods=1), "min" if func == "min" else "max")()
    except (TypeError, NotImplementedError, AttributeError) as exc:
        raise Unsupported(f"rolling {func}: {exc}") from exc
    # `groupby(...).rolling(...)` prefixes the group key onto the index; drop it and realign.
    if getattr(rolled.index, "nlevels", 1) > 1:
        rolled = rolled.reset_index(level=0, drop=True)
    return rolled.sort_index()
