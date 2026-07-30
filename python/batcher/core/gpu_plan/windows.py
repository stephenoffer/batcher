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
    if any(k.get("e") != "col" for k in ir["partition_keys"]):
        return False
    if any(k["expr"].get("e") != "col" for k in ir["order_keys"]):
        return False
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
    return _is_running_frame(frame) and func in _RUNNING


def _is_running_frame(frame: dict) -> bool:
    """Whether an explicit frame is exactly `ROWS UNBOUNDED PRECEDING → CURRENT ROW`."""
    return (
        frame.get("units") == "rows"
        and frame.get("start", {}).get("kind") == "unbounded_preceding"
        and frame.get("end", {}).get("kind") == "current_row"
    )


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
    part = [k["name"] for k in ir["partition_keys"]]
    order = [k["expr"]["name"] for k in ir["order_keys"]]
    ascending = [not k["descending"] for k in ir["order_keys"]]
    nulls_first = bool(ir["order_keys"] and ir["order_keys"][0].get("nulls_first"))

    out = df.copy()
    out[_ROW] = _arange(be, len(out))
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
    return out.drop(columns=[_ROW, _POS, _PID])


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
    running = bool(order) if frame is None else _is_running_frame(frame)
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
