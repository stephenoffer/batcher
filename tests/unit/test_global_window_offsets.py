"""The ordered-bucket-offset algebra, checked against a whole-relation oracle.

`dist/global_window/offsets.py` is what lets a *global* window be split at all: window each
ordered bucket on its own, then shift it by the prior buckets' contribution. Both schedules
that use it -- the single-node stream and the two distributed executors -- are the same
arithmetic, so a defect here is a wrong answer on every one of them at once, and one that
only shows up above the bucket-count threshold.

These tests need no engine. Each one computes the window twice in plain Python: once over the
whole relation (the oracle), and once bucket by bucket followed by `OrderedBucketOffsets`. If
the offsets are right the two agree row for row; if a shift is off by a bucket, counts a row
where it should count a distinct key, or accumulates in the wrong direction, they do not.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from batcher.dist.global_window.offsets import (
    OrderedBucketOffsets,
    bucket_order,
    inject_window_helpers,
    supports_ordered_bucket_offsets,
    unoffsettable_functions,
)
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Scan, Window, WindowFuncSpec
from batcher.plan.schema import SchemaRef

#: One relation, already in ORDER BY order, with deliberate ties on `t` so a peer group can
#: be observed to stay whole, and a null in `v` so the null-carrying offsets are exercised.
_T = [10, 10, 20, 30, 30, 30, 40, 50, 50, 60]
_V = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def _window(funcs: list[tuple[str, str | None, str]], descending: bool = False) -> Window:
    """A global `Window` over `(t, v)` carrying `funcs` as ``(func, input, alias)``."""
    from batcher.plan.logical.aggregate import SortKeySpec

    schema = SchemaRef(pa.schema([("t", pa.int64()), ("v", pa.float64())]))
    return Window(
        input=Scan(source_id=0, schema=schema),
        partition_keys=(),
        order_keys=(SortKeySpec(Col("t"), descending=descending, nulls_first=False),),
        functions=tuple(
            WindowFuncSpec(
                func=f,
                input=None if i is None else Col(i),
                alias=a,
                # `ntile` carries its tile count in `offset`, which every other function
                # leaves at the lag/lead default of 1.
                offset=_NTILE_TILES if f == "ntile" else 1,
            )
            for f, i, a in funcs
        ),
        rank_limit=None,
    )


def _kernel(rows: list[tuple[int, float | None]], func: str) -> list:
    """The window kernel's answer over ONE ordered run of rows -- the oracle.

    Deliberately a direct transcription of the SQL definition (`RANGE UNBOUNDED PRECEDING TO
    CURRENT ROW`, tied rows sharing the end-of-peer-group value) rather than anything the
    engine computes, so feeding it the whole relation and feeding it one bucket are the same
    function of their input and the comparison has something to say.
    """
    n = len(rows)
    out: list = [None] * n
    if func == "row_number":
        return [i + 1 for i in range(n)]
    if func in ("rank", "dense_rank"):
        rank, dense, prev = 0, 0, object()
        for i, (t, _v) in enumerate(rows):
            if t != prev:
                rank, dense, prev = i + 1, dense + 1, t
            out[i] = dense if func == "dense_rank" else rank
        return out
    # The aggregates share the peer-group shape: accumulate through the row, then write the
    # end-of-peer-group value back across the whole group.
    acc: list = []
    start = 0
    for i, (t, v) in enumerate(rows):
        if v is not None:
            acc.append(v)
        if i + 1 == n or rows[i + 1][0] != t:
            if func == "sum":
                val = sum(acc) if acc else None
            elif func == "count":
                val = len(acc)
            elif func == "min":
                val = min(acc) if acc else None
            elif func == "max":
                val = max(acc) if acc else None
            elif func == "avg":
                val = (sum(acc) / len(acc)) if acc else None
            elif func == "first_value":
                val = rows[0][1]
            elif func == "last_value":
                # Whole-partition, not end-of-peer-group: the engine's `last_value` with no
                # explicit frame reads the partition's final row, which is what makes it the
                # one correction that cannot be made until the walk has ended.
                val = rows[-1][1]
            elif func == "var":
                val = _sample_var(acc)
            elif func == "stddev":
                var = _sample_var(acc)
                val = None if var is None else math.sqrt(var)
            elif func == "row_count":
                # Rows (null-valued ones included) through the end of this peer group — the
                # numerator `cume_dist` divides by the relation's total.
                val = i + 1
            elif func == "percent_rank":
                val = 0.0 if n == 1 else (start_rank(rows, start) - 1) / (n - 1)
            elif func == "cume_dist":
                val = (i + 1) / n
            elif func == "ntile":
                val = None  # positional; written per row below
            else:  # pragma: no cover - the test only asks for the offsettable set
                raise AssertionError(func)
            for j in range(start, i + 1):
                out[j] = val
            start = i + 1
    if func == "ntile":
        return [_ntile_of(j + 1, n, _NTILE_TILES) for j in range(n)]
    return out


#: The tile count every `ntile` case in this file uses, so the oracle and the `Window` spec
#: cannot disagree about it.
_NTILE_TILES = 4


def _sample_var(values: list[float]) -> float | None:
    """Sample variance (ddof=1) of `values`, or None below two of them — SQL's `var`."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


def _ntile_of(row: int, total: int, tiles: int) -> int:
    """SQL `NTILE`: `total` rows into `tiles` groups, the first `total % tiles` one larger."""
    base, rem = divmod(total, tiles)
    head = rem * (base + 1)
    if row <= head:
        return -(-row // (base + 1))
    return rem + -(-(row - head) // max(base, 1))


def start_rank(rows, start: int) -> int:
    """The `rank` of the peer group beginning at index `start` (1-based, ties share it)."""
    return start + 1


def _buckets(rows: list[tuple[int, float | None]], cuts: list[int]):
    """Split `rows` into ordered buckets at the given `t` boundaries (equal `t` stays whole)."""
    out: list[list[tuple[int, float | None]]] = [[] for _ in range(len(cuts) + 1)]
    for t, v in rows:
        b = sum(1 for c in cuts if t > c)
        out[b].append((t, v))
    return out


#: The output type each function's column carries. Spelled out rather than inferred, because
#: a bucket whose values are all null would otherwise infer arrow's `null` type and the
#: offsets would be tested against a column shape the kernel never produces.
_OUT_TYPE = {
    "row_number": pa.int64(),
    "rank": pa.int64(),
    "dense_rank": pa.int64(),
    "count": pa.int64(),
    "sum": pa.float64(),
    "min": pa.float64(),
    "max": pa.float64(),
    "avg": pa.float64(),
    "first_value": pa.float64(),
    "last_value": pa.float64(),
    "var": pa.float64(),
    "stddev": pa.float64(),
    "percent_rank": pa.float64(),
    "cume_dist": pa.float64(),
    "ntile": pa.int64(),
    "row_count": pa.int64(),
}

#: Each helper role's oracle function, so the test asks `_kernel` for exactly the column
#: `inject_window_helpers` asked the engine for. Read from the injection rather than restated
#: per function: a helper added there and forgotten here would leave the offset reading a
#: column the test never built, which is a `KeyError` rather than a silent pass.
_HELPER_KERNEL = {
    "sum": "sum",
    "cnt": "count",
    "avg": "avg",
    "rank": "rank",
    "rows": "row_count",
    "rn": "row_number",
}


def _offset_answer(win: Window, rows, cuts, func: str, alias: str, descending: bool = False):
    """Window each bucket alone, run the offsets, and return the rows in global order.

    The helper columns are the ones `inject_window_helpers` actually asks the engine for,
    read off that function rather than restated here — the previous spelling hard-coded
    `avg`'s pair and stopped compiling the moment a second function needed helpers, which is
    the drift this file exists to catch in the engine and had itself.
    """
    helpers = inject_window_helpers(win, {"functions": []})
    buckets = _buckets(rows, cuts)
    offsets = OrderedBucketOffsets(win, helpers)
    corrected: list[pa.Table] = []
    for b in bucket_order(len(buckets), descending):
        br = buckets[b]
        if not br:
            continue
        cols = {
            "t": pa.array([t for t, _ in br], pa.int64()),
            "v": pa.array([v for _, v in br], pa.float64()),
            alias: pa.array(_kernel(br, func), _OUT_TYPE[func]),
        }
        for role, helper in helpers.get(alias, {}).items():
            kf = _HELPER_KERNEL[role]
            cols[helper] = pa.array(_kernel(br, kf), _OUT_TYPE[kf])
        corrected.append(offsets.apply(pa.table(cols)))
    # `finalize` closes out the corrections that need the whole relation. It is a no-op for
    # every function that does not, so the harness calls it unconditionally — which is also
    # what pins that it *is* a no-op for them.
    final = offsets.finalize(pa.concat_tables(corrected))
    assert final.column_names == ["t", "v", alias], "helper columns leaked"
    return final.column(alias).to_pylist()


@pytest.mark.unit
@pytest.mark.parametrize(
    "func,arg",
    [
        ("row_number", None),
        ("rank", None),
        ("dense_rank", None),
        ("sum", "v"),
        ("count", "v"),
        ("min", "v"),
        ("max", "v"),
        ("avg", "v"),
        ("first_value", "v"),
        # The six below are corrected either by a moment combination (`var`, `stddev`) or by
        # `finalize` over the assembled result, and none of them existed in this algebra
        # before. They are exactly the shapes a per-bucket run gets confidently wrong: a
        # `percent_rank` divided by its own bucket's row count is a plausible number.
        ("var", "v"),
        ("stddev", "v"),
        ("last_value", "v"),
        ("percent_rank", None),
        ("cume_dist", None),
        ("ntile", None),
    ],
)
@pytest.mark.parametrize("cuts", [[25], [15, 35], [10, 30, 50], [5], [99]])
def test_bucketed_plus_offsets_equals_the_whole_relation(func, arg, cuts):
    """Every offsettable function: bucket-then-offset == window over the whole relation.

    `cuts` sweeps the bucket layout -- an even split, an uneven one, a cut that lands exactly
    on a tied value (the peer group must stay in one bucket), and two that leave a bucket
    empty at each end. The answer may not depend on any of it.
    """
    rows = list(zip(_T, _V, strict=True))
    win = _window([(func, arg, "r")])
    expected = _kernel(rows, func)
    assert _offset_answer(win, rows, cuts, func, "r") == pytest.approx(
        expected, rel=1e-12, nan_ok=True
    )


@pytest.mark.unit
def test_descending_visits_buckets_in_reverse():
    """A descending ORDER BY reverses the *visit* order, not the arithmetic.

    Range partitioning cuts on the key's value either way, so bucket 0 always holds the lowest
    keys; it is `bucket_order` that must walk them highest-first when the window is descending.
    Walking them the other way accumulates the offsets against the sort and silently produces
    a running sum that counts the wrong prefix.
    """
    rows = sorted(zip(_T, _V, strict=True), key=lambda r: -r[0])
    win = _window([("sum", "v", "r")], descending=True)
    assert _offset_answer(win, rows, [25], "sum", "r", descending=True) == pytest.approx(
        _kernel(rows, "sum"), rel=1e-12, nan_ok=True
    )


@pytest.mark.unit
def test_a_running_sum_survives_a_bucket_that_opens_with_nulls():
    """Regression: `NULL + prior_total` is NULL, and it used to eat the prior total.

    The kernel's within-bucket running sum is NULL until that bucket's first non-null input.
    Offsetting it with a plain `add` left it NULL, so every row from a bucket's start up to
    its first non-null value reported no running sum at all — even though earlier buckets had
    already contributed one. It needed a bucket that both opens with a null *and* is not the
    first, which is why the single-node streamer carried it unnoticed.
    """
    rows = list(zip(_T, _V, strict=True))
    win = _window([("sum", "v", "r")])
    # `_V[2]` is the null and `t == 20`, so this cut opens bucket 1 on it.
    got = _offset_answer(win, rows, [15], "sum", "r")
    assert got[2] is not None, "the prior bucket's running total was dropped"
    assert got == pytest.approx(_kernel(rows, "sum"), rel=1e-12, nan_ok=True)


@pytest.mark.unit
def test_dense_rank_shifts_by_distinct_keys_not_rows():
    """`dense_rank`'s offset is the prior buckets' *distinct* key count.

    Shifting it by the row count instead is the single most tempting mistake here, because
    every other rank offset is a row count and the two agree exactly when there are no ties.
    `_T` has ties on both sides of the cut, so they disagree.
    """
    rows = list(zip(_T, _V, strict=True))
    win = _window([("dense_rank", None, "r")])
    got = _offset_answer(win, rows, [25], "dense_rank", "r")
    assert got == _kernel(rows, "dense_rank")
    assert got[-1] == len(set(_T)), "the last row's dense rank is the distinct key count"
    assert got[-1] != len(_T), "a row-count shift would have produced this"


@pytest.mark.unit
def test_the_unoffsettable_functions_are_refused():
    """The predicate that gates every caller: anything not offsettable must be declined.

    Each of these reads something a bucket does not hold -- a following row, a whole-partition
    total, an explicit frame -- so computing it per bucket would return a confidently wrong
    number rather than an error. This is the only thing standing between that and a user.
    """
    assert supports_ordered_bucket_offsets(_window([("row_number", None, "r")]))
    # Genuinely unoffsettable, on either kind of driver: each reads a row its own bucket does
    # not hold, in an order the kernel does not return the bucket in.
    for func, arg in [("lag", "v"), ("lead", "v"), ("median", "v"), ("count_distinct", "v")]:
        win = _window([(func, arg, "r")])
        assert not supports_ordered_bucket_offsets(win), func
        assert not supports_ordered_bucket_offsets(win, assembled=True), func
        assert unoffsettable_functions(win, assembled=True) == [func], func
    # These four *are* offsettable, but only for a driver that holds every bucket before it
    # returns any row: three divide by the relation's total row count and one is its last
    # value. Declining them by default is what keeps the streaming driver — which yields
    # buckets as it goes — from emitting a `percent_rank` divided by one bucket's rows.
    for func, arg in [
        ("last_value", "v"),
        ("ntile", None),
        ("percent_rank", None),
        ("cume_dist", None),
    ]:
        win = _window([(func, arg, "r")])
        assert not supports_ordered_bucket_offsets(win), func
        assert supports_ordered_bucket_offsets(win, assembled=True), func
        assert unoffsettable_functions(win, assembled=True) == [], func
    # A function it *does* cover is never named as the culprit. The message this feeds used
    # to list every function in the window, so a `lag` beside a `row_number` reported both.
    mixed = _window([("row_number", None, "a"), ("lag", "v", "b")])
    assert unoffsettable_functions(mixed, assembled=True) == ["lag"]
    # A PARTITION BY has its own (hash) shuffle and must not be routed here.
    win = _window([("row_number", None, "r")])
    assert not supports_ordered_bucket_offsets(
        Window(
            input=win.input,
            partition_keys=(Col("t"),),
            order_keys=win.order_keys,
            functions=win.functions,
            rank_limit=None,
        )
    )
    # Two order keys ARE accepted: only the LEADING one has to be a plain column, because it
    # is the only one the range partitioner reads values from. This asserted the opposite,
    # on the ground that "there is no single column to range-partition on" — but there is,
    # and all three drivers already cut on `order_keys[0]` alone.
    #
    # The bucket argument survives the extra keys: a peer group under a multi-key `ORDER BY`
    # is a set of rows equal on *every* key, so it sits inside the set of rows equal on the
    # leading key, which the partitioner puts in one bucket. No peer group straddles a cut,
    # and an earlier bucket's rows have a strictly smaller leading key so the buckets stay
    # ordered. Refusing was not conservative: a global window is not a `_split_at`
    # pass-through, so nothing carried it up and `ORDER BY a, b` **raised** `PlanError` on
    # distributed data instead of falling back to the materializing kernel.
    # `tests/differential/test_diff_global_window_split_folds.py` holds the split result equal
    # to the single-node one for ten functions, `rank` and `dense_rank` among them — the two
    # that would expose a broken peer group.
    from batcher.plan.logical.aggregate import SortKeySpec

    assert supports_ordered_bucket_offsets(
        Window(
            input=win.input,
            partition_keys=(),
            order_keys=(*win.order_keys, SortKeySpec(Col("v"))),
            functions=win.functions,
            rank_limit=None,
        )
    )
    # The relaxation is to the TRAILING keys only. A computed *leading* key has no column for
    # the partitioner to read, so it is still refused — the control that keeps the assertion
    # above from being satisfied by a uniformly permissive predicate.
    assert not supports_ordered_bucket_offsets(
        Window(
            input=win.input,
            partition_keys=(),
            order_keys=(SortKeySpec(Col("t") + Col("v")), *win.order_keys),
            functions=win.functions,
            rank_limit=None,
        )
    )
