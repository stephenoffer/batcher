"""The three shapes with no key to partition on must run out of core and still match DuckDB.

A grace spill bounds memory by hashing a key into buckets. Three operators have no such key,
and all three used to refuse an input over the memory envelope with
`MemoryBudgetExceededError`:

* a window with no `PARTITION BY` -- the relation is one partition;
* a range join whose *right* side exceeds the envelope -- only the left side was chunked;
* an ASOF join with no `by` keys -- every left row may match any right row.

Each now has a bounded-memory form in the engine (`bc-interp`: `ops/window_stream.rs`,
`join_par/range_blocked.rs`, `join_par/asof_stream.rs`). These tests run each shape under an
envelope far below its input and compare against DuckDB. Every one also carries a control
showing the envelope actually bit: a result from an in-memory run would pass the comparison
equally well and prove nothing about the new path.

The plans are written as IR and run through the engine directly, in a fresh process per run
(`tests/_engine_run.py` says why).

Window and join outputs are unordered relations, so rows are compared as a multiset. Every
function whose value depends on a tie-break (`row_number`, `lag`, `lead`, `first_value`) is
ordered by a unique key as well, because SQL leaves the order of peers open and DuckDB and
Batcher may resolve it differently.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from _engine_run import run_plan as _run
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

pytest.importorskip("batcher._native", reason="native engine not built")

#: Far below every input here (each is several hundred KB), so each operator has to take its
#: bounded path. The engine's data-plane budget is this times `memory.hard_limit`.
_TIGHT = 100_000
#: Large enough that nothing spills: the in-memory reference run.
_WIDE = 1 << 30


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


# --------------------------------------------------------------------------------------------
# A window with no PARTITION BY
# --------------------------------------------------------------------------------------------

_WINDOW_ROWS = 40_000


def _window_table(n: int) -> pa.Table:
    """A tie-heavy float order key with nulls, -0.0/0.0 and NaN; a nullable value; a unique id."""

    def t(i: int) -> float | None:
        k = i % 23
        if k == 0:
            return None
        if k == 1:
            return -0.0
        if k == 2:
            return 0.0
        if k == 3:
            return math.nan
        return float((i * 7) % 211) / 4.0

    return pa.table(
        {
            "id": pa.array(range(n), pa.int64()),
            "t": pa.array([t(i) for i in range(n)], pa.float64()),
            "v": pa.array([None if i % 5 == 0 else i % 97 for i in range(n)], pa.int64()),
        }
    )


def _window_plan(order_keys: list[dict], functions: list[dict]) -> dict:
    return {
        "op": "window",
        "input": {"op": "scan", "source_id": 0},
        "partition_keys": [],
        "order_keys": order_keys,
        "functions": functions,
        "rank_limit": None,
    }


def _key(name: str, descending: bool, nulls_first: bool) -> dict:
    return {"expr": _col(name), "descending": descending, "nulls_first": nulls_first}


def _sql_order(cols: list[tuple[str, bool, bool]]) -> str:
    return ", ".join(
        f"{c} {'DESC' if d else 'ASC'} NULLS {'FIRST' if nf else 'LAST'}" for c, d, nf in cols
    )


_ORDERINGS = [(False, False), (True, False), (False, True), (True, True)]


@pytest.mark.parametrize(("descending", "nulls_first"), _ORDERINGS)
def test_a_global_ranking_window_streams_under_a_tight_envelope(
    duck, tmp_path, descending, nulls_first
):
    """`rank`, `dense_rank` and a running `count` over a tied key: tie-independent values."""
    table = _window_table(_WINDOW_ROWS)
    funcs = [
        {"func": "rank", "alias": "rk", "offset": 1},
        {"func": "dense_rank", "alias": "dr", "offset": 1},
        {"func": "count", "input": _col("v"), "alias": "cnt", "offset": 1},
    ]
    plan = _window_plan([_key("t", descending, nulls_first)], funcs)
    report = _run(tmp_path, plan, [table], _TIGHT)
    assert report["spilled"]["window"], "the envelope must force the window out of core"
    order = _sql_order([("t", descending, nulls_first)])
    duck.register("w", table)
    expected = duck.sql(
        f"SELECT id, t, v, rank() OVER (ORDER BY {order}) AS rk, "
        f"dense_rank() OVER (ORDER BY {order}) AS dr, "
        f"count(v) OVER (ORDER BY {order}) AS cnt FROM w"
    )
    assert_same(report["table"], expected)


@pytest.mark.parametrize(("descending", "nulls_first"), _ORDERINGS)
def test_global_positional_windows_stream_under_a_tight_envelope(
    duck, tmp_path, descending, nulls_first
):
    """`row_number`, `first_value`, `lag` and `lead`, ordered to a total order by `id`."""
    table = _window_table(_WINDOW_ROWS)
    funcs = [
        {"func": "row_number", "alias": "rn", "offset": 1},
        {"func": "first_value", "input": _col("v"), "alias": "fv", "offset": 1},
        {"func": "lag", "input": _col("v"), "alias": "lg", "offset": 3},
        {"func": "lead", "input": _col("t"), "alias": "ld", "offset": 2},
    ]
    keys = [_key("t", descending, nulls_first), _key("id", False, False)]
    report = _run(tmp_path, _window_plan(keys, funcs), [table], _TIGHT)
    assert report["spilled"]["window"], "the envelope must force the window out of core"
    order = _sql_order([("t", descending, nulls_first), ("id", False, False)])
    duck.register("w", table)
    expected = duck.sql(
        f"SELECT id, t, v, row_number() OVER (ORDER BY {order}) AS rn, "
        f"first_value(v) OVER (ORDER BY {order}) AS fv, "
        f"lag(v, 3) OVER (ORDER BY {order}) AS lg, "
        f"lead(t, 2) OVER (ORDER BY {order}) AS ld FROM w"
    )
    assert_same(report["table"], expected)


@pytest.mark.parametrize("rows", [0, 1])
def test_a_global_window_over_empty_and_one_row_input(duck, tmp_path, rows):
    """The degenerate sizes: nothing to stream, but the answer must still be right."""
    table = _window_table(rows)
    funcs = [
        {"func": "rank", "alias": "rk", "offset": 1},
        {"func": "lag", "input": _col("v"), "alias": "lg", "offset": 1},
    ]
    report = _run(tmp_path, _window_plan([_key("t", False, False)], funcs), [table], _TIGHT)
    duck.register("w", table)
    expected = duck.sql(
        "SELECT id, t, v, rank() OVER (ORDER BY t) AS rk, lag(v, 1) OVER (ORDER BY t) AS lg FROM w"
    ).fetch_arrow_table()
    got = report["table"]
    assert (0 if got is None else got.num_rows) == expected.num_rows == rows
    if rows:
        assert_tables_equal(got, expected)


def test_control_a_window_the_stream_declines_still_refuses(tmp_path):
    """The same envelope and input with a running `sum`, which has no exact streamed form.

    It must still raise. That proves this envelope is one the window genuinely does not fit
    in, so the successes above came from the streamed path rather than from an in-memory run
    that happened to fit.
    """
    plan = _window_plan(
        [_key("t", False, False)],
        [{"func": "sum", "input": _col("v"), "alias": "s", "offset": 1}],
    )
    report = _run(tmp_path, plan, [_window_table(_WINDOW_ROWS)], _TIGHT)
    assert "raised" in report, report
    assert "cannot spill" in report["raised"]


# --------------------------------------------------------------------------------------------
# A range join whose right side exceeds the envelope
# --------------------------------------------------------------------------------------------


def _points(n: int) -> pa.Table:
    def x(i: int) -> float | None:
        k = i % 31
        if k == 0:
            return None
        if k == 1:
            return -0.0
        return float((i * 13) % 9_000)

    return pa.table(
        {"pid": pa.array(range(n), pa.int64()), "x": pa.array([x(i) for i in range(n)])}
    )


def _bands(n: int) -> pa.Table:
    """Overlapping intervals (duplicated bounds included), a null bound, and a zero bound."""

    def lo(i: int) -> float | None:
        if i % 97 == 0:
            return None
        if i % 89 == 0:
            return 0.0
        return float((i * 7) % 9_000)

    los = [lo(i) for i in range(n)]
    return pa.table(
        {
            "bid": pa.array(range(n), pa.int64()),
            "lo": pa.array(los),
            "hi": pa.array([None if v is None else v + 3.0 for v in los]),
        }
    )


_JOIN_SQL = {
    "inner": "SELECT pid, x, bid, lo FROM p JOIN b ON p.x >= b.lo AND p.x < b.hi",
    "left": "SELECT pid, x, bid, lo FROM p LEFT JOIN b ON p.x >= b.lo AND p.x < b.hi",
    "right": "SELECT pid, x, bid, lo FROM p RIGHT JOIN b ON p.x >= b.lo AND p.x < b.hi",
    "full": "SELECT pid, x, bid, lo FROM p FULL JOIN b ON p.x >= b.lo AND p.x < b.hi",
    "semi": "SELECT pid, x FROM p WHERE EXISTS (SELECT 1 FROM b WHERE p.x >= b.lo AND p.x < b.hi)",
    "anti": "SELECT pid, x FROM p WHERE NOT EXISTS "
    "(SELECT 1 FROM b WHERE p.x >= b.lo AND p.x < b.hi)",
}


def _range_plan(join_type: str) -> dict:
    out = [
        {"side": "left", "name": "pid", "alias": "pid"},
        {"side": "left", "name": "x", "alias": "x"},
    ]
    if join_type not in ("semi", "anti"):
        out += [
            {"side": "right", "name": "bid", "alias": "bid"},
            {"side": "right", "name": "lo", "alias": "lo"},
        ]
    return {
        "op": "range_join",
        "left": {"op": "scan", "source_id": 0},
        "right": {"op": "scan", "source_id": 1},
        "conditions": [
            {"left_key": "x", "right_key": "lo", "op": "ge"},
            {"left_key": "x", "right_key": "hi", "op": "lt"},
        ],
        "join_type": join_type,
        "output": out,
    }


@pytest.mark.parametrize("join_type", list(_JOIN_SQL))
def test_a_range_join_with_an_oversized_right_side_matches_duckdb(duck, tmp_path, join_type):
    """Every flavor, with the right side several times the envelope."""
    points, bands = _points(12_000), _bands(20_000)
    # Control: the right side alone is far past the engine's budget, which is exactly the
    # case that used to raise rather than run.
    assert bands.nbytes > 2 * _TIGHT
    report = _run(tmp_path, _range_plan(join_type), [points, bands], _TIGHT)
    assert "spilled" in report, report
    duck.register("p", points)
    duck.register("b", bands)
    assert_same(report["table"], duck.sql(_JOIN_SQL[join_type]))


def test_control_the_blocked_range_join_equals_the_in_memory_one(tmp_path):
    """Batcher against itself: the tight run and the unconstrained run are the same relation."""
    points, bands = _points(12_000), _bands(20_000)
    tight = _run(tmp_path, _range_plan("full"), [points, bands], _TIGHT)["table"]
    wide = _run(tmp_path, _range_plan("full"), [points, bands], _WIDE)["table"]
    assert_tables_equal(tight, wide)


# --------------------------------------------------------------------------------------------
# An ASOF join with no `by` keys
# --------------------------------------------------------------------------------------------


def _asof_sides(unique_right: bool) -> tuple[pa.Table, pa.Table]:
    def lt(i: int) -> float | None:
        k = i % 29
        if k == 0:
            return None
        if k == 1:
            return -0.0
        return float((i * 11) % 40_000) / 2.0

    left = pa.table(
        {"lid": pa.array(range(15_000), pa.int64()), "t": pa.array([lt(i) for i in range(15_000)])}
    )
    if unique_right:
        rt = [None if i % 41 == 0 else float(i * 3) for i in range(8_000)]
    else:
        rt = [None if i % 41 == 0 else float((i * 3) % 2_000) for i in range(8_000)]
    right = pa.table({"rt": pa.array(rt), "q": pa.array(range(8_000), pa.int64())})
    return left, right


def _asof_plan(direction: str, **extra: object) -> dict:
    return {
        "op": "asof_join",
        "left": {"op": "scan", "source_id": 0},
        "right": {"op": "scan", "source_id": 1},
        "left_on": "t",
        "right_on": "rt",
        "left_by": [],
        "right_by": [],
        "direction": direction,
        "tolerance": extra.get("tolerance"),
        "allow_exact_matches": extra.get("allow_exact_matches", True),
        "output": [
            {"side": "left", "name": "lid", "alias": "lid"},
            {"side": "left", "name": "t", "alias": "t"},
            {"side": "right", "name": "q", "alias": "q"},
        ],
    }


@pytest.mark.parametrize(("direction", "op"), [("backward", ">="), ("forward", "<=")])
def test_a_keyless_asof_join_streams_and_matches_duckdb(duck, tmp_path, direction, op):
    """Unique right keys, so the match is defined without a tie rule DuckDB may not share."""
    left, right = _asof_sides(unique_right=True)
    report = _run(tmp_path, _asof_plan(direction), [left, right], _TIGHT)
    assert report["spilled"]["asof_join"], "the envelope must force the ASOF join out of core"
    duck.register("l", left)
    duck.register("r", right)
    expected = duck.sql(f"SELECT lid, t, q FROM l ASOF LEFT JOIN r ON l.t {op} r.rt")
    assert_same(report["table"], expected)


@pytest.mark.parametrize(
    "settings",
    [
        {"direction": "backward"},
        {"direction": "forward"},
        {"direction": "nearest"},
        {"direction": "backward", "allow_exact_matches": False},
        {"direction": "nearest", "tolerance": 1.0},
    ],
    ids=["backward", "forward", "nearest", "strict", "nearest_tolerance"],
)
def test_a_keyless_asof_join_over_tied_right_keys_equals_the_in_memory_join(tmp_path, settings):
    """Heavily tied right keys: which tied row is chosen must not move out of core."""
    left, right = _asof_sides(unique_right=False)
    settings = dict(settings)
    plan = _asof_plan(settings.pop("direction"), **settings)
    tight = _run(tmp_path, plan, [left, right], _TIGHT)
    assert tight["spilled"]["asof_join"], "the envelope must force the ASOF join out of core"
    wide = _run(tmp_path, plan, [left, right], _WIDE)
    assert not wide["spilled"]["asof_join"], "the reference run must be the in-memory kernel"
    assert_tables_equal(tight["table"], wide["table"])
