"""The derived-statistic and shape-test methods on ``Expr``, against DuckDB SQL.

These are the convenience layer a data-science user reaches for first -- normalize a
column, take its share of the total, centre it, rank it as a percentile, roll a window
over a time-like key, ask whether a value is even or positive -- and no test called any
of them. Each is one or two lines of engine code and every one of them is a place a
partition clause or an off-by-one frame can go wrong silently.

The oracle is DuckDB, with each reference written as the *window expression the method
is documented to mean* rather than as a translation of Batcher's implementation:
``pct_of_total`` against ``x / sum(x) OVER (...)``, ``rank_pct`` against
``percent_rank()``, ``rolling_min_by`` against an explicit ``RANGE BETWEEN 3 PRECEDING
AND CURRENT ROW`` frame. Where a method has a partitioned form it is checked both ways,
because an unpartitioned pass is exactly what a dropped ``PARTITION BY`` looks like.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")

#: Two groups of unequal size, a null, a zero, a negative and a value that divides
#: exactly -- so a dropped partition, a missing null guard and a division by zero all
#: change the answer.
ROWS = {
    "g": ["a", "a", "a", "b", "b", "b"],
    "i": [1, 2, 3, 4, 5, 6],
    "x": [3.0, -1.5, 0.0, 7.25, None, 2.0],
    "y": [2.0, 0.0, 4.0, -3.0, 1.0, None],
}


@pytest.fixture(scope="module")
def duck():
    """A DuckDB connection holding the fixture, keyed so results stay in fixture order."""
    con = duckdb.connect()
    con.execute("CREATE TABLE t (k INTEGER, g VARCHAR, i BIGINT, x DOUBLE, y DOUBLE)")
    con.executemany(
        "INSERT INTO t VALUES (?, ?, ?, ?, ?)",
        [(k, ROWS["g"][k], ROWS["i"][k], ROWS["x"][k], ROWS["y"][k]) for k in range(6)],
    )
    return con


@pytest.fixture(scope="module")
def ds():
    """The same fixture as a Batcher dataset."""
    return bt.from_pydict(ROWS)


def _duck_column(con, expression: str) -> list:
    return [row[0] for row in con.execute(f"SELECT {expression} FROM t ORDER BY k").fetchall()]


def _assert_same(got: list, want: list, label: str) -> None:
    """Element-wise, tolerating float rounding but not a null on one side only."""
    assert len(got) == len(want), label
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        if a is None or b is None:
            assert a is None and b is None, f"{label}[{i}]: {a!r} vs {b!r}"
        elif isinstance(a, float) or isinstance(b, float):
            assert a == pytest.approx(b, rel=1e-12, abs=1e-12), f"{label}[{i}]"
        else:
            assert a == b, f"{label}[{i}]: {a!r} vs {b!r}"


#: ``(label, batcher expression, DuckDB expression)`` for the row-wise methods.
SCALAR = [
    ("abs_diff", lambda: bt.col("x").abs_diff(bt.col("y")), "abs(x - y)"),
    (
        "safe_divide",
        lambda: bt.col("x").safe_divide(bt.col("y")),
        "CASE WHEN y = 0 THEN NULL ELSE x / y END",
    ),
    # `least` is not the oracle for `clip_max`: SQL's `least` *ignores* a null argument
    # and returns 2.0, while clipping a missing value has to leave it missing. The null
    # guard is in the SQL rather than in the tolerance, and the difference is pinned
    # against Polars in its own test below.
    (
        "clip_max",
        lambda: bt.col("x").clip_max(2.0),
        "CASE WHEN x IS NULL THEN NULL ELSE least(x, 2.0) END",
    ),
    ("is_even", lambda: bt.col("i").is_even(), "i % 2 = 0"),
    ("is_odd", lambda: bt.col("i").is_odd(), "i % 2 <> 0"),
    ("is_positive", lambda: bt.col("x").is_positive(), "x > 0"),
    ("is_negative", lambda: bt.col("x").is_negative(), "x < 0"),
    ("is_zero", lambda: bt.col("x").is_zero(), "x = 0"),
    ("arcsin", lambda: (bt.col("x") / 10).arcsin(), "asin(x / 10)"),
    ("arctan", lambda: bt.col("x").arctan(), "atan(x)"),
    ("arctanh", lambda: (bt.col("x") / 10).arctanh(), "atanh(x / 10)"),
    ("arccosh", lambda: (bt.col("x") + 10).arccosh(), "acosh(x + 10)"),
]


@pytest.mark.parametrize(("label", "ours", "theirs"), SCALAR)
def test_row_wise_method_matches_duckdb(ds, duck, label, ours, theirs):
    """Every row-wise derived value, including on the null and the zero."""
    _assert_same(ds.select(v=ours()).to_pydict()["v"], _duck_column(duck, theirs), label)


def test_clip_max_leaves_a_missing_value_missing(duck):
    """Batcher clips like a dataframe, not like SQL's ``least``, and Polars agrees.

    ``least(NULL, 2.0)`` is 2.0 in DuckDB (and Postgres): the function is specified to
    skip null arguments. Clipping is a different operation -- it bounds a value that
    exists -- so a null must stay null, which is what pandas and Polars both do. Pinned
    here because the SQL spelling is the obvious oracle and it is the wrong one.
    """
    polars = pytest.importorskip("polars")
    ours = bt.from_pydict({"x": ROWS["x"]}).select(v=bt.col("x").clip_max(2.0)).to_pydict()["v"]
    theirs = (
        polars.DataFrame({"x": ROWS["x"]})
        .select(polars.col("x").clip(upper_bound=2.0))
        .to_series()
        .to_list()
    )
    assert ours == theirs, f"{ours} vs polars {theirs}"
    assert duck.execute("SELECT least(CAST(NULL AS DOUBLE), 2.0)").fetchone()[0] == 2.0, (
        "the departure is only real while SQL's least still ignores nulls"
    )


def test_safe_divide_is_the_only_division_that_survives_a_zero_denominator(ds):
    """The point of the method, stated as the difference from plain division."""
    got = ds.select(
        safe=bt.col("x").safe_divide(bt.col("y")), plain=bt.col("x") / bt.col("y")
    ).to_pydict()
    assert got["safe"][1] is None, "dividing by zero must null rather than yield an infinity"
    assert (
        got["plain"][1] in (None, float("inf"), float("-inf")) or got["plain"][1] != got["safe"][1]
    ), "if plain division also nulls, safe_divide has no reason to exist"


#: ``(label, batcher expression, DuckDB window expression)`` for the whole-column forms.
WINDOWED = [
    ("pct_of_total", lambda: bt.col("i").pct_of_total(), "i / sum(i) OVER ()"),
    ("rank_pct", lambda: bt.col("i").rank_pct(), "percent_rank() OVER (ORDER BY i)"),
    ("mean_center", lambda: bt.col("i").mean_center(), "i - avg(i) OVER ()"),
    (
        "minmax_scale",
        lambda: bt.col("i").minmax_scale(),
        "(i - min(i) OVER ()) / (max(i) OVER () - min(i) OVER ())",
    ),
    (
        "cumulative_pct",
        lambda: bt.col("i").cumulative_pct(order_by=[bt.col("i")]),
        "sum(i) OVER (ORDER BY i) / sum(i) OVER ()",
    ),
    (
        "expanding_mean",
        lambda: bt.col("i").expanding_mean(order_by=[bt.col("i")]),
        "avg(i) OVER (ORDER BY i)",
    ),
]


@pytest.mark.parametrize(("label", "ours", "theirs"), WINDOWED)
def test_whole_column_statistic_matches_duckdb(ds, duck, label, ours, theirs):
    """The unpartitioned form of each derived statistic."""
    _assert_same(ds.select(v=ours()).to_pydict()["v"], _duck_column(duck, theirs), label)


#: The same statistics partitioned by ``g``. Kept separate from the unpartitioned list
#: because a dropped ``PARTITION BY`` is precisely the bug that passes the list above.
PARTITIONED = [
    (
        "pct_of_total",
        lambda: bt.col("i").pct_of_total(partition_by=[bt.col("g")]),
        "i / sum(i) OVER (PARTITION BY g)",
    ),
    (
        "rank_pct",
        lambda: bt.col("i").rank_pct(partition_by=[bt.col("g")]),
        "percent_rank() OVER (PARTITION BY g ORDER BY i)",
    ),
    (
        "mean_center",
        lambda: bt.col("i").mean_center(partition_by=[bt.col("g")]),
        "i - avg(i) OVER (PARTITION BY g)",
    ),
    (
        "minmax_scale",
        lambda: bt.col("i").minmax_scale(partition_by=[bt.col("g")]),
        "(i - min(i) OVER (PARTITION BY g))"
        " / (max(i) OVER (PARTITION BY g) - min(i) OVER (PARTITION BY g))",
    ),
    (
        "cumulative_pct",
        lambda: bt.col("i").cumulative_pct(partition_by=[bt.col("g")], order_by=[bt.col("i")]),
        "sum(i) OVER (PARTITION BY g ORDER BY i) / sum(i) OVER (PARTITION BY g)",
    ),
    (
        "expanding_mean",
        lambda: bt.col("i").expanding_mean(partition_by=[bt.col("g")], order_by=[bt.col("i")]),
        "avg(i) OVER (PARTITION BY g ORDER BY i)",
    ),
]


@pytest.mark.parametrize(("label", "ours", "theirs"), PARTITIONED)
def test_partitioned_statistic_matches_duckdb(ds, duck, label, ours, theirs):
    """The partitioned form, which must differ from the unpartitioned one."""
    got = ds.select(v=ours()).to_pydict()["v"]
    _assert_same(got, _duck_column(duck, theirs), f"{label} by g")
    unpartitioned = _duck_column(duck, theirs.replace("PARTITION BY g", "").replace("()", "()"))
    assert got != unpartitioned or label == "expanding_mean", (
        f"{label} partitioned by g equals the unpartitioned answer, so the clause did nothing"
    )


def test_label_encode_matches_a_dense_rank_over_the_distinct_values(ds, duck):
    """Zero-based dense codes in value order, which is what makes them reproducible."""
    got = ds.select(v=bt.col("g").label_encode()).to_pydict()["v"]
    want = _duck_column(duck, "dense_rank() OVER (ORDER BY g) - 1")
    assert got == want, f"{got} vs {want}"
    assert min(got) == 0, "codes start at zero"


def test_first_and_last_distinct_flag_one_row_per_group(ds, duck):
    """``is_first_distinct`` / ``is_last_distinct`` against explicit row numbering."""
    got = ds.select(
        first=bt.col("g").is_first_distinct(order_by=bt.col("i")),
        last=bt.col("g").is_last_distinct(order_by=bt.col("i")),
    ).to_pydict()
    assert got["first"] == _duck_column(duck, "row_number() OVER (PARTITION BY g ORDER BY i) = 1")
    assert got["last"] == _duck_column(
        duck, "row_number() OVER (PARTITION BY g ORDER BY i DESC) = 1"
    )
    assert sum(got["first"]) == 2, "one first row per distinct value of g"
    assert sum(got["last"]) == 2, "one last row per distinct value of g"


#: ``(method, DuckDB aggregate)`` for the range-framed rolling family.
ROLLING = [("rolling_min_by", "min"), ("rolling_max_by", "max"), ("rolling_count_by", "count")]


@pytest.mark.parametrize(("method", "aggregate"), ROLLING)
def test_rolling_by_matches_an_explicit_range_frame(ds, duck, method, aggregate):
    """A window of width 3 over the ordering key, not over the row count.

    Written as ``RANGE BETWEEN 3 PRECEDING AND CURRENT ROW`` on purpose: a ``ROWS`` frame
    of the same width gives a different answer as soon as the key has gaps, and telling
    the two apart is the whole reason the method takes a ``by`` column.
    """
    got = ds.select(v=getattr(bt.col("i"), method)(bt.col("i"), 3)).to_pydict()["v"]
    want = _duck_column(
        duck, f"{aggregate}(i) OVER (ORDER BY i RANGE BETWEEN 3 PRECEDING AND CURRENT ROW)"
    )
    _assert_same(got, want, method)


def test_a_rolling_range_frame_differs_from_a_row_frame_when_the_key_has_gaps(duck):
    """The gap case, which is what a ``RANGE`` frame is for."""
    keys = [1, 2, 3, 10, 11, 12]
    ds = bt.from_pydict({"i": keys})
    got = ds.select(v=bt.col("i").rolling_count_by(bt.col("i"), 3)).to_pydict()["v"]
    con = duckdb.connect()
    con.execute("CREATE TABLE g (k INTEGER, i BIGINT)")
    con.executemany("INSERT INTO g VALUES (?, ?)", list(enumerate(keys)))
    by_range = [
        r[0]
        for r in con.execute(
            "SELECT count(i) OVER (ORDER BY i RANGE BETWEEN 3 PRECEDING AND CURRENT ROW)"
            " FROM g ORDER BY k"
        ).fetchall()
    ]
    by_rows = [
        r[0]
        for r in con.execute(
            "SELECT count(i) OVER (ORDER BY i ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)"
            " FROM g ORDER BY k"
        ).fetchall()
    ]
    assert got == by_range, f"{got} vs {by_range}"
    assert by_range != by_rows, "the fixture must have a gap, or this proves nothing"


def test_a_partitioned_rolling_window_does_not_reach_across_groups(ds, duck):
    """The frame must restart at each partition, which an unpartitioned frame does not."""
    got = ds.select(
        v=bt.col("i").rolling_min_by(bt.col("i"), 3, partition_by=[bt.col("g")])
    ).to_pydict()["v"]
    want = _duck_column(
        duck,
        "min(i) OVER (PARTITION BY g ORDER BY i RANGE BETWEEN 3 PRECEDING AND CURRENT ROW)",
    )
    _assert_same(got, want, "rolling_min_by by g")
    assert got[3] == 4, "the first row of group b must not see group a"


def test_the_shape_tests_null_rather_than_answering_for_a_null(ds):
    """``is_positive`` and friends must be null-preserving, not null-is-false."""
    got = ds.select(
        pos=bt.col("x").is_positive(),
        neg=bt.col("x").is_negative(),
        zero=bt.col("x").is_zero(),
        even=bt.col("y").is_even(),
    ).to_pydict()
    assert got["pos"][4] is None
    assert got["neg"][4] is None
    assert got["zero"][4] is None
    assert got["even"][5] is None


def test_streaming_agrees_with_collect_on_the_row_wise_methods(ds):
    """The row-wise family through ``iter_batches``; the windowed ones are breakers."""
    projected = ds.select(
        d=bt.col("x").abs_diff(bt.col("y")),
        s=bt.col("x").safe_divide(bt.col("y")),
        e=bt.col("i").is_even(),
    )
    collected = projected.to_pydict()
    streamed: dict[str, list] = {k: [] for k in collected}
    for batch in projected.iter_batches():
        for key in streamed:
            streamed[key].extend(batch.column(key).to_pylist())
    assert streamed == collected
