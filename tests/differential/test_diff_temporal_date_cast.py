"""`CAST(ts AS DATE) <op> DATE 'd'` becomes a timestamp band — DuckDB is the oracle.

The rewrite replaces a per-row cast with a bound on the raw column, so the only thing that can
go wrong is the *boundary*, and it can go wrong silently in either direction: an inclusive bound
written exclusive drops the midnight row, and an exclusive one written inclusive keeps a row
from the next day. The fixture therefore puts a row on every edge — exactly midnight, one
microsecond before it, one microsecond after, the last instant of the day — and every one of the
six comparisons is checked against DuckDB evaluating the cast as written.

One divergence is deliberately not asserted against DuckDB, and it is the engines' rather than
this family's: Batcher casts a *timezone-aware* timestamp to a date in the column's zone while
DuckDB uses its session zone, so the two disagree about which local day an instant falls on
before any optimization runs. The rewrite declines on such a column anyway (a local midnight is
not a fixed instant), and that decline is asserted as plan shape instead.

The pre-epoch rows are there for a specific reason. The rewrite is exact only because the cast
*floors* rather than truncating toward zero, which matters only for negative instants: a
truncating cast would map `1969-12-31 12:00` to `1970-01-01` and the band would be wrong by a
day on that side of the epoch. Those rows are what would catch it.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.temporal_date_cast  # registers the rules under test
from _harness import assert_same
from batcher import col, lit

pytestmark = pytest.mark.differential

_DAY = dt.date(2024, 1, 1)
_MICRO = dt.timedelta(microseconds=1)


@pytest.fixture
def t(duck):
    """Rows on every edge of 2024-01-01, plus pre-epoch instants and a null."""
    instants = [
        dt.datetime(2023, 12, 31, 23, 59, 59, 999999),  # the microsecond before the day
        dt.datetime(2024, 1, 1),  # exactly midnight — the inclusive lower bound
        dt.datetime(2024, 1, 1, 0, 0, 0, 1),  # just inside
        dt.datetime(2024, 1, 1, 12),  # midday
        dt.datetime(2024, 1, 1, 23, 59, 59, 999999),  # the last instant of the day
        dt.datetime(2024, 1, 2),  # exactly the exclusive upper bound
        dt.datetime(2024, 6, 15, 8, 30),  # well after
        # Pre-epoch: only a *flooring* cast puts these on the dates they belong to.
        dt.datetime(1969, 12, 31, 12),
        dt.datetime(1969, 12, 31),
        dt.datetime(1969, 12, 30, 23, 59, 59),
    ]
    tbl = pa.table(
        {
            "ts": pa.array([*instants, None], pa.timestamp("us")),
            "n": pa.array(list(range(len(instants) + 1)), pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table({"ts": pa.array([], pa.timestamp("us")), "n": pa.array([], pa.int64())})
    duck.register("empty", tbl)
    return tbl


@pytest.fixture
def tzt(duck):
    """A timezone-aware column, where the rewrite must decline and still agree."""
    tbl = pa.table(
        {
            "ts": pa.array(
                [
                    dt.datetime(2024, 1, 1, 2, tzinfo=dt.UTC),
                    dt.datetime(2024, 1, 1, 20, tzinfo=dt.UTC),
                    None,
                ],
                pa.timestamp("us", tz="UTC"),
            )
        }
    )
    duck.register("tzt", tbl)
    return tbl


@pytest.mark.parametrize(
    ("pred", "sql"),
    [
        (lambda: col("ts").cast("date") == lit(_DAY), "CAST(ts AS DATE) = DATE '2024-01-01'"),
        (lambda: col("ts").cast("date") != lit(_DAY), "CAST(ts AS DATE) <> DATE '2024-01-01'"),
        (lambda: col("ts").cast("date") < lit(_DAY), "CAST(ts AS DATE) < DATE '2024-01-01'"),
        (lambda: col("ts").cast("date") <= lit(_DAY), "CAST(ts AS DATE) <= DATE '2024-01-01'"),
        (lambda: col("ts").cast("date") > lit(_DAY), "CAST(ts AS DATE) > DATE '2024-01-01'"),
        (lambda: col("ts").cast("date") >= lit(_DAY), "CAST(ts AS DATE) >= DATE '2024-01-01'"),
        # The mirrored spelling, which arrives as the opposite operator.
        (lambda: lit(_DAY) < col("ts").cast("date"), "DATE '2024-01-01' < CAST(ts AS DATE)"),
        (lambda: lit(_DAY) >= col("ts").cast("date"), "DATE '2024-01-01' >= CAST(ts AS DATE)"),
        # A pre-epoch day, where a truncating (rather than flooring) cast would be off by one.
        (
            lambda: col("ts").cast("date") == lit(dt.date(1969, 12, 31)),
            "CAST(ts AS DATE) = DATE '1969-12-31'",
        ),
        (
            lambda: col("ts").cast("date") >= lit(dt.date(1969, 12, 31)),
            "CAST(ts AS DATE) >= DATE '1969-12-31'",
        ),
    ],
)
def test_every_comparison_matches_duckdb(duck, t, pred, sql):
    out = bt.from_arrow(t).filter(pred()).collect()
    assert_same(out, duck.sql(f"SELECT * FROM t WHERE {sql}"))


def test_band_keeps_the_midnight_row_and_drops_the_next_days(duck, t):
    # Stated as a value assertion as well as an oracle comparison, because this is the exact
    # off-by-one the rewrite could introduce and `assert_same` alone would not name it.
    out = bt.from_arrow(t).filter(col("ts").cast("date") == lit(_DAY)).collect()
    kept = set(out.column("ts").to_pylist())
    assert dt.datetime(2024, 1, 1) in kept, "the inclusive lower bound (midnight) was dropped"
    assert dt.datetime(2024, 1, 1, 23, 59, 59, 999999) in kept, "the last instant was dropped"
    assert dt.datetime(2024, 1, 2) not in kept, "the exclusive upper bound was kept"
    assert dt.datetime(2023, 12, 31, 23, 59, 59, 999999) not in kept, "the prior day was kept"
    assert_same(out, duck.sql("SELECT * FROM t WHERE CAST(ts AS DATE) = DATE '2024-01-01'"))


def test_timezone_aware_column_is_left_as_written(tzt):
    """The rewrite declines on a tz-aware column, asserted as plan shape rather than by oracle.

    DuckDB cannot be the oracle here, and the reason is the engines', not this rule's: Batcher
    casts a tz-aware timestamp to a date in the **column's** zone, DuckDB in its **session**
    zone. So `2024-01-01 02:00Z` is 2024-01-01 to Batcher and 2023-12-31 to DuckDB under a
    Los Angeles session, and the two disagree about which local day an instant falls on before
    any optimization happens. That divergence is pre-existing and outside this family's scope;
    what is in scope is that the rewrite does not fire, since a local midnight is not a fixed
    instant and no naive band names it.
    """
    from batcher import core, kyber
    from batcher.plan.expr_ir import Cast
    from batcher.plan.logical import Filter
    from batcher.plan.visitor import walk

    dataset = bt.from_arrow(tzt).filter(col("ts").cast("date") == lit(_DAY))
    plan = kyber.optimize_logical(dataset._plan, sources=dataset._sources, hub=core.default_hub())
    predicates = [n.predicate for n in walk(plan) if isinstance(n, Filter)]
    assert predicates, "the filter disappeared"
    assert any(isinstance(e, Cast) for p in predicates for e in _walk_expr(p)), (
        "the cast was rewritten away on a timezone-aware column"
    )


def _walk_expr(expr):
    """Every node of an expression tree, for the plan-shape assertion above."""
    from batcher.plan.expr_rewrite import transform_expr_up

    seen = []
    transform_expr_up(expr, lambda e: (seen.append(e), e)[1])
    return seen


def test_empty_input(duck, empty):
    out = bt.from_arrow(empty).filter(col("ts").cast("date") == lit(_DAY)).collect()
    assert_same(out, duck.sql("SELECT * FROM empty WHERE CAST(ts AS DATE) = DATE '2024-01-01'"))


def test_inside_a_conjunction(duck, t):
    out = bt.from_arrow(t).filter((col("ts").cast("date") >= lit(_DAY)) & (col("n") > 3)).collect()
    assert_same(
        out, duck.sql("SELECT * FROM t WHERE CAST(ts AS DATE) >= DATE '2024-01-01' AND n > 3")
    )


def test_inside_a_disjunction(duck, t):
    out = bt.from_arrow(t).filter((col("ts").cast("date") == lit(_DAY)) | (col("n") == 0)).collect()
    assert_same(
        out, duck.sql("SELECT * FROM t WHERE CAST(ts AS DATE) = DATE '2024-01-01' OR n = 0")
    )


def test_under_a_negation(duck, t):
    # `NOT` is where a NULL turning into a FALSE would become visible, so the three-valued
    # behaviour of the band is checked rather than assumed.
    out = bt.from_arrow(t).filter(~(col("ts").cast("date") == lit(_DAY))).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE NOT (CAST(ts AS DATE) = DATE '2024-01-01')"))


def test_in_a_projection(duck, t):
    # In a projection the three-valued result itself is observable, not just which rows pass.
    out = bt.from_arrow(t).select(n=col("n"), r=col("ts").cast("date") == lit(_DAY)).collect()
    assert_same(out, duck.sql("SELECT n, CAST(ts AS DATE) = DATE '2024-01-01' AS r FROM t"))


def test_every_execution_path_agrees(t):
    from _harness import assert_tables_equal

    def build(ds):
        return ds.filter(col("ts").cast("date") >= lit(_DAY)).sort("n")

    oracle = build(bt.from_arrow(t)).collect()
    assert_tables_equal(build(bt.from_arrow(t)).collect(spill=True), oracle, ordered=True)
    batches = list(build(bt.from_arrow(t)).iter_batches())
    streamed = (
        pa.Table.from_batches(batches, schema=batches[0].schema) if batches else oracle.slice(0, 0)
    )
    assert_tables_equal(streamed, oracle, ordered=True)
