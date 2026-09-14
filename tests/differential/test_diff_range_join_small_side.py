"""Range joins against a handful of right rows, vs DuckDB.

A right side of at most 32 rows takes a different strategy inside the operator
(`bc-runtime/src/join/range/small.rs`): the matches are found by scanning each right row
across the left key column rather than by sorting the left side. The choice is made on size
alone, so it can fire under any inequality shape, any operator combination and any join type
-- which is what these cover.

**Every case is over 16,384 left rows on purpose.** That is the operator's own floor for the
scan, so a smaller fixture would run the sorted path and this file would report on code it
never reached. The fixture also crosses the 65,536-row chunk boundary the scan parallelizes
on, so a row emitted by two chunks or by none would show up here.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_ROWS = 70_000


@pytest.fixture(scope="module")
def probe() -> pa.Table:
    # A tenth of the keys are NULL, and the values run either side of every bucket edge.
    keys = [None if i % 10 == 7 else (i * 7919) % 1000 for i in range(_ROWS)]
    return pa.table({"k": pa.array(keys, pa.int64()), "v": pa.array(range(_ROWS), pa.int64())})


def _buckets() -> pa.Table:
    """Six disjoint bands, plus one that overlaps two of them and one that is empty."""
    lo = [0, 100, 200, 300, 400, 500, 150, 900]
    hi = [100, 200, 300, 400, 500, 600, 350, 900]
    return pa.table({"lo": pa.array(lo, pa.int64()), "hi": pa.array(hi, pa.int64())})


@pytest.fixture
def both(duck, probe):
    """A Batcher session and the DuckDB connection, holding the same two tables."""
    session = bt.Session()
    for name, table in (("probe", probe), ("buckets", _buckets())):
        session.register(name, table)
        duck.register(name, table)

    def run(sql: str):
        assert_same(session.sql(sql).collect(), duck.sql(sql))

    return run


_BAND = "p.k >= b.lo AND p.k < b.hi"


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT p.v, b.lo FROM probe p JOIN buckets b ON {_BAND}",
        f"SELECT p.v, b.lo FROM probe p LEFT JOIN buckets b ON {_BAND}",
        f"SELECT p.v FROM probe p WHERE EXISTS (SELECT 1 FROM buckets b WHERE {_BAND})",
        f"SELECT p.v FROM probe p WHERE NOT EXISTS (SELECT 1 FROM buckets b WHERE {_BAND})",
    ],
    ids=["inner", "left", "semi", "anti"],
)
def test_a_bucket_table_matches_duckdb(both, sql):
    """The shape the strategy exists for: two right columns bounding one left key.

    Inner, outer and both existence forms, because the scan emits unmatched left rows itself
    rather than inheriting them from a sorted sweep. (`SEMI JOIN` / `ANTI JOIN` are written as
    `EXISTS` / `NOT EXISTS` because the SQL front-end does not parse the explicit spelling.)
    """
    both(sql)


@pytest.mark.parametrize(
    "predicate",
    [
        "p.k < b.lo",
        "p.k <= b.lo",
        "p.k > b.lo",
        "p.k >= b.lo",
        "p.k > b.lo AND p.k > b.hi",
        "p.k <= b.lo AND p.k >= b.hi",
        "p.k >= b.lo AND p.k <= b.hi",
    ],
)
def test_every_operator_combination_matches_duckdb(both, predicate):
    """One inequality and two, including two facing the same way -- neither a band nor a
    bucket, and the shape the sorted paths route to IEJoin."""
    both(f"SELECT COUNT(*) AS n, SUM(p.v) AS s FROM probe p JOIN buckets b ON {predicate}")


@pytest.mark.parametrize("n_right", [1, 32, 33])
def test_the_answer_does_not_change_across_the_strategy_threshold(duck, probe, n_right):
    """32 right rows scan and 33 sort, so the same query either side of that must agree --
    with DuckDB and therefore with each other."""
    bands = pa.table(
        {
            "lo": pa.array([i * 25 for i in range(n_right)], pa.int64()),
            "hi": pa.array([i * 25 + 40 for i in range(n_right)], pa.int64()),
        }
    )
    session = bt.Session()
    session.register("probe", probe)
    session.register("bands", bands)
    duck.register("probe", probe)
    duck.register("bands", bands)
    sql = (
        "SELECT COUNT(*) AS n, SUM(p.v) AS s, MIN(b.lo) AS lo FROM probe p "
        "JOIN bands b ON p.k >= b.lo AND p.k < b.hi"
    )
    assert_same(session.sql(sql).collect(), duck.sql(sql))
