"""A metadata answer must equal executing the query. Always. That is the whole contract.

Kyber can answer some terminals from statistics without touching a row — `count(*)` from a row
count, `min(x)` from a footer bound. That is only ever an *optimization*: the answer it returns
must be the answer the engine would have computed. If the two can differ, the optimizer is a
wrong-answer generator, and the faster it is the worse it is.

They *did* differ. `max(f)` over a float column containing NaN was answered from the stored
bound as the largest **non-NaN** value, because both producers of that bound deliberately drop
NaN (the KLL sketch ignores it — it has no rank; the Parquet spec omits it from statistics) —
while SQL's total order, and our own `ORDER BY`, make NaN the *greatest* value. So the metadata
path said `3.0` and executing said `nan`, and no test compared the two.

This file compares the two. It does not assert *what* the answer is (the per-operator
differential tests do that) — it asserts the shortcut and the long way round agree, which is
the property the optimization actually owes us.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

#: Columns chosen so the *bounds* are interesting: NaN (greater than everything), -0.0/0.0
#: (equal but differently encoded), nulls, and an all-NaN column (no bound exists at all).
COLUMNS = {
    "i": pa.array([5, 1, 9, 3, None], pa.int64()),
    "f": pa.array([2.5, float("nan"), 0.5, -0.0, None], pa.float64()),
    "f_clean": pa.array([2.5, 0.5, 8.5, 4.0, None], pa.float64()),
    "f_allnan": pa.array([float("nan")] * 5, pa.float64()),
    "s": pa.array(["pear", "apple", None, "fig", "kiwi"]),
}

AGGREGATES = {
    "min": lambda c: bt.col(c).min(),
    "max": lambda c: bt.col(c).max(),
    "count": lambda c: bt.col(c).count(),
    "count_distinct": lambda c: bt.col(c).n_unique(),
}


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory) -> str:
    """The same table on disk, so the footer-bound path is exercised too."""
    path = str(tmp_path_factory.mktemp("meta") / "t.parquet")
    pq.write_table(pa.table(COLUMNS), path)
    return path


def _forced(ds):
    """The same relation, with the metadata layer switched off — the comparison's whole point.

    This used to be a `filter(true)`, on the theory that an identity filter "downgrades the
    stats away from EXACT". It does not: the optimizer folds an always-true predicate away, the
    statistics come back EXACT, and the "forced" path was answered from metadata too — so the
    test compared a metadata answer to itself and could not have caught the very class of bug
    it was written for.

    `map_batches` genuinely forces it: the IR cannot describe a Python callback, so Kyber
    refuses to reason about the plan at all and every terminal falls through to the engine. The
    callback is the identity, so not a row changes — this is the same relation, computed the
    long way round.
    """
    return ds.map_batches(lambda batch: batch)


@pytest.mark.parametrize("column", sorted(COLUMNS))
@pytest.mark.parametrize("agg", sorted(AGGREGATES))
@pytest.mark.parametrize("source", ["memory", "parquet"])
def test_metadata_answer_equals_execution(parquet_path, source, column, agg):
    """The shortcut and the full run must produce the same value, for every (agg, column)."""
    table = pa.table(COLUMNS)
    base = bt.from_arrow(table) if source == "memory" else bt.read.parquet(parquet_path)
    shortcut = base.agg(out=AGGREGATES[agg](column)).collect()

    # The same query, genuinely forced through the engine (see `_forced`).
    forced = _forced(base).agg(out=AGGREGATES[agg](column)).collect()

    assert_tables_equal(shortcut, forced)


@pytest.mark.parametrize("column", sorted(COLUMNS))
@pytest.mark.parametrize("agg", sorted(AGGREGATES))
def test_metadata_answer_matches_duckdb(duck, column, agg):
    """...and both match the oracle."""
    table = pa.table(COLUMNS)
    duck.register("t", table)
    got = bt.from_arrow(table).agg(out=AGGREGATES[agg](column)).collect()
    sql = {
        "min": f"SELECT min({column}) AS out FROM t",
        "max": f"SELECT max({column}) AS out FROM t",
        "count": f"SELECT count({column}) AS out FROM t",
        "count_distinct": f"SELECT count(DISTINCT {column}) AS out FROM t",
    }[agg]
    want = duck.sql(sql).to_arrow_table()
    assert_tables_equal(got, want)
