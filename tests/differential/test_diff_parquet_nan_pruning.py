"""A float filter over Parquet keeps the NaN rows the same filter keeps in memory.

The engine ranks NaN above every number, as `ORDER BY` does, so `x > 0.9`, `x >= 0.9` and
`x != 0.5` are true for a NaN row. Parquet, Delta and Iceberg all keep NaN *out* of a float
column's recorded min/max, so a footer's max is the largest non-NaN value. Every layer that
proved a predicate empty from that max -- the native reader's row-group and page pruning, the
pyarrow filter, and Kyber's zone-map rules -- therefore dropped NaN rows the engine keeps, and
every terminal agreed on the wrong answer: over a row group of `[0.1, 0.5, NaN]`, `collect()`,
`count()`, `is_empty()`, `iter_batches()` and `collect(spill=True)` all said no rows where the
answer is one.

DuckDB has the same trap and an option out of it: its default prunes on the float max, and
`can_have_nan=true` treats the max as unknown. The oracle here is DuckDB with that option set,
and, independently, Batcher's own answer over the same table held in memory, where no footer
exists to mislead anything.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    # Row group 0 holds small values and NaNs, so its recorded max (0.5) sits below every
    # threshold tested; row group 1 holds large values; row group 2 is all NaN, which a writer
    # records with no bounds at all.
    small = [0.1, 0.5, float("nan")] * 200
    large = [0.95, 2.0, 3.0] * 200
    nans = [float("nan")] * 600
    f = np.array(small + large + nans)
    return pa.table({"f": pa.array(f), "id": pa.array(np.arange(len(f)))})


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    out = tmp_path_factory.mktemp("nan_pruning") / "t.parquet"
    pq.write_table(_table(), out, row_group_size=600)
    return str(out)


_PREDICATES = [
    ("f > 0.9", bt.col("f") > 0.9),
    ("f >= 0.9", bt.col("f") >= 0.9),
    ("f > 5.0", bt.col("f") > 5.0),
    ("f != 0.5", bt.col("f") != 0.5),
    ("f < 0.2", bt.col("f") < 0.2),
    ("f <= 0.5", bt.col("f") <= 0.5),
    ("f = 2.0", bt.col("f") == 2.0),
    ("NOT (f > 0.9)", ~(bt.col("f") > 0.9)),
]
_IDS = [sql for sql, _ in _PREDICATES]


@pytest.mark.parametrize("sql_pred, expr", _PREDICATES, ids=_IDS)
def test_rows_match_duckdb_and_memory(duck, path, sql_pred, expr):
    got = bt.read.parquet(path).filter(expr).select("id").collect()
    memory = bt.from_arrow(_table()).filter(expr).select("id").collect()
    sql = f"SELECT id FROM read_parquet('{path}', can_have_nan=true) WHERE {sql_pred}"
    assert_same(got, duck.sql(sql))
    assert sorted(got.column("id").to_pylist()) == sorted(memory.column("id").to_pylist())


@pytest.mark.parametrize("sql_pred, expr", _PREDICATES, ids=_IDS)
def test_every_terminal_agrees(duck, path, sql_pred, expr):
    (expected,) = duck.sql(
        f"SELECT count(*) FROM read_parquet('{path}', can_have_nan=true) WHERE {sql_pred}"
    ).fetchone()
    ds = bt.read.parquet(path).filter(expr)
    assert ds.count() == expected
    assert ds.is_empty() is (expected == 0)
    assert sum(b.num_rows for b in ds.iter_batches()) == expected
    assert ds.collect(spill=True).num_rows == expected


def test_the_fixture_hides_its_nans_from_the_footer(path):
    """The premise, checked: a row group whose NaNs its recorded max does not show."""
    stats = pq.ParquetFile(path).metadata.row_group(0).column(0).statistics
    assert stats.has_min_max and stats.max == 0.5
