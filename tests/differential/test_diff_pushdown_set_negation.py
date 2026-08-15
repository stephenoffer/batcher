"""A pushed `IN`/`NOT`/`LIKE` returns the rows DuckDB returns, over a real parquet file.

The unit tests beside these check what each translator *emits*. This checks what the
emitted filter *selects*, which is the only thing that can catch the failure mode that
matters: a pushdown is invisible when it is right and silent when it is wrong, because the
engine's `Filter` re-checks the rows a source hands back — so a filter that returns too
*few* rows produces a wrong answer with no error anywhere.

Reading through parquet is what makes these exercise the pushdown at all; the same
predicates over `from_pydict` never reach a source. Each case is run against DuckDB over
the identical file, which fixes the three-valued-logic questions (`NOT` and `IN` over a
null, `LIKE` over a null) against a reference rather than against our own reading of them.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_DATA = {
    "a": [1, 2, 3, None, 5, 1],
    "b": [10, 20, 30, 40, None, 60],
    "s": ["ab", "cd", "ab_x", None, "ZZ", "a.b"],
}

# (batcher predicate, equivalent DuckDB WHERE clause)
_CASES = [
    (bt.col("a").is_in([1, 3]), "a IN (1, 3)"),
    (~bt.col("a").is_in([1, 3]), "NOT (a IN (1, 3))"),
    (~(bt.col("a") == 1), "NOT (a = 1)"),
    (bt.col("a").is_in([]), "false"),
    (bt.col("s").str.starts_with("ab"), "s LIKE 'ab%'"),
    (bt.col("s").str.ends_with("b"), "s LIKE '%b'"),
    (bt.col("s").str.contains("b"), "s LIKE '%b%'"),
    (~bt.col("s").str.starts_with("ab"), "NOT (s LIKE 'ab%')"),
    (bt.col("a").is_in([1, 3]) & (bt.col("b") > 10), "a IN (1, 3) AND b > 10"),
    (bt.col("a").is_in([1, 3]) | bt.col("s").str.starts_with("Z"), "a IN (1, 3) OR s LIKE 'Z%'"),
    (~((bt.col("a") == 1) & (bt.col("b") > 10)), "NOT (a = 1 AND b > 10)"),
    (~bt.col("a").is_null(), "a IS NOT NULL"),
    (bt.col("a").is_in([1, 3]) & bt.col("s").str.starts_with("a"), "a IN (1,3) AND s LIKE 'a%'"),
]


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory):
    """`_DATA` written as a parquet file both engines read."""
    path = tmp_path_factory.mktemp("pushdown") / "t.parquet"
    pq.write_table(pa.table(_DATA), path)
    return str(path)


@pytest.mark.parametrize(("predicate", "where"), _CASES, ids=[where for _, where in _CASES])
def test_pushed_filter_matches_duckdb(duck, parquet_path, predicate, where):
    got = bt.read.parquet(parquet_path).filter(predicate).collect()
    assert_same(got, duck.sql(f"SELECT * FROM read_parquet('{parquet_path}') WHERE {where}"))


@pytest.mark.parametrize(("predicate", "where"), _CASES, ids=[where for _, where in _CASES])
def test_pushed_filter_matches_the_unpushed_engine_filter(parquet_path, predicate, where):
    """The source-pushed path and the in-memory path agree.

    A second oracle for the same property that does not depend on DuckDB's spelling of
    `LIKE`: `from_pydict` reaches no source, so its `Filter` is the engine's own.
    """
    from _harness import assert_tables_equal

    pushed = bt.read.parquet(parquet_path).filter(predicate).collect()
    unpushed = bt.from_pydict(_DATA).filter(predicate).collect()
    assert_tables_equal(pushed, unpushed)
