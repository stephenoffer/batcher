"""A dictionary-encoded column must produce DuckDB's answer, from metadata or from a scan.

The exact statistics for a categorical column -- distinct count, mean, sum, bounds -- were all
`None`, because `pa.types.is_integer` is `False` for a dictionary type and Arrow's aggregate
kernels have no dictionary kernel at all. Enabling them turns on a *metadata* fast path that
answers an unfiltered aggregate without re-scanning, and a fast path that disagrees with
execution is the worst kind of wrong: silently wrong, only on the second run, only on the
columns Parquet and pandas hand you by default.

So the oracle is asked directly. Every aggregate here has an exact metadata answer available,
and DuckDB decides whether that answer is right.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

_ROWS = 5_000


def _encoded(values: np.ndarray | list) -> pa.Table:
    """The same relation twice: dictionary-encoded, alongside a plain grouping column."""
    return pa.table(
        {
            "c": pa.array(values).dictionary_encode(),
            "g": pa.array(np.arange(len(values)) % 4),
        }
    )


_SHAPES = {
    "int64": (np.arange(_ROWS) % 97).astype("int64"),
    "float64": ((np.arange(_ROWS) % 97) * 1.5).astype("float64"),
    "string": np.array([f"v{i % 97}" for i in range(_ROWS)]),
    "with_nulls": [None if i % 13 == 0 else int(i % 97) for i in range(_ROWS)],
    "single_value": np.zeros(_ROWS, dtype="int64"),
    "all_distinct": np.arange(_ROWS).astype("int64"),
}

_NUMERIC = ("int64", "float64", "with_nulls", "single_value", "all_distinct")


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_count_distinct_matches_duckdb(duck, shape):
    """The statistic that was `None` on every categorical column."""
    table = _encoded(_SHAPES[shape])
    duck.register("t", table)
    got = bt.from_arrow(table).select(n=bt.col("c").n_unique()).collect()
    assert_same(got, duck.sql("SELECT COUNT(DISTINCT c) AS n FROM t"))


@pytest.mark.parametrize("shape", sorted(_NUMERIC))
def test_min_max_sum_mean_match_duckdb(duck, shape):
    """The ordered-type gate rejected these for reading the dictionary label."""
    table = _encoded(_SHAPES[shape])
    duck.register("t", table)
    got = bt.from_arrow(table).select(
        lo=bt.col("c").min(),
        hi=bt.col("c").max(),
        total=bt.col("c").sum(),
        avg=bt.col("c").mean(),
    )
    assert_same(
        got.collect(),
        duck.sql("SELECT MIN(c) AS lo, MAX(c) AS hi, SUM(c) AS total, AVG(c) AS avg FROM t"),
    )


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_group_by_a_dictionary_column_matches_duckdb(duck, shape):
    table = _encoded(_SHAPES[shape])
    duck.register("t", table)
    got = bt.from_arrow(table).group_by("c").agg(n=bt.col("g").count()).collect()
    assert_same(got, duck.sql("SELECT c, COUNT(g) AS n FROM t GROUP BY c"))


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_filtering_a_dictionary_column_matches_duckdb(duck, shape):
    """Equality on a categorical is the most common predicate there is."""
    table = _encoded(_SHAPES[shape])
    duck.register("t", table)
    needle = table.column("c").to_pylist()[1]
    if needle is None:
        pytest.skip("shape has no non-null value at that position")
    got = bt.from_arrow(table).filter(bt.col("c") == needle).select("c", "g").collect()
    duck.execute("CREATE OR REPLACE TEMP TABLE needle AS SELECT ? AS v", [needle])
    assert_same(got, duck.sql("SELECT c, g FROM t WHERE c = (SELECT v FROM needle)"))


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_null_counting_matches_duckdb(duck, shape):
    table = _encoded(_SHAPES[shape])
    duck.register("t", table)
    got = bt.from_arrow(table).filter(bt.col("c").is_not_null()).select(n=bt.col("c").count())
    assert_same(got.collect(), duck.sql("SELECT COUNT(c) AS n FROM t WHERE c IS NOT NULL"))


def test_a_dictionary_join_key_matches_duckdb(duck):
    """A categorical join key is the star-schema case, and it drives the build-side choice."""
    left = _encoded(np.array([f"k{i % 50}" for i in range(_ROWS)]))
    right = pa.table(
        {
            "c": pa.array([f"k{i}" for i in range(50)]).dictionary_encode(),
            "label": pa.array([f"label-{i}" for i in range(50)]),
        }
    )
    duck.register("l", left)
    duck.register("r", right)
    got = bt.from_arrow(left).join(bt.from_arrow(right), on="c").collect()
    assert_same(got, duck.sql("SELECT l.c, l.g, r.label FROM l JOIN r USING (c)"))


def test_the_encoded_and_plain_forms_agree():
    """The invariant underneath all of it: encoding is a storage choice, not a semantics."""
    values = np.array([f"v{i % 97}" for i in range(_ROWS)])
    plain = pa.table({"c": pa.array(values)})
    encoded = pa.table({"c": pa.array(values).dictionary_encode()})
    for query in (
        lambda d: d.select(n=bt.col("c").n_unique()),
        lambda d: d.group_by("c").agg(n=bt.col("c").count()),
        lambda d: d.filter(bt.col("c") == "v3"),
    ):
        assert_tables_equal(
            query(bt.from_arrow(encoded)).collect(), query(bt.from_arrow(plain)).collect()
        )
