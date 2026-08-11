"""A date-typed Hive partition key must answer every predicate the way DuckDB does.

Typing the key as a date is what makes these queries expressible at all, but it also puts
the column in the *pruning* path: the predicate is translated to a pyarrow dataset filter
that decides which directories are opened. A wrong translation there does not raise, it
silently returns too few rows, so the oracle has to see the answers rather than the plan.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

D = dt.date


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """28 daily partitions, one row each, written by Batcher's own partitioned writer."""
    out = str(tmp_path_factory.mktemp("dates") / "t")
    bt.from_pydict(
        {"day": [D(2024, 1, i) for i in range(1, 29)], "n": list(range(28))}
    ).write.parquet(out, partition_by=["day"])
    return out


@pytest.fixture(scope="module")
def oracle(tree):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t AS SELECT * FROM read_parquet(?, hive_partitioning=true)",
        [tree + "/**/*.parquet"],
    )
    return con


@pytest.mark.parametrize(
    ("predicate", "where"),
    [
        (bt.col("day") == D(2024, 1, 15), "day = DATE '2024-01-15'"),
        (bt.col("day") != D(2024, 1, 15), "day <> DATE '2024-01-15'"),
        (bt.col("day") > D(2024, 1, 20), "day > DATE '2024-01-20'"),
        (bt.col("day") >= D(2024, 1, 20), "day >= DATE '2024-01-20'"),
        (bt.col("day") < D(2024, 1, 5), "day < DATE '2024-01-05'"),
        (
            (bt.col("day") >= D(2024, 1, 10)) & (bt.col("day") <= D(2024, 1, 12)),
            "day BETWEEN DATE '2024-01-10' AND DATE '2024-01-12'",
        ),
        (
            (bt.col("day") == D(2024, 1, 1)) | (bt.col("day") == D(2024, 1, 28)),
            "day = DATE '2024-01-01' OR day = DATE '2024-01-28'",
        ),
        (
            bt.col("day").is_in([D(2024, 1, 3), D(2024, 1, 7)]),
            "day IN (DATE '2024-01-03', DATE '2024-01-07')",
        ),
        (bt.col("day").is_not_null(), "day IS NOT NULL"),
        # SQL's own spelling: a string literal against a date column. DuckDB coerces it,
        # and so must Batcher -- the scanner has no `equal(date32, string)` kernel, so the
        # term has to be declined from pushdown rather than pushed and raised on.
        (bt.col("day") == "2024-01-15", "day = '2024-01-15'"),
        (bt.col("day") > "2024-01-20", "day > '2024-01-20'"),
        # Prunes every directory: the empty answer must be empty, not the whole table.
        (bt.col("day") > D(2025, 1, 1), "day > DATE '2025-01-01'"),
        # A partition predicate AND a data predicate: one prunes, the other cannot.
        (
            (bt.col("day") > D(2024, 1, 20)) & (bt.col("n") < 25),
            "day > DATE '2024-01-20' AND n < 25",
        ),
    ],
)
def test_a_date_partition_predicate_matches_duckdb(tree, oracle, predicate, where):
    wanted = sorted(row[0] for row in oracle.execute(f"SELECT n FROM t WHERE {where}").fetchall())
    assert sorted(bt.read.parquet(tree).filter(predicate).to_pydict()["n"]) == wanted


def test_the_partition_column_itself_matches_duckdb(tree, oracle):
    """Not just the surviving rows — the key's own values and type have to agree."""
    wanted = sorted(row[0] for row in oracle.execute("SELECT day FROM t").fetchall())
    assert sorted(bt.read.parquet(tree).to_pydict()["day"]) == wanted


def test_grouping_by_the_date_key_matches_duckdb(tree, oracle):
    wanted = sorted(oracle.execute("SELECT day, count(*) FROM t GROUP BY day").fetchall())
    grouped = bt.read.parquet(tree).group_by("day").agg(c=bt.col("n").count()).to_pydict()
    assert sorted(zip(grouped["day"], grouped["c"], strict=True)) == wanted
