"""A join that prunes a partitioned fact table must return exactly what DuckDB returns.

Giving a Hive partition column min/max bounds unlocks `runtime_join_filter`: the join's
other side implies a `BETWEEN` on the partition key, which sinks to the scan and eliminates
whole directories before they are read. That is dynamic partition pruning, and its failure
mode is silent — a directory wrongly dropped returns *fewer rows*, not an error — so the
oracle has to see the answers rather than the plan.

The star-schema shape is the one the optimization exists for: a small dimension naming a few
days, joined to a fact table with a directory per day.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

D = dt.date


@pytest.fixture(scope="module")
def facts(tmp_path_factory):
    """28 daily partitions, three rows each, written by Batcher's own partitioned writer."""
    out = str(tmp_path_factory.mktemp("facts") / "t")
    days = [D(2024, 1, i) for i in range(1, 29) for _ in range(3)]
    bt.from_pydict({"day": days, "v": list(range(len(days)))}).write.parquet(
        out, partition_by=["day"]
    )
    # A directory whose rows were deleted. A rewrite leaves `dt=x` standing, so this is an
    # ordinary state for a real table — and the bounds read off directory names see it, which
    # makes every case below also a check that a widened bound changes no answer.
    os.makedirs(os.path.join(out, "day=2023-12-25"), exist_ok=True)
    return out


@pytest.fixture(scope="module")
def dim(tmp_path_factory):
    """A dimension naming three of the twenty-eight days, plus one the facts do not have."""
    out = str(tmp_path_factory.mktemp("dim") / "d.parquet")
    bt.from_pydict(
        {
            "day": [D(2024, 1, 5), D(2024, 1, 6), D(2024, 1, 20), D(2024, 2, 9)],
            "label": ["a", "b", "c", "absent"],
        }
    ).write.parquet(out)
    return out


@pytest.fixture(scope="module")
def oracle(facts, dim):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE f AS SELECT * FROM read_parquet(?, hive_partitioning=true)",
        [facts + "/**/*.parquet"],
    )
    con.execute("CREATE TABLE d AS SELECT * FROM read_parquet(?)", [dim])
    return con


@pytest.mark.parametrize(
    ("how", "sql"),
    [
        ("inner", "SELECT f.day, f.v FROM f JOIN d ON f.day = d.day"),
        ("left", "SELECT f.day, f.v FROM f LEFT JOIN d ON f.day = d.day"),
        ("semi", "SELECT f.day, f.v FROM f WHERE f.day IN (SELECT day FROM d)"),
        ("anti", "SELECT f.day, f.v FROM f WHERE f.day NOT IN (SELECT day FROM d)"),
    ],
)
def test_a_join_against_a_partitioned_fact_matches_duckdb(facts, dim, oracle, how, sql):
    """Every join type, including the ones whose prunable side is not the fact table."""
    got = (
        bt.read.parquet(facts)
        .join(bt.read.parquet(dim), on="day", how=how)
        .select("day", "v")
        .collect()
    )
    assert_same(got, oracle.sql(sql))


def test_the_dimension_side_is_also_reduced(facts, dim, oracle):
    """The fact table's own range implies a filter on the dimension — the mirror case.

    The dimension names 2024-02-09, which no partition holds. Dropping it via a range filter
    derived from the fact table's partition bounds has to agree with dropping it by matching.
    """
    got = (
        bt.read.parquet(dim)
        .join(bt.read.parquet(facts), on="day", how="inner")
        .select("day", "label")
        .distinct()
        .collect()
    )
    assert_same(got, oracle.sql("SELECT DISTINCT d.day, d.label FROM d JOIN f ON d.day = f.day"))


def test_a_join_that_prunes_everything_returns_nothing(facts, oracle, tmp_path):
    """A dimension outside every partition must yield an empty result, not the whole table."""
    empty_dim = str(tmp_path / "far.parquet")
    bt.from_pydict({"day": [D(2030, 1, 1)], "label": ["x"]}).write.parquet(empty_dim)
    oracle.execute("CREATE OR REPLACE TABLE far AS SELECT * FROM read_parquet(?)", [empty_dim])
    got = (
        bt.read.parquet(facts)
        .join(bt.read.parquet(empty_dim), on="day", how="inner")
        .select("day", "v")
        .collect()
    )
    assert_same(got, oracle.sql("SELECT f.day, f.v FROM f JOIN far ON f.day = far.day"))
