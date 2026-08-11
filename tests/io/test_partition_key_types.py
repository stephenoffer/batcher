"""A date-valued Hive partition key must read back as a date.

``partition_by=["day"]`` on a date column is the most common Hive layout there is, and it
used to read back as a string, because pyarrow's discovery infers integers and nothing
else. That is not cosmetic: the obvious query against the layout,
``filter(col("day") == date(2024, 2, 10))``, died with ``Function 'equal' has no kernel
matching input types (string, date32[day])``. DuckDB returns DATE for the same tree.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.io


@pytest.fixture
def day_tree(tmp_path):
    out = str(tmp_path / "t")
    bt.from_pydict(
        {
            "day": [dt.date(2024, 1, 1), dt.date(2024, 2, 10), dt.date(2024, 12, 3)],
            "g": ["a", "b", "a"],
            "n": [1, 2, 3],
        }
    ).write.parquet(out, partition_by=["day", "g"])
    return out


def test_a_date_partition_key_round_trips_as_a_date(day_tree):
    """The write's own column type is what the read gives back."""
    back = bt.read.parquet(day_tree)
    assert back.schema.field("day").type == pa.date32()
    assert sorted(back.to_pydict()["day"]) == [
        dt.date(2024, 1, 1),
        dt.date(2024, 2, 10),
        dt.date(2024, 12, 3),
    ]


@pytest.mark.parametrize(
    ("predicate", "wanted"),
    [
        (bt.col("day") == dt.date(2024, 2, 10), [2]),
        (bt.col("day") > dt.date(2024, 1, 1), [2, 3]),
        (bt.col("day") <= dt.date(2024, 1, 1), [1]),
    ],
)
def test_the_obvious_date_query_works_without_a_cast(day_tree, predicate, wanted):
    assert sorted(bt.read.parquet(day_tree).filter(predicate).to_pydict()["n"]) == wanted


def test_a_non_date_key_beside_a_date_key_is_left_alone(day_tree):
    back = bt.read.parquet(day_tree)
    assert back.schema.field("g").type == pa.string()


def test_a_key_that_is_only_sometimes_a_date_stays_a_string(tmp_path):
    """Promotion is unanimous or not at all — half a tree parsing is worse than none."""
    out = str(tmp_path / "t")
    bt.from_pydict({"k": ["2024-01-01", "not-a-date"], "n": [1, 2]}).write.parquet(
        out, partition_by=["k"]
    )
    back = bt.read.parquet(out)
    assert back.schema.field("k").type == pa.string()
    assert sorted(back.to_pydict()["n"]) == [1, 2]


def test_a_null_date_partition_survives_the_promotion(tmp_path):
    """`__HIVE_DEFAULT_PARTITION__` must not veto the promotion nor be read as a date."""
    out = str(tmp_path / "t")
    bt.from_pydict({"day": [dt.date(2024, 1, 1), None], "n": [1, 2]}).write.parquet(
        out, partition_by=["day"]
    )
    back = bt.read.parquet(out)
    assert back.schema.field("day").type == pa.date32()
    assert sorted(back.to_pydict()["n"]) == [1, 2]
    assert back.filter(bt.col("day").is_null()).to_pydict()["n"] == [2]


@pytest.mark.parametrize("terminal", ["collect", "iter_batches", "count"])
def test_every_terminal_op_sees_the_same_rows(day_tree, terminal):
    back = bt.read.parquet(day_tree)
    if terminal == "count":
        assert back.count() == 3
    elif terminal == "collect":
        assert sorted(back.to_pydict()["n"]) == [1, 2, 3]
    else:
        rows = [v for b in back.iter_batches() for v in b.column("n").to_pylist()]
        assert sorted(rows) == [1, 2, 3]


def test_an_integer_key_keeps_the_type_discovery_already_gave_it(tmp_path):
    out = str(tmp_path / "t")
    bt.from_pydict({"i": [1, 2], "n": [10, 20]}).write.parquet(out, partition_by=["i"])
    back = bt.read.parquet(out)
    assert sorted(back.filter(bt.col("i") > 1).to_pydict()["n"]) == [20]


@pytest.mark.parametrize(
    ("predicate", "wanted"),
    [
        (bt.col("day") == "2024-02-10", [2]),
        (bt.col("day") > "2024-01-01", [2, 3]),
        (bt.col("day") <= "2024-01-01", [1]),
    ],
)
def test_a_string_literal_still_matches_a_date_key(day_tree, predicate, wanted):
    """The spelling that worked while the key was text must keep working now it is a date.

    This is the regression typing the key creates if nothing else changes: the predicate is
    *pushed down* to the pyarrow scanner, which has no ``equal(date32, string)`` kernel and
    raises rather than declining. The scan declines the term instead and lets the engine's
    own Filter answer it, which coerces the literal the way DuckDB does.
    """
    assert sorted(bt.read.parquet(day_tree).filter(predicate).to_pydict()["n"]) == wanted


def test_writing_into_a_partition_directory_makes_a_directory_of_parts(tmp_path):
    """``write.parquet("t/day=2024-01-01")`` is one partition of a table, as in Spark.

    Batcher wrote a *file* at that exact path, because the path carries no extension and a
    single-shard write goes straight to its destination. The tree was then unreadable by
    Batcher itself: ``read.parquet("t")`` found no ``.parquet`` files and raised.
    """
    root = str(tmp_path / "t")
    for day in ("2024-01-01", "2024-01-02"):
        bt.from_pydict({"n": [int(day[-2:])]}).write.parquet(f"{root}/day={day}")
    back = bt.read.parquet(root)
    assert sorted(back.to_pydict()["n"]) == [1, 2]
    assert back.schema.field("day").type == pa.date32()


def test_a_path_that_is_not_a_partition_directory_is_still_one_file(tmp_path):
    """The rule keys on ``col=value``, so an ordinary extensionless path is untouched."""
    out = str(tmp_path / "data")
    bt.from_pydict({"n": [4]}).write.parquet(out)
    assert (tmp_path / "data").is_file()


def test_single_file_wins_over_the_partition_directory_rule(tmp_path):
    out = str(tmp_path / "day=2024-05-05")
    bt.from_pydict({"n": [5]}).write.parquet(out, single_file=True)
    assert (tmp_path / "day=2024-05-05").is_file()
