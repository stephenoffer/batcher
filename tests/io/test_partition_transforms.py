"""Partition transforms on write, and partition-column recovery on read.

Two halves of the same round trip. Writing accepts an *expression* partition key, which
is how Iceberg's ``days(ts)`` / ``bucket(16, id)`` and Spark's generated partition column
are spelled here: the expression is evaluated once and its alias becomes the directory
name. Reading a directory laid out that way now routes to the partition-aware reader, so
the column the data is organized by comes back instead of silently vanishing -- a lossy
round trip through Batcher's own writer, which every engine users arrive from gets right.
"""

from __future__ import annotations

import glob
import os

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration


def _dirs(root: str) -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(f"{root}/*") if os.path.isdir(p))


def _events() -> bt.Dataset:
    return bt.from_pydict(
        {
            "ts": ["2024-01-05", "2024-02-09", "2025-03-01"],
            "id": [7, 21, 33],
            "v": [1, 2, 3],
        }
    ).with_columns(ts=bt.col("ts").cast("date"))


def test_an_expression_key_writes_its_alias_as_the_directory(tmp_path):
    out = str(tmp_path / "events")
    _events().write.parquet(out, partition_by=[bt.col("ts").dt.year().alias("year")])
    assert _dirs(out) == ["year=2024", "year=2025"]


def test_the_derived_column_is_in_the_path_not_the_data(tmp_path):
    import pyarrow.parquet as pq

    out = str(tmp_path / "events")
    _events().write.parquet(out, partition_by=[bt.col("ts").dt.year().alias("year")])
    for f in glob.glob(f"{out}/*/*.parquet"):
        assert "year" not in pq.read_schema(f).names


def test_the_derived_column_comes_back_on_read(tmp_path):
    out = str(tmp_path / "events")
    _events().write.parquet(out, partition_by=[bt.col("ts").dt.year().alias("year")])
    back = bt.read.parquet(out)
    assert "year" in back.columns
    assert sorted(back.select("year").to_pydict()["year"]) == [2024, 2024, 2025]


def test_a_filter_on_the_derived_key_prunes_and_still_answers(tmp_path):
    out = str(tmp_path / "events")
    _events().write.parquet(out, partition_by=[bt.col("ts").dt.year().alias("year")])
    got = bt.read.parquet(out).filter(bt.col("year") == 2024).sort("v").to_pydict()
    assert got["v"] == [1, 2]


def test_a_bucketing_transform_spreads_keys_over_fixed_directories(tmp_path):
    # Iceberg's `bucket(N, col)`, spelled as the modulo it is.
    out = str(tmp_path / "bucketed")
    _events().write.parquet(out, partition_by=[(bt.col("id") % 4).alias("bucket")])
    assert _dirs(out) == ["bucket=1", "bucket=3"]
    assert bt.read.parquet(out).count() == 3


def test_a_name_and_an_expression_can_be_mixed(tmp_path):
    out = str(tmp_path / "mixed")
    ds = _events().with_columns(region=bt.lit("eu"))
    ds.write.parquet(out, partition_by=["region", bt.col("ts").dt.year().alias("year")])
    assert _dirs(out) == ["region=eu"]
    assert sorted(_dirs(f"{out}/region=eu")) == ["year=2024", "year=2025"]
    back = bt.read.parquet(out)
    assert {"region", "year"} <= set(back.columns)


def test_an_unnamed_expression_is_refused_with_the_fix(tmp_path):
    out = str(tmp_path / "unnamed")
    with pytest.raises(PlanError, match="alias"):
        _events().write.parquet(out, partition_by=[bt.col("ts").dt.year()])


def test_a_bare_column_expression_partitions_by_its_own_name(tmp_path):
    out = str(tmp_path / "bare")
    _events().with_columns(region=bt.lit("eu")).write.parquet(out, partition_by=[bt.col("region")])
    assert _dirs(out) == ["region=eu"]


def test_the_spark_spelling_takes_an_expression_too(tmp_path):
    out = str(tmp_path / "spark")
    _events().write.parquet(out, partitionBy=[bt.col("ts").dt.year().alias("year")])
    assert _dirs(out) == ["year=2024", "year=2025"]


def test_a_plain_partition_column_also_survives_the_round_trip(tmp_path):
    # The bug this routing fixes: `partition_by=["g"]` wrote `g` into the path and the
    # flat reader read it back without `g` at all, losing a column through Batcher's own
    # writer while reporting every row.
    out = str(tmp_path / "plain")
    bt.from_pydict({"g": ["a", "b", "a"], "v": [1, 2, 3]}).write.parquet(out, partition_by=["g"])
    back = bt.read.parquet(out).sort("v")
    assert sorted(back.columns) == ["g", "v"]
    assert back.to_pydict() == {"v": [1, 2, 3], "g": ["a", "b", "a"]}


def test_a_flat_directory_of_parts_keeps_the_flat_reader(tmp_path):
    out = str(tmp_path / "flat")
    bt.from_pydict({"v": list(range(10))}).write.parquet(out, max_rows_per_file=4)
    assert len(glob.glob(f"{out}/*.parquet")) == 3
    assert sorted(bt.read.parquet(out).to_pydict()["v"]) == list(range(10))


def test_a_reader_option_the_partitioned_source_cannot_honor_keeps_the_flat_reader(tmp_path):
    # `n_rows` is a whole-source cap the partitioned reader does not take. Swapping the
    # reader in anyway would trade a missing column for a silently ignored option.
    out = str(tmp_path / "capped")
    bt.from_pydict({"g": ["a", "b", "a"], "v": [1, 2, 3]}).write.parquet(out, partition_by=["g"])
    assert bt.read.parquet(out, n_rows=2).count() == 2


def test_a_single_file_read_is_untouched(tmp_path):
    path = str(tmp_path / "one.parquet")
    bt.from_pydict({"x": [1, 2]}).write.parquet(path)
    assert bt.read.parquet(path).to_pydict() == {"x": [1, 2]}


def test_a_path_carrying_connection_options_keeps_the_flat_reader(tmp_path):
    # `?endpoint_override=` is how an on-prem S3 is addressed. The partitioned source
    # builds its dataset straight from the path and would not understand it, so the
    # upgrade must decline rather than turn a working read into a connection failure.
    from batcher.io.detect import partition_aware_format

    out = str(tmp_path / "q")
    bt.from_pydict({"g": ["a"], "v": [1]}).write.parquet(out, partition_by=["g"])
    assert partition_aware_format(out, "parquet", {}) == "parquet_dataset"
    assert partition_aware_format(f"{out}?endpoint_override=x", "parquet", {}) == "parquet"


def test_a_glob_is_not_a_directory_to_inspect(tmp_path):
    from batcher.io.detect import partition_aware_format

    out = str(tmp_path / "g")
    bt.from_pydict({"g": ["a"], "v": [1]}).write.parquet(out, partition_by=["g"])
    assert partition_aware_format(f"{out}/*/*.parquet", "parquet", {}) == "parquet"


def test_a_non_parquet_format_is_never_upgraded(tmp_path):
    from batcher.io.detect import partition_aware_format

    out = str(tmp_path / "c")
    bt.from_pydict({"g": ["a"], "v": [1]}).write.csv(out, partition_by=["g"])
    assert partition_aware_format(out, "csv", {}) == "csv"


def test_an_empty_partitioned_write_still_leaves_a_readable_relation(tmp_path):
    """A filter that matched nothing is not an error.

    With no rows there are no partition values, so there are no ``col=v`` directories to
    write -- and writing nothing left the destination absent, so the next read failed with
    "path does not exist" where every other write shape returns no rows.
    """
    out = str(tmp_path / "empty")
    manifest = (
        bt.from_pydict({"g": ["a"], "v": [1]})
        .filter(bt.col("v") > 99)
        .write.parquet(out, partition_by=["g"])
    )
    assert manifest.num_files == 1
    back = bt.read.parquet(out)
    assert back.count() == 0
    # The columns survive too: nothing was moved into a path, so none was dropped.
    assert sorted(back.columns) == ["g", "v"]


def test_a_non_empty_partitioned_write_is_unchanged(tmp_path):
    out = str(tmp_path / "full")
    bt.from_pydict({"g": ["a", "b"], "v": [1, 2]}).write.parquet(out, partition_by=["g"])
    assert _dirs(out) == ["g=a", "g=b"]
    assert sorted(bt.read.parquet(out).columns) == ["g", "v"]


@pytest.mark.parametrize(
    ("values", "recovered"),
    [
        ([1, 2, 1], [1, 2, 1]),  # integers survive
        (["a", "b", "a"], ["a", "b", "a"]),  # strings survive
    ],
)
def test_a_partition_key_of_a_recoverable_type_round_trips(tmp_path, values, recovered):
    out = str(tmp_path / f"rt{values[0]}")
    bt.from_pydict({"g": values, "v": [1, 2, 3]}).write.parquet(out, partition_by=["g"])
    assert bt.read.parquet(out).sort("v").to_pydict()["g"] == recovered


def test_a_date_partition_key_comes_back_as_a_date(tmp_path):
    """A Hive path segment is text, but a date is recoverable from it and now is.

    This test previously pinned the opposite, as a visible limitation. It was the wrong
    thing to pin: the values are unambiguous, DuckDB recovers them, and leaving them as
    text made ``filter(col("dt") == date(...))`` fail with an Arrow kernel error on the
    single most common Hive layout. See `test_partition_key_types.py` for the full rules.
    """
    import datetime as dt

    out = str(tmp_path / "dates")
    days = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 1)]
    bt.from_pydict({"dt": days, "v": [1, 2, 3]}).write.parquet(out, partition_by=["dt"])
    assert bt.read.parquet(out).sort("v").to_pydict()["dt"] == days


def test_a_partition_key_whose_type_the_path_cannot_carry_comes_back_as_text(tmp_path):
    """The limitation that remains, pinned so it is visible rather than discovered.

    A float is not recoverable the way a date is: ``1.5`` and ``"1.5"`` produce the same
    directory name and nothing distinguishes them, so promoting it would be guessing. A
    caller who needs the original type casts it, and a table whose partition types must
    survive exactly wants Delta or Iceberg, which record the partition schema in the log.
    """
    out = str(tmp_path / "floats")
    bt.from_pydict({"r": [1.5, 2.5, 1.5], "v": [1, 2, 3]}).write.parquet(out, partition_by=["r"])
    assert bt.read.parquet(out).sort("v").to_pydict()["r"] == ["1.5", "2.5", "1.5"]
    typed = bt.read.parquet(out).with_columns(r=bt.col("r").cast("float64")).sort("v")
    assert typed.to_pydict()["r"] == [1.5, 2.5, 1.5]


def test_a_partition_value_with_reserved_characters_round_trips_exactly(tmp_path):
    # `/` would spawn a subdirectory and `=` would look like another key, so the writer
    # URL-encodes and the reader decodes. A value that came back mangled would be a wrong
    # answer, not a formatting nit.
    out = str(tmp_path / "chars")
    values = ["a/b", "c d", "e=f", "g%h", "i+j"]
    bt.from_pydict({"g": values, "v": [1, 2, 3, 4, 5]}).write.parquet(out, partition_by=["g"])
    assert bt.read.parquet(out).sort("v").to_pydict()["g"] == values


def test_a_null_partition_value_round_trips_as_null(tmp_path):
    out = str(tmp_path / "nulls")
    bt.from_pydict({"g": ["a", None, "b"], "v": [1, 2, 3]}).write.parquet(out, partition_by=["g"])
    assert bt.read.parquet(out).sort("v").to_pydict()["g"] == ["a", None, "b"]


def test_a_partition_value_equal_to_the_null_sentinel_is_refused(tmp_path):
    """The one value a Hive layout cannot represent, refused instead of corrupted.

    ``__HIVE_DEFAULT_PARTITION__`` is how a directory name spells NULL, and every reader
    (pyarrow's included) decodes the segment *before* comparing it, so no escaping
    survives -- the row would come back with a null partition key and nothing downstream
    could tell. Hive and Spark have the identical hole and fill it silently.
    """
    out = str(tmp_path / "sentinel")
    ds = bt.from_pydict({"g": ["__HIVE_DEFAULT_PARTITION__", "a"], "v": [1, 2]})
    with pytest.raises(PlanError, match="HIVE_DEFAULT_PARTITION"):
        ds.write.parquet(out, partition_by=["g"])


def test_a_real_null_partition_value_is_still_written(tmp_path):
    # The sentinel keeps doing its job; only a *value* that collides with it is refused.
    out = str(tmp_path / "realnull")
    bt.from_pydict({"g": ["a", None], "v": [1, 2]}).write.parquet(out, partition_by=["g"])
    assert bt.read.parquet(out).sort("v").to_pydict()["g"] == ["a", None]


def test_an_empty_string_partition_value_round_trips(tmp_path):
    out = str(tmp_path / "emptystr")
    bt.from_pydict({"g": ["", "a"], "v": [1, 2]}).write.parquet(out, partition_by=["g"])
    assert bt.read.parquet(out).sort("v").to_pydict()["g"] == ["", "a"]
