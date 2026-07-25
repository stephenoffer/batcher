"""A write to a partitioned Delta table must keep it partitioned, without being told again.

Delta keeps a partition value in the *directory name*, not in the data file. So a write that
lays its file out flat does not merely organise the table differently — the value is nowhere,
and the row reads back null in the column the table is organised by. The row survives, the
value is gone, and nothing raises.

That is what happened whenever `partition_by=` was omitted, which is the ordinary way to
append to a table that was partitioned when it was created:

    ds.write.delta(path, partition_by=["region"])        # region=us/, region=eu/
    more.write.delta(path, mode="append")                # part-00000.parquet at the root
    bt.read.delta(path)                                  # -> the new row has region = None

The layout is a property of the *table*, so the sink resolves it from the table's own log when
the call does not restate it. `terminal.core` already had the hook for this — `partitions_itself`,
added when a partitioned *Iceberg* table turned out to be unwritable for the same reason — and
this uses its path-aware form, because whether a Delta table is partitioned depends on how it
was created and only the table can say.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.io

bt = pytest.importorskip("batcher")
pytest.importorskip("deltalake")

BASE = pa.table(
    {
        "id": pa.array([1, 2, 3], pa.int64()),
        "region": pa.array(["us", "eu", "eu"]),
        "amount": pa.array([10.0, 20.0, 30.0], pa.float64()),
    }
)


def _rows(path):
    got = bt.read.delta(str(path)).collect().to_pydict()
    return sorted(zip(got["id"], got["region"], strict=True))


def _partitioned(path):
    bt.from_arrow(BASE).write.delta(str(path), partition_by=["region"])
    return path


def _new(ids, regions):
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "region": pa.array(regions),
            "amount": pa.array([0.0] * len(ids), pa.float64()),
        }
    )


@pytest.mark.parametrize(
    ("mode", "kwargs", "expected"),
    [
        ("append", {}, [(1, "us"), (2, "eu"), (3, "eu"), (99, "us")]),
        ("append", {"partition_by": ["region"]}, [(1, "us"), (2, "eu"), (3, "eu"), (99, "us")]),
        ("overwrite", {}, [(99, "us")]),
    ],
    ids=["append", "append-restated", "overwrite"],
)
def test_a_write_without_partition_by_keeps_the_partition_values(tmp_path, mode, kwargs, expected):
    path = _partitioned(tmp_path / "t")
    bt.from_arrow(_new([99], ["us"])).write.delta(str(path), mode=mode, **kwargs)
    assert _rows(path) == expected


def test_a_shard_spanning_several_partitions_keeps_every_value(tmp_path):
    """`write` returns one file, so a multi-partition shard has to take the fan-out path."""
    path = _partitioned(tmp_path / "t")
    bt.from_arrow(_new([88, 89], ["us", "ap"])).write.delta(str(path), mode="append")
    assert _rows(path) == [(1, "us"), (2, "eu"), (3, "eu"), (88, "us"), (89, "ap")]


def test_a_multi_key_partitioning_is_inherited_too(tmp_path):
    path = tmp_path / "m"
    first = pa.table(
        {"id": pa.array([1, 2], pa.int64()), "a": pa.array(["x", "y"]), "b": pa.array(["p", "q"])}
    )
    bt.from_arrow(first).write.delta(str(path), partition_by=["a", "b"])
    later = pa.table({"id": pa.array([3], pa.int64()), "a": pa.array(["z"]), "b": pa.array(["r"])})
    bt.from_arrow(later).write.delta(str(path), mode="append")
    got = bt.read.delta(str(path)).collect().to_pydict()
    assert sorted(zip(got["id"], got["a"], got["b"], strict=True)) == [
        (1, "x", "p"),
        (2, "y", "q"),
        (3, "z", "r"),
    ]


def test_replace_where_scopes_to_the_tables_own_partition_columns(tmp_path):
    """A backfill targets an existing table, so restating `partition_by` is redundant.

    The scoping check read only *this call's* `partition_by`, so omitting it made every such
    write raise — and the error then advised partitioning the table on the backfill columns,
    which it already was.
    """
    path = _partitioned(tmp_path / "t")
    bt.from_arrow(_new([99], ["us"])).write.delta(
        str(path), mode="overwrite", replace_where=bt.col("region") == "us"
    )
    assert _rows(path) == [(2, "eu"), (3, "eu"), (99, "us")]


def test_an_unpartitioned_table_is_left_unpartitioned(tmp_path):
    """Inheriting must not invent partitioning where the table has none."""
    path = tmp_path / "flat"
    bt.from_arrow(BASE).write.delta(str(path))
    bt.from_arrow(_new([99], ["us"])).write.delta(str(path), mode="append")
    assert _rows(path) == [(1, "us"), (2, "eu"), (3, "eu"), (99, "us")]
    assert not [d for d in path.iterdir() if d.is_dir() and d.name.startswith("region=")]


def test_replace_where_still_refuses_what_it_cannot_scope(tmp_path):
    """Widening a scoped overwrite into a full one is the worst failure here, so it raises."""
    from batcher._internal.errors import CommitError

    path = _partitioned(tmp_path / "t")
    with pytest.raises(CommitError):  # `amount` is not a partition column
        bt.from_arrow(_new([99], ["us"])).write.delta(
            str(path), mode="overwrite", replace_where=bt.col("amount") > 5
        )
    flat = tmp_path / "flat"
    bt.from_arrow(BASE).write.delta(str(flat))
    with pytest.raises(CommitError):  # the table has no partitions to scope to
        bt.from_arrow(_new([99], ["us"])).write.delta(
            str(flat), mode="overwrite", replace_where=bt.col("region") == "us"
        )
