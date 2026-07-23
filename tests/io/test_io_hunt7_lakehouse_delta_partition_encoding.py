"""Delta partition values with URL-special characters must read back correctly.

A Delta ``add.path`` is a URI: the protocol stores it URL-encoded and a reader must
*decode* it to get the physical data-file path. A partition value that contains a
``/``, a space, or a ``%`` is therefore written to a directory encoded once
(``a/b`` -> ``p=a%2Fb``) and recorded in the log encoded again (``p=a%252Fb``).

Batcher's manifest-driven reader used the raw log path as a filesystem path, so it
looked for ``p=a%252Fb`` — which does not exist — and raised ``FileNotFoundError`` on
every partitioned Delta table whose partition value held such a character (a space is
extremely common: ``"New York"``). deltalake's own reader, which decodes, read the same
table fine. This pins the decode.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("deltalake", reason="deltalake not installed")

from deltalake import DeltaTable, write_deltalake


def _pydict(path: str, version: int | None = None) -> dict:
    ds = bt.read.delta(path, version=version) if version is not None else bt.read.delta(path)
    return ds.sort("id").to_pydict()


def test_special_char_partition_values_round_trip(tmp_path) -> None:
    """``/``, space, and ``%`` in a partition value read back as written."""
    tbl = str(tmp_path / "t")
    data = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "p": ["a/b", None, "New York", "a/b", "100%"],
            "v": [10, 20, 30, 40, 50],
        }
    )
    bt.from_arrow(data).write.delta(tbl, mode="append", partition_by=["p"])

    got = _pydict(tbl)
    ref = DeltaTable(tbl).to_pyarrow_table().sort_by("id").to_pydict()
    assert got == ref
    assert got["p"] == ["a/b", None, "New York", "a/b", "100%"]


def test_special_char_partition_written_by_delta_rs(tmp_path) -> None:
    """A table delta-rs itself wrote (double-encoded log paths) reads correctly too."""
    tbl = str(tmp_path / "t")
    data = pa.table({"id": [1, 2, 3], "p": ["a/b", "New York", "100%"], "v": [10, 20, 30]})
    write_deltalake(tbl, data, partition_by=["p"], mode="append")

    got = _pydict(tbl)
    ref = DeltaTable(tbl).to_pyarrow_table().sort_by("id").to_pydict()
    assert got == ref


def test_null_partition_rows_survive_a_filtered_read(tmp_path) -> None:
    """A predicate on a data column must not drop the null-partition file's rows.

    The write recorded that file's statistics by masking the shard with ``col == value``;
    for the null partition that mask is all-NULL, so the file was indexed as holding 0
    rows with no column bounds. A filtered read then pruned it and its rows vanished:
    ``filter(v > 15)`` returned ``[3, 4]`` instead of ``[2, 3, 4]`` and ``count()``
    returned 3 for a 4-row table.
    """
    from batcher import col

    tbl = str(tmp_path / "t")
    data = pa.table({"id": [1, 2, 3, 4], "p": ["a", None, "a", "b"], "v": [10, 20, 30, 40]})
    bt.from_arrow(data).write.delta(tbl, mode="append", partition_by=["p"])

    assert bt.read.delta(tbl).count() == 4
    assert bt.read.delta(tbl).filter(col("v") > 15).sort("id").to_pydict()["id"] == [2, 3, 4]
    assert bt.read.delta(tbl).filter(col("v") == 20).sort("id").to_pydict()["id"] == [2]


def test_integer_null_partition_stats_are_exact(tmp_path) -> None:
    """The same null-partition stats fix holds for a non-string partition column."""
    from batcher import col

    tbl = str(tmp_path / "t")
    data = pa.table({"id": [1, 2, 3], "k": pa.array([5, None, 7], pa.int64()), "v": [10, 20, 30]})
    bt.from_arrow(data).write.delta(tbl, mode="append", partition_by=["k"])

    assert bt.read.delta(tbl).count() == 3
    assert bt.read.delta(tbl).filter(col("v") >= 20).sort("id").to_pydict()["id"] == [2, 3]


def test_special_char_partition_predicate_and_count(tmp_path) -> None:
    """A predicate on the special-char partition column, and ``count()``, are exact."""
    tbl = str(tmp_path / "t")
    data = pa.table({"id": [1, 2, 3, 4], "p": ["a/b", None, "a/b", "x y"], "v": [10, 20, 30, 40]})
    bt.from_arrow(data).write.delta(tbl, mode="append", partition_by=["p"])

    from batcher import col

    hit = bt.read.delta(tbl).filter(col("p") == "a/b").sort("id").to_pydict()
    assert hit == {"id": [1, 3], "p": ["a/b", "a/b"], "v": [10, 30]}
    assert bt.read.delta(tbl).count() == 4
