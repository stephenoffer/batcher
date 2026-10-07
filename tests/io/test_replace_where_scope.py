"""``write(replace_where=...)`` replaces exactly the rows its predicate is TRUE on.

Two ways it used to do otherwise, both silent. The copy-on-write path (a plain Parquet
table) kept ``filter(~predicate)``, and on a row where the predicate is NULL the negation is
NULL too, so the backfill deleted rows it never claimed. The Delta partition-scoped path
retired the named partitions but committed every file the write produced, so an incoming row
outside the predicate was appended beside the rows already in its partition.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

pytestmark = pytest.mark.io

bt = pytest.importorskip("batcher")


def _sorted(path: str, fmt: str) -> list[tuple]:
    got = getattr(bt.read, fmt)(path).collect().to_pydict()
    return sorted(zip(got["r"], got["v"], strict=True), key=lambda t: (str(t[0]), t[1]))


def test_a_null_predicate_row_survives_a_copy_on_write_backfill(tmp_path) -> None:
    path = str(tmp_path / "t")
    bt.from_pydict({"r": ["us", None, "eu"], "v": [1, 2, 3]}).write.parquet(path)
    bt.from_pydict({"r": ["us"], "v": [10]}).write.parquet(
        path, mode="overwrite", replace_where=bt.col("r") == "us"
    )
    assert _sorted(path, "parquet") == [(None, 2), ("eu", 3), ("us", 10)]


def _delta(path: str) -> None:
    pytest.importorskip("deltalake")
    bt.from_pydict({"r": ["us", "eu"], "v": [1, 3]}).write.delta(path, partition_by=["r"])


def test_a_delta_backfill_with_rows_outside_the_predicate_is_refused(tmp_path) -> None:
    from batcher._internal.errors import CommitError

    path = str(tmp_path / "t")
    _delta(path)
    with pytest.raises(CommitError, match="does not cover"):
        bt.from_pydict({"r": ["us", "eu"], "v": [10, 30]}).write.delta(
            path, mode="overwrite", replace_where=bt.col("r") == "us"
        )
    # Refused before the commit, so the table is untouched rather than half-replaced.
    assert _sorted(path, "delta") == [("eu", 3), ("us", 1)]


def test_an_or_of_partitions_is_checked_too(tmp_path) -> None:
    from batcher._internal.errors import CommitError

    path = str(tmp_path / "t")
    pytest.importorskip("deltalake")
    bt.from_pydict({"r": ["us", "eu", "ap"], "v": [1, 3, 5]}).write.delta(path, partition_by=["r"])
    predicate = (bt.col("r") == "us") | (bt.col("r") == "eu")
    with pytest.raises(CommitError, match="does not cover"):
        bt.from_pydict({"r": ["us", "ap"], "v": [10, 50]}).write.delta(
            path, mode="overwrite", replace_where=predicate
        )
    bt.from_pydict({"r": ["us", "eu"], "v": [10, 30]}).write.delta(
        path, mode="overwrite", replace_where=predicate
    )
    assert _sorted(path, "delta") == [("ap", 5), ("eu", 30), ("us", 10)]


def test_an_in_scope_typed_backfill_still_commits(tmp_path) -> None:
    """The check compares typed values, so an integer partition is not refused as text."""
    pytest.importorskip("deltalake")
    path = str(tmp_path / "t")
    bt.from_arrow(pa.table({"r": ["us", "us"], "k": [1, 2], "v": [1, 2]})).write.delta(
        path, partition_by=["r", "k"]
    )
    bt.from_arrow(pa.table({"r": ["us"], "k": [1], "v": [100]})).write.delta(
        path, mode="overwrite", replace_where=(bt.col("r") == "us") & (bt.col("k") == 1)
    )
    got = bt.read.delta(path).collect().to_pydict()
    assert sorted(zip(got["k"], got["v"], strict=True)) == [(1, 100), (2, 2)]


def test_a_date_partition_backfill_commits(tmp_path) -> None:
    """A date literal reaches delta-rs as ``2024-01-05``, not as its day count ``19727``."""
    pytest.importorskip("deltalake")
    path = str(tmp_path / "t")
    day, other, third = dt.date(2024, 1, 5), dt.date(2024, 1, 6), dt.date(2024, 1, 7)
    bt.from_arrow(pa.table({"d": [day, other, third], "v": [1, 2, 3]})).write.delta(
        path, partition_by=["d"]
    )
    bt.from_arrow(pa.table({"d": [day], "v": [100]})).write.delta(
        path, mode="overwrite", replace_where=bt.col("d") == day
    )
    # An OR of dates takes the explicit-removal path, which matches the same text form.
    bt.from_arrow(pa.table({"d": [other, third], "v": [200, 300]})).write.delta(
        path, mode="overwrite", replace_where=(bt.col("d") == other) | (bt.col("d") == third)
    )
    got = bt.read.delta(path).collect().to_pydict()
    assert sorted(zip(got["d"], got["v"], strict=True)) == [
        (day, 100),
        (other, 200),
        (third, 300),
    ]
