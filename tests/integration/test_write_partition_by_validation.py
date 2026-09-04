"""`partition_by` names a column that must exist, and says so like the rest of the engine.

An unknown partition column used to reach Arrow's sort kernel, which reported it as a bare
`ArrowInvalid` reading ``No match for FieldRef.Name(zz) in a: int64 ...`` followed by a dump
of the table's schema **and its data** into the exception message. Three things wrong with
that: it is untyped, so `except ColumnNotFoundError` (which every other unknown-column miss
in the engine raises) did not catch it; it is unreadable; and it puts row values into an
exception string, which is how data reaches a log.

The check runs at the sink, which is the one choke point every writer passes through --
single-node and distributed alike.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError

pytestmark = pytest.mark.integration


def test_unknown_partition_column_raises_column_not_found(tmp_path):
    ds = bt.from_pydict({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(ColumnNotFoundError) as excinfo:
        ds.write.parquet(str(tmp_path / "out"), partition_by=["zz"])
    assert excinfo.value.column == "zz"
    assert "zz" in str(excinfo.value)


def test_error_suggests_the_near_miss(tmp_path):
    ds = bt.from_pydict({"a": [1], "bb": [2]})
    with pytest.raises(ColumnNotFoundError, match="Did you mean 'bb'"):
        ds.write.parquet(str(tmp_path / "out"), partition_by=["b"])


def test_error_does_not_dump_row_values(tmp_path):
    """The old ArrowInvalid embedded the table's data; an error message is not a dump."""
    ds = bt.from_pydict({"a": [111111, 222222], "b": ["secret-value", "other"]})
    with pytest.raises(ColumnNotFoundError) as excinfo:
        ds.write.parquet(str(tmp_path / "out"), partition_by=["zz"])
    message = str(excinfo.value)
    assert "secret-value" not in message
    assert "111111" not in message


def test_is_a_keyerror_too(tmp_path):
    """`ColumnNotFoundError` is also a `KeyError`; looking a column up is a mapping lookup."""
    ds = bt.from_pydict({"a": [1]})
    with pytest.raises(KeyError):
        ds.write.parquet(str(tmp_path / "out"), partition_by=["nope"])


def test_valid_partition_by_still_writes(tmp_path):
    """The guard must not disturb the working path, nulls included."""
    out = str(tmp_path / "out")
    ds = bt.from_pydict({"a": [None, 1, None, 2], "b": [10, 20, 30, 40]})
    manifest = ds.write.parquet(out, partition_by=["a"])
    assert sum(f.rows for f in manifest.files) == 4
    back = bt.read.parquet(out).collect().to_pydict()
    assert sorted(back["b"]) == [10, 20, 30, 40]
