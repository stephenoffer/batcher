"""`AvroSource.iter_batches` streams, and agrees with `read()` byte for byte.

`_read_file` does `fh.read()` on the whole compressed file and decodes all of it, so peak
memory is the compressed size plus the larger decoded size — the constraint that caps how
big an Avro shard a worker can handle. `_iter_file` reads the container incrementally.

Doing so means `iter_batches` uses the **fastavro** decoder while `read` prefers the
**native** one, because only fastavro decodes incrementally. That is the risk this file
exists to contain: if the two ever disagree, the same source returns different data
depending on which terminal you called, and nothing else would catch it.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pyarrow as pa
import pytest

pytest.importorskip("fastavro")

from batcher.io.formats.structured.avro import AvroSink, AvroSource

pytestmark = pytest.mark.unit


def _roundtrip(tmp_path, table: pa.Table) -> AvroSource:
    AvroSink().write(table, str(tmp_path / "a.avro"))
    return AvroSource(str(tmp_path))


def _assert_paths_agree(src: AvroSource) -> pa.Table:
    read = pa.Table.from_batches(src.read())
    streamed = pa.Table.from_batches(list(src.iter_batches()))
    assert read.schema.equals(streamed.schema), f"{read.schema}\nvs\n{streamed.schema}"
    assert read.equals(streamed)
    return read


def test_read_and_iter_batches_agree_on_mixed_types(tmp_path) -> None:
    table = pa.table(
        {
            "i": list(range(5000)),
            "s": [f"v{i}" for i in range(5000)],
            "f": [i / 3 for i in range(5000)],
            "b": [i % 2 == 0 for i in range(5000)],
            "n": [None if i % 7 == 0 else i for i in range(5000)],
        }
    )
    assert _assert_paths_agree(_roundtrip(tmp_path, table)).num_rows == 5000


@pytest.mark.parametrize(
    ("name", "column"),
    [
        ("date32", pa.array([dt.date(2024, 1, 1 + (i % 28)) for i in range(100)], pa.date32())),
        (
            "timestamp_tz",
            pa.array(
                [dt.datetime(2024, 1, 1) + dt.timedelta(seconds=i) for i in range(100)],
                pa.timestamp("us", tz="+00:00"),
            ),
        ),
        (
            "timestamp_naive",
            pa.array(
                [dt.datetime(2024, 1, 1) + dt.timedelta(seconds=i) for i in range(100)],
                pa.timestamp("us"),
            ),
        ),
        ("time64", pa.array([dt.time(1, 2, 3), dt.time(4, 5, 6)], pa.time64("us"))),
        (
            "decimal",
            pa.array([Decimal("1.23"), Decimal("4.56")], pa.decimal128(9, 2)),
        ),
    ],
)
def test_logical_types_round_trip_with_their_type_intact(tmp_path, name, column) -> None:
    """Regression: the writer mapped every temporal/decimal type to Avro ``string``.

    `to_pylist()` yields `date`/`datetime`/`Decimal` objects for these columns, and
    fastavro refuses them against a string branch — so writing *any* such column raised,
    while the reader had always mapped the logical types back correctly. Assert both that
    the write succeeds and that the type survives, not merely that the values do.
    """
    table = pa.table({name: column})
    src = _roundtrip(tmp_path, table)
    result = _assert_paths_agree(src)

    assert result.schema.field(name).type == column.type
    assert result.column(name).to_pylist() == column.to_pylist()


def test_read_and_iter_batches_agree_when_every_value_is_null(tmp_path) -> None:
    table = pa.table({"n": pa.array([None] * 200, pa.int64())})
    _assert_paths_agree(_roundtrip(tmp_path, table))


def test_empty_file(tmp_path) -> None:
    table = pa.table({"i": pa.array([], pa.int64())})
    src = _roundtrip(tmp_path, table)
    assert sum(b.num_rows for b in src.iter_batches()) == 0


def test_streaming_honors_projection_and_its_order(tmp_path) -> None:
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.0, 2.0, 3.0]})
    src = _roundtrip(tmp_path, table)
    streamed = pa.Table.from_batches(list(src.iter_batches(["b", "a"])))

    assert streamed.schema.names == ["b", "a"]
    assert streamed.column("a").to_pylist() == [1, 2, 3]


def test_streaming_yields_more_than_one_batch(tmp_path) -> None:
    """If it returned a single batch it would not be streaming, and the test above
    would still pass — so assert the batching itself."""
    from batcher.config import active_config

    rows = active_config().execution.morsel_rows * 2 + 5
    table = pa.table({"i": list(range(rows))})
    src = _roundtrip(tmp_path, table)

    assert len(list(src.iter_batches())) >= 3
