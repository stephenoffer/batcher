"""Regression: a pushed predicate on a timezone-aware Delta timestamp column.

Lakehouse timestamp columns are UTC-normalized — a Delta ``timestamp`` is stored with a
timezone — so an event-time table's time column is almost always ``timestamp[us, tz=UTC]``.
When Kyber pushed a ``WHERE ts > …`` predicate to such a table, the reader built a tz-naive
``timestamp[us]`` literal and pyarrow refused the comparison outright::

    ArrowInvalid: Cannot compare timestamp with timezone to timestamp without timezone

so the read crashed instead of pruning — including the worker-side split read and the
fused ``count(*)`` path, where the source filter is the only one applied. The fix types a
temporal literal to its column (`io.predicate.to_pyarrow_expression` takes the schema), so
the literal's UTC instant is expressed in the column's own unit and zone.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from batcher.io.formats.lakehouse.delta.sink import DeltaSink
from batcher.io.formats.lakehouse.delta.source import DeltaSource
from batcher.io.manifest import WriteManifest

pytestmark = pytest.mark.io

pytest.importorskip("deltalake")


def _append(table: pa.Table, path: str, idx: int) -> None:
    sink = DeltaSink(mode="append")
    files = sink.write_partitioned(table, path, file_index=idx)
    sink.commit(WriteManifest(tuple(files), schema=table.schema), path)


def _ts_predicate(column: str, boundary: dt.datetime) -> dict:
    micros = int((boundary - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)).total_seconds() * 1e6)
    return {
        "e": "binary",
        "op": "gt",
        "left": {"e": "col", "name": column},
        "right": {"e": "lit", "value": {"timestamp": micros}},
    }


def test_pushed_predicate_on_utc_timestamp_column(tmp_path) -> None:
    """A ``ts > x`` predicate on a tz-aware column reads (does not crash) and prunes right."""
    path = str(tmp_path / "events")
    base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    ts_type = pa.timestamp("us", tz="UTC")
    _append(
        pa.table({"id": [1, 2], "t": pa.array([base, base + dt.timedelta(days=1)], type=ts_type)}),
        path,
        0,
    )
    _append(
        pa.table(
            {
                "id": [3, 4],
                "t": pa.array(
                    [base + dt.timedelta(days=10), base + dt.timedelta(days=11)], type=ts_type
                ),
            }
        ),
        path,
        1,
    )

    boundary = dt.datetime(2024, 1, 5, tzinfo=dt.timezone.utc)
    predicate = _ts_predicate("t", boundary)

    # The pushed read must match a full scan re-filtered in memory — same rows, no crash.
    src = DeltaSource(path)
    got = pa.Table.from_batches(src.read(predicate=predicate))
    allrows = pa.Table.from_batches(DeltaSource(path).read())
    expected = allrows.filter(pc.greater(allrows.column("t"), pa.scalar(boundary)))

    assert sorted(got.column("id").to_pylist()) == sorted(expected.column("id").to_pylist())
    assert sorted(got.column("id").to_pylist()) == [3, 4]


def test_pushed_predicate_on_utc_timestamp_split(tmp_path) -> None:
    """The worker-side split read applies the same predicate without crashing."""
    path = str(tmp_path / "events2")
    ts_type = pa.timestamp("us", tz="UTC")
    base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    _append(
        pa.table({"id": [1], "t": pa.array([base], type=ts_type)}),
        path,
        0,
    )
    _append(
        pa.table({"id": [2], "t": pa.array([base + dt.timedelta(days=10)], type=ts_type)}),
        path,
        1,
    )
    predicate = _ts_predicate("t", dt.datetime(2024, 1, 5, tzinfo=dt.timezone.utc))
    ids: list[int] = []
    for split in DeltaSource(path).splits(predicate=predicate):
        for batch in split.read(predicate=predicate):
            ids += pa.Table.from_batches([batch]).column("id").to_pylist()
    assert sorted(ids) == [2]
