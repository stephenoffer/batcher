"""Workstream E — the `rate` and `socket` development streaming sources."""

from __future__ import annotations

import socket
import threading

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.dev import RateSource

pytestmark = pytest.mark.integration


def test_rate_source_registered_and_schema():
    assert "rate" in SOURCES
    assert "socket" in SOURCES
    src = RateSource(rows_per_second=5, num_rows=12, pace=False)
    assert src.bounded is False
    assert src.schema().names == ["timestamp", "value"]


def test_rate_source_generates_bounded_stream():
    ds = bt.read.rate(5, num_rows=12, pace=False)
    rows = pa.Table.from_batches(list(ds.iter_batches()))
    assert rows.num_rows == 12
    # `value` counts up 0..11; timestamps strictly increase.
    assert rows.column("value").to_pylist() == list(range(12))
    ts = rows.column("timestamp").to_pylist()
    assert ts == sorted(ts) and len(set(ts)) == 12


def test_rate_source_unbounded_read_raises():
    with pytest.raises(PlanError, match="unbounded"):
        RateSource(rows_per_second=1).read()


def test_rate_through_pipeline_streams():
    # The rate source flows through the normal relational engine; an unbounded
    # source streams via iter_batches (count() would refuse to materialize it).
    ds = bt.read.rate(10, num_rows=30, pace=False).filter(bt.col("value") >= 10)
    rows = pa.Table.from_batches(list(ds.iter_batches()))
    assert rows.num_rows == 20
    assert min(rows.column("value").to_pylist()) == 10


def test_socket_source_reads_lines():
    # Stand up a one-shot TCP server that sends three lines then closes.
    lines = [b"alpha\n", b"beta\n", b"gamma\n"]
    srv = socket.create_server(("localhost", 0))
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        with conn:
            for line in lines:
                conn.sendall(line)
        srv.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    ds = bt.read.socket("localhost", port)
    rows = pa.Table.from_batches(list(ds.iter_batches()))
    t.join(timeout=5)
    assert rows.column("value").to_pylist() == ["alpha", "beta", "gamma"]
