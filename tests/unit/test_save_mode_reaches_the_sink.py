"""Regression: a save mode must reach the sink that implements it.

`_MODE_AWARE_SINKS` is the list of sinks that consume `mode` as a constructor option. A
sink missing from that list gets the worst of both worlds, which is exactly what happened
to `snowflake`: `mode="append"` was rejected at the gate even though `SnowflakeSink`
implements it, and `mode="overwrite"` passed the gate but never reached the sink, so the
write quietly appended instead.

A save mode that silently does the opposite of what it says is data corruption, not a
missing feature. These tests pin the wiring: the mode the caller asked for is the mode the
sink is constructed with.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.io_namespace.writer import _MODE_AWARE_SINKS
from batcher.io.manifest import WrittenFile
from batcher.io.sink import SINKS

pytestmark = pytest.mark.unit


class _RecordingSink:
    """Stands in for a real warehouse sink and records how it was constructed."""

    last_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = dict(kwargs)

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        return WrittenFile(path=path, rows=table.num_rows, bytes=0)

    def commit(self, manifest: Any, path: str) -> None:
        return None


@pytest.fixture
def recording_snowflake(monkeypatch):
    """Swap the registered `snowflake` sink for one that records its kwargs."""
    _RecordingSink.last_kwargs = {}
    monkeypatch.setitem(SINKS._items, "snowflake", _RecordingSink)
    return _RecordingSink


def test_snowflake_is_declared_mode_aware():
    # The bug was an omission from this set, so assert the membership directly.
    assert "snowflake" in _MODE_AWARE_SINKS


@pytest.mark.parametrize("mode", ["append", "overwrite"])
def test_snowflake_write_passes_the_mode_to_the_sink(recording_snowflake, mode):
    ds = bt.from_pydict({"x": [1, 2, 3]})
    ds.write.snowflake("db.schema.t", mode=mode, connection_kwargs={})
    assert recording_snowflake.last_kwargs.get("mode") == mode


def test_snowflake_append_is_no_longer_rejected(recording_snowflake):
    # `mode="append"` used to raise PlanError at the gate, despite the sink implementing it.
    ds = bt.from_pydict({"x": [1]})
    ds.write.snowflake("db.schema.t", mode="append", connection_kwargs={})
    assert recording_snowflake.last_kwargs["mode"] == "append"


def test_a_file_sink_still_rejects_append():
    # File sinks genuinely cannot append; the gate must keep saying so.
    ds = bt.from_pydict({"x": [1]})
    with pytest.raises(PlanError, match="append"):
        ds.write.parquet("/tmp/does-not-matter.parquet", mode="append")
