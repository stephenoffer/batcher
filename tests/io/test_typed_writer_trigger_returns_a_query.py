"""A typed file writer given `trigger=` returns a `StreamingQuery` (AP-422).

`write.parquet` and its siblings were annotated `-> WriteManifest` while forwarding to a
call that switches to a streaming query on a trigger. The annotation is now an overload on
`trigger`, and these tests pin the runtime half of that contract.
"""

from __future__ import annotations

import typing

import pytest

import batcher as bt
from batcher.api.io_namespace.writer import Writer
from batcher.api.streaming import StreamingQuery
from batcher.io.manifest import WriteManifest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("fmt", ["parquet", "csv", "json"])
def test_a_trigger_returns_a_streaming_query(tmp_path, fmt):
    path = str(tmp_path / "s")
    query = getattr(bt.from_pydict({"v": [1, 2, 3]}).write, fmt)(
        path, trigger=bt.Trigger.available_now()
    )
    assert isinstance(query, StreamingQuery)
    query.await_termination()
    assert getattr(bt.read, fmt)(path).count() == 3


@pytest.mark.parametrize("fmt", ["parquet", "csv", "json"])
def test_no_trigger_on_a_bounded_dataset_returns_a_manifest(tmp_path, fmt):
    manifest = getattr(bt.from_pydict({"v": [1, 2, 3]}).write, fmt)(str(tmp_path / "b"))
    assert isinstance(manifest, WriteManifest)


@pytest.mark.parametrize("name", ["__call__", "parquet", "csv", "json", "delta", "iceberg"])
def test_every_streamable_writer_declares_both_overloads(name):
    overloads = typing.get_overloads(getattr(Writer, name))
    returns = sorted(str(o.__annotations__["return"]) for o in overloads)
    assert returns == ["StreamingQuery", "WriteManifest"]
