"""Regression: `bt.read.bigquery(query)` must bind the SQL to `query`, not `project`.

`BigQuerySource`'s first field is `project`, and the reader forwarded its `query` argument
*positionally*. So the SQL text landed in `project`, and a caller who also passed
`project=` — which BigQuery requires — got

    TypeError: got multiple values for argument 'project'

instead of a query. The documented spelling of the API simply did not work. This pins the
binding by capturing what the source is actually constructed with.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.source import SOURCES

pytestmark = pytest.mark.unit


class _RecordingSource:
    """Stands in for the real BigQuery source and records its constructor kwargs."""

    last_kwargs: ClassVar[dict[str, Any]] = {}
    supports_predicate = False

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = dict(kwargs)

    def schema(self) -> pa.Schema:
        return pa.schema([pa.field("x", pa.int64())])

    def read(self, projection=None, predicate=None) -> list[pa.RecordBatch]:
        return [pa.record_batch({"x": pa.array([1], pa.int64())})]

    def row_count(self) -> int | None:
        return 1

    def identity(self) -> str:
        return "recording"

    def splits(self, target_size=None, predicate=None):
        return []


@pytest.fixture
def recording_bigquery(monkeypatch):
    _RecordingSource.last_kwargs = {}
    monkeypatch.setitem(SOURCES._items, "bigquery", _RecordingSource)
    return _RecordingSource


def test_query_binds_to_query_not_project(recording_bigquery):
    bt.read.bigquery("SELECT 1", project="my-project")
    assert recording_bigquery.last_kwargs == {
        "query": "SELECT 1",
        "project": "my-project",
    }


def test_table_form_still_works(recording_bigquery):
    bt.read.bigquery(table="p.d.t", project="my-project")
    assert recording_bigquery.last_kwargs == {
        "table": "p.d.t",
        "project": "my-project",
    }
    assert "query" not in recording_bigquery.last_kwargs


def test_extra_options_pass_through(recording_bigquery):
    bt.read.bigquery("SELECT 1", project="p", max_streams=4)
    assert recording_bigquery.last_kwargs["max_streams"] == 4
    assert recording_bigquery.last_kwargs["query"] == "SELECT 1"
