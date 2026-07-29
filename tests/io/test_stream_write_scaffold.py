"""Every single-file streaming write goes through the same scaffold.

`FileSink.write_stream` owns destination normalization, filesystem resolution honoring the
caller's `filesystem`/`storage_options`, the `resume` short-circuit, the empty-stream case,
and the `WrittenFile` accounting. Two sinks (NDJSON and CSV) override `write_stream` to swap
the *encoding* — straight-through NDJSON, a parallel CSV window — and each had restated the
scaffold around it. Both copies had drifted: they resolved the filesystem with the
module-level `resolve_filesystem`, silently dropping `storage_options` and `filesystem`, and
they skipped `_dest`, so a `Path` destination reached the manifest un-normalized.

These tests pin the scaffold's behavior across every sink that overrides `write_stream`, so
the next override cannot quietly drop a piece of it again.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.base.sink import FileSink
from batcher.io.formats.semistructured.json import JSONSink
from batcher.io.formats.structured.csv import CSVSink
from batcher.io.formats.structured.parquet.sink import ParquetSink

#: The sinks whose `write_stream` is either the base or an encoding-only override.
SINKS = [
    pytest.param(ParquetSink, "out.parquet", id="parquet-base"),
    pytest.param(JSONSink, "out.json", id="json-override"),
    pytest.param(CSVSink, "out.csv", id="csv-override"),
]


def _batches():
    return iter(bt.from_pydict({"x": [1, 2, 3], "y": ["a", "b", "c"]}).iter_batches())


@pytest.mark.parametrize(("sink_cls", "name"), SINKS)
def test_write_stream_normalizes_a_path_destination(tmp_path, sink_cls, name):
    # A `PosixPath` in the manifest breaks the commit that reads it back, so the destination
    # must be normalized to a plain string whichever sink wrote it.
    written = sink_cls().write_stream(_batches(), tmp_path / name)
    assert isinstance(written.path, str)
    assert Path(written.path).exists()
    assert written.rows == 3


@pytest.mark.parametrize(("sink_cls", "name"), SINKS)
def test_write_stream_honors_the_callers_filesystem(tmp_path, monkeypatch, sink_cls, name):
    # `storage_options`/`filesystem` are how a caller passes credentials, and only
    # `FileSink._resolve` applies them. A sink that calls the module-level
    # `resolve_filesystem` instead writes to a private bucket with the worker's ambient
    # credentials and never says so, which is what both overrides used to do.
    seen: list[dict[str, str] | None] = []
    real = FileSink._resolve

    def spy(self, path):
        seen.append(self._storage_options)
        return real(self, path)

    monkeypatch.setattr(FileSink, "_resolve", spy)
    sink_cls(storage_options={"marker": "1"}).write_stream(_batches(), str(tmp_path / name))
    assert seen == [{"marker": "1"}], (
        f"{sink_cls.__name__}.write_stream bypassed self._resolve, so the caller's "
        f"storage_options never reached the filesystem"
    )


@pytest.mark.parametrize(("sink_cls", "name"), SINKS)
def test_write_stream_resume_leaves_a_complete_file_untouched(tmp_path, sink_cls, name):
    first = sink_cls().write_stream(_batches(), tmp_path / name)
    again = sink_cls().write_stream(_batches(), tmp_path / name, resume=True)
    assert again.rows == 0  # skipped, not rewritten
    assert again.bytes == first.bytes


@pytest.mark.parametrize(("sink_cls", "name"), SINKS)
def test_write_stream_of_nothing_still_writes_a_valid_file(tmp_path, sink_cls, name):
    schema = pa.schema([("x", pa.int64()), ("y", pa.string())])
    written = sink_cls().write_stream(iter([]), tmp_path / name, schema=schema)
    assert written.rows == 0
    assert Path(written.path).exists()
