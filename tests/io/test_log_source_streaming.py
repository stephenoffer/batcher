"""A log file must stream, not materialize — it is the archetypal larger-than-memory input.

`_read_file` batched internally but accumulated every batch before returning, so
`iter_batches` was streaming in name only: a multi-GB log was fully resident as Arrow
before its first batch reached the consumer. The neighbouring text source already fixed
this shape for its line mode; the log format, which exists for nothing else, kept it.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt
from batcher.config import Config, config_context
from batcher.io.formats.semistructured.logs import LOG_SCHEMA, LogSource


def _write_log(path, lines: int) -> str:
    p = path / "app.log"
    p.write_text("".join(f"line {i}\n" for i in range(lines)))
    return str(p)


@pytest.fixture
def small_morsels():
    base = Config()
    cfg = base.replace(execution=dataclasses.replace(base.execution, morsel_rows=4))
    with config_context(cfg):
        yield


def test_the_first_batch_arrives_before_the_file_is_fully_decoded(tmp_path, small_morsels):
    """The property that makes it a stream: one batch out with most of the file unread."""
    path = _write_log(tmp_path, 1_000)
    stream = LogSource(path).iter_batches()
    first = next(stream)
    assert first.num_rows == 4
    assert first.column("line")[0].as_py() == "line 0"
    stream.close()


def test_streaming_yields_every_line_once_in_order(tmp_path, small_morsels):
    path = _write_log(tmp_path, 37)
    batches = list(LogSource(path).iter_batches())
    assert sum(b.num_rows for b in batches) == 37
    lines = [v for b in batches for v in b.column("line").to_pylist()]
    assert lines == [f"line {i}" for i in range(37)]
    numbers = [v for b in batches for v in b.column("line_number").to_pylist()]
    assert numbers == list(range(1, 38))


def test_streaming_and_reading_whole_agree(tmp_path, small_morsels):
    """`read()` keeps its list contract by draining the same loop, so the two cannot drift."""
    path = _write_log(tmp_path, 30)
    src = LogSource(path)
    streamed = [b.to_pydict() for b in src.iter_batches()]
    whole = [b.to_pydict() for b in src.read()]
    assert streamed == whole


def test_batches_are_capped_at_a_morsel(tmp_path, small_morsels):
    path = _write_log(tmp_path, 26)
    sizes = [b.num_rows for b in LogSource(path).iter_batches()]
    assert sizes == [4, 4, 4, 4, 4, 4, 2]


def test_an_empty_log_still_yields_a_schema_bearing_batch(tmp_path, small_morsels):
    """A reader that produced no batch at all leaves the caller nothing to type an empty
    result from."""
    p = tmp_path / "empty.log"
    p.write_text("")
    batches = list(LogSource(str(p)).iter_batches())
    assert [b.num_rows for b in batches] == [0]
    assert batches[0].schema == LOG_SCHEMA


def test_a_projection_narrows_the_streamed_batches(tmp_path, small_morsels):
    path = _write_log(tmp_path, 10)
    batches = list(LogSource(path).iter_batches(["line"]))
    assert all(b.schema.names == ["line"] for b in batches)


def test_the_public_reader_streams_the_same_rows(tmp_path, small_morsels):
    path = _write_log(tmp_path, 12)
    ds = bt.read(path, format="logs")
    assert ds.count() == 12
    streamed = [v for b in ds.iter_batches() for v in b.column("line").to_pylist()]
    assert streamed == [f"line {i}" for i in range(12)]


# --------------------------------------------------------------------------
# The same shape in the protobuf reader — and the two that must NOT be changed.
# --------------------------------------------------------------------------
def test_the_protobuf_reader_streams_and_reads_through_one_loop():
    """Asserted structurally because `protarrow` is an optional extra: what matters is that
    `read()` drains the *same* generator `iter_batches` pulls, so they cannot drift."""
    import inspect

    from batcher.io.formats.semistructured import protobuf

    assert hasattr(protobuf.ProtobufSource, "_iter_file")
    read = inspect.getsource(protobuf.ProtobufSource._read_file)
    assert "_batches_from" in read
    stream = inspect.getsource(protobuf.ProtobufSource._iter_file)
    assert "_batches_from" in stream


@pytest.mark.parametrize("module", ["msgpack", "webdataset"])
def test_the_inference_bound_readers_deliberately_do_not_stream(module):
    """MessagePack must see every record before it can type any of them, and a WebDataset
    shard's schema comes from the union of member extensions across the whole shard. Both
    would produce batches that fail to concatenate if split, so materializing is the
    correct behavior — pinned so a later "optimization" does not quietly break typing."""
    import importlib
    import inspect

    pkg = "semistructured" if module == "msgpack" else "ml"
    mod = importlib.import_module(f"batcher.io.formats.{pkg}.{module}")
    cls = next(
        obj
        for _n, obj in vars(mod).items()
        if inspect.isclass(obj) and hasattr(obj, "_read_file") and obj.__module__ == mod.__name__
    )
    assert "_iter_file" not in vars(cls), (
        f"{cls.__name__} gained a streaming reader; its schema is inferred across the whole "
        "file, so split batches would disagree on types"
    )
