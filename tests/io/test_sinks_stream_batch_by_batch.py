"""Every streaming-capable sink added for the reader/writer wave encodes one batch at a time.

`FileSink.write_stream` falls back to collecting the whole stream into one table when a sink
has no incremental writer, and that fallback is invisible from the output: the file is
identical either way. So the only way to prove a sink streams is to watch which path ran. Each
case below subclasses the sink so the incremental hook records the batch sizes it was handed
and the buffered `_write_file` raises; a sink that silently fell back would fail loudly.

The NumPy sink is deliberately absent: a ``.npy`` header states the row count before the
data, so it cannot stream, and `write.numpy` bounds memory with ``max_rows_per_file``.
"""

from __future__ import annotations

import io
import tarfile
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from batcher.io.formats.ml.tfrecord import TFRecordSink
from batcher.io.formats.ml.webdataset import WebDatasetSink
from batcher.io.formats.semistructured.xml import XMLSink
from batcher.io.formats.structured.arrow_ipc import ArrowIPCSink
from batcher.io.formats.unstructured.text import TextSink

pytestmark = pytest.mark.io


def _spy(sink_cls: type, **kwargs):
    """An instance of `sink_cls` that records streamed batch sizes and forbids buffering."""

    def _write_batch(self, writer, batch):
        self.seen.append(batch.num_rows)
        sink_cls._write_batch(self, writer, batch)

    def _write_file(self, table, fh):
        raise AssertionError(f"{sink_cls.__name__} buffered the stream into one table")

    spy_cls = type(
        f"Spy{sink_cls.__name__}",
        (sink_cls,),
        {"__slots__": ("seen",), "_write_batch": _write_batch, "_write_file": _write_file},
    )
    sink = spy_cls(**kwargs)
    sink.seen = []
    return sink


def _text_lines(data: bytes) -> list[str]:
    return data.decode().splitlines()


def _xml_values(data: bytes) -> list[str]:
    return [row.findtext("v") for row in ET.fromstring(data).findall("ROW")]


def _tar_names(data: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return tar.getnames()


def _ipc_stream_values(data: bytes) -> list[str]:
    return ipc.open_stream(pa.BufferReader(data)).read_all().column("v").to_pylist()


def _record_count(data: bytes) -> int:
    count, offset = 0, 0
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 8], "little")
        offset += 8 + 4 + length + 4
        count += 1
    return count


VALUES = [f"v{i}" for i in range(7)]

CASES = [
    (TextSink, {}, pa.table({"v": VALUES}), _text_lines, VALUES),
    (XMLSink, {}, pa.table({"v": VALUES}), _xml_values, VALUES),
    (
        WebDatasetSink,
        {},
        pa.table({"__key__": [f"k{i}" for i in range(7)], "v": VALUES}),
        _tar_names,
        [f"k{i}.v" for i in range(7)],
    ),
    (ArrowIPCSink, {"ipc_format": "stream"}, pa.table({"v": VALUES}), _ipc_stream_values, VALUES),
    (TFRecordSink, {}, pa.table({"v": VALUES}), _record_count, 7),
]


@pytest.mark.parametrize(
    ("sink_cls", "kwargs", "table", "decode", "expected"),
    CASES,
    ids=[case[0].__name__ for case in CASES],
)
def test_a_streaming_write_hands_the_sink_one_batch_at_a_time(
    tmp_path, sink_cls, kwargs, table, decode, expected
):
    if sink_cls is TFRecordSink:
        try:
            import google_crc32c  # noqa: F401
        except ImportError:
            pytest.importorskip("crc32c")
    sink = _spy(sink_cls, **kwargs)
    out = tmp_path / "stream.out"
    written = sink.write_stream(iter(table.to_batches(max_chunksize=3)), str(out))
    assert sink.seen == [3, 3, 1]
    assert written.rows == 7
    assert decode(out.read_bytes()) == expected
