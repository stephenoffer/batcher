"""The Arrow IPC *stream* format, written with ``ipc_format="stream"`` and read by detection.

Polars writes and reads both IPC layouts (``write_ipc`` / ``write_ipc_stream``); Batcher wrote
and read only the file format, so a stream from Polars was unreadable and nothing could produce
one. The stream has no footer, which is also why the reader cannot split it by block: the
split test pins that it falls back to one whole-file split rather than failing.
"""

from __future__ import annotations

import pickle

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

import batcher as bt
from batcher._internal.errors import FormatError
from batcher.io.formats.structured.arrow_ipc import ArrowIPCSink, ArrowIPCSource

pytestmark = pytest.mark.io

ROWS = {
    "i": [1, None, 3],
    "s": ["a", None, "c" * 1_500_000],
    "nested": [{"k": [1, 2]}, None, {"k": []}],
}


def test_stream_format_round_trips_nulls_nested_and_large_strings(tmp_path):
    out = str(tmp_path / "t.arrows")
    source = bt.from_pydict(ROWS)
    manifest = source.write.arrow(out, ipc_format="stream")
    assert manifest.files[0].rows == 3

    with pytest.raises(pa.ArrowInvalid):  # positive control: it really is not the file format
        ipc.open_file(out)
    assert ipc.open_stream(out).read_all().num_rows == 3

    back = bt.read.arrow(out)
    assert back.schema == source.schema
    assert back.to_pydict() == ROWS


def test_file_format_is_still_the_default(tmp_path):
    out = str(tmp_path / "t.arrow")
    bt.from_pydict({"i": [1]}).write.arrow(out)
    assert ipc.open_file(out).num_record_batches == 1


def test_an_empty_stream_reads_back_with_its_schema(tmp_path):
    out = str(tmp_path / "empty.arrows")
    bt.from_pydict({"i": [1]}).filter(bt.col("i") > 5).write.arrow(out, ipc_format="stream")
    back = bt.read.arrow(out)
    assert back.schema.field("i").type == pa.int64()
    assert back.count() == 0


def test_a_stream_is_one_picklable_whole_file_split(tmp_path):
    out = str(tmp_path / "s.arrows")
    bt.from_pydict({"i": list(range(10))}).write.arrow(out, ipc_format="stream")
    splits = ArrowIPCSource(out).splits()
    assert len(splits) == 1
    rebuilt = pickle.loads(pickle.dumps(splits[0]))
    assert sum(b.num_rows for b in rebuilt.read()) == 10


def test_an_unknown_ipc_format_is_refused():
    with pytest.raises(FormatError, match="ipc_format"):
        ArrowIPCSink(ipc_format="feather")


def test_polars_reads_and_writes_the_same_streams(tmp_path):
    pl = pytest.importorskip("polars")
    ours = str(tmp_path / "ours.arrows")
    bt.from_pydict({"i": [1, None], "f": [0.5, 1.5]}).write.arrow(ours, ipc_format="stream")
    frame = pl.read_ipc_stream(ours)
    assert frame.to_dict(as_series=False) == {"i": [1, None], "f": [0.5, 1.5]}
    assert frame.schema == {"i": pl.Int64, "f": pl.Float64}

    theirs = str(tmp_path / "theirs.arrows")
    pl.DataFrame({"i": [7, None], "f": [2.5, None]}).write_ipc_stream(theirs)
    back = bt.read.arrow(theirs)
    assert back.to_pydict() == {"i": [7, None], "f": [2.5, None]}
    assert back.schema.field("i").type == pa.int64()
