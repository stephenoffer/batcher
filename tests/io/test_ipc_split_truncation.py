"""An intermediate IPC stream file truncated at a batch boundary must not read back short.

The Arrow IPC stream reader treats a bare end of file as the end of the stream, so a file cut
exactly between two batches decodes cleanly with the tail missing. `IpcFileSplit` carries the
row count captured at write time and holds the read to it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import FormatError
from batcher.io.splits.file import IpcFileSplit


def _write(path, batches: int) -> list[int]:
    """Write `batches` 1,000-row batches; return the byte offset after each one."""
    batch = pa.record_batch({"x": list(range(1000))})
    offsets = []
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, batch.schema) as w:
        for _ in range(batches):
            w.write_batch(batch)
            offsets.append(sink.tell())
    return offsets


def test_an_intact_file_reads_in_full(tmp_path):
    path = tmp_path / "part.arrows"
    _write(path, 3)
    split = IpcFileSplit(str(path), rows=3000)
    assert sum(b.num_rows for b in split.read()) == 3000
    assert sum(b.num_rows for b in split.iter_batches()) == 3000


def test_a_boundary_truncation_is_an_error_not_a_short_read(tmp_path):
    path = tmp_path / "part.arrows"
    offsets = _write(path, 3)
    data = path.read_bytes()
    path.write_bytes(data[: offsets[1]])  # drop the last batch and the end-of-stream marker
    # Positive control: the format itself reads this back without complaint.
    with pa.OSFile(str(path), "rb") as src, pa.ipc.open_stream(src) as reader:
        assert sum(b.num_rows for b in reader) == 2000

    split = IpcFileSplit(str(path), rows=3000)
    with pytest.raises(FormatError, match="2000 rows but 3000"):
        split.read()
    with pytest.raises(FormatError):
        list(split.iter_batches())


def test_an_unknown_row_count_is_not_checked(tmp_path):
    path = tmp_path / "part.arrows"
    _write(path, 2)
    assert sum(b.num_rows for b in IpcFileSplit(str(path)).read()) == 2000
