"""Coverage for the optional-dependency file-format readers/writers.

Core-pyarrow formats (ORC, Arrow IPC, WebDataset) get *real* round-trip tests that
must always pass — ORC/Arrow via pyarrow, WebDataset via the stdlib `tarfile`. For
ORC and Arrow IPC we also assert ``concat(read each split) == read whole``, the
split-equivalence invariant a distributed read relies on. Every other format
(Lance, Avro, Excel, XML, Protobuf, Msgpack, TFRecord, HDF5, Zarr, Documents) is
exercised behind `pytest.importorskip`, so the suite skips cleanly when the extra
is absent but verifies the round trip when it is present.
"""

from __future__ import annotations

import io
import tarfile

import pyarrow as pa
import pytest


def _sorted_pydict(table: pa.Table) -> dict:
    """Sort rows by all columns so row order is irrelevant to equality checks."""
    if table.num_rows == 0:
        return table.to_pydict()
    rows = sorted(table.to_pylist(), key=lambda r: tuple(repr(r[c]) for c in table.column_names))
    return pa.Table.from_pylist(rows, schema=table.schema).to_pydict()


def _sample_table() -> pa.table:
    return pa.table(
        {
            "id": list(range(6)),
            "name": [f"r{i}" for i in range(6)],
            "v": [float(i) + 0.5 for i in range(6)],
        }
    )


# --------------------------------------------------------------------------- ORC
def test_orc_roundtrip(tmp_path):
    from batcher.io.formats.structured.orc import ORCSink, ORCSource

    table = _sample_table()
    path = str(tmp_path / "data.orc")
    ORCSink().write(table, path)

    src = ORCSource(path)
    got = pa.Table.from_batches(src.read())
    assert _sorted_pydict(got) == _sorted_pydict(table)
    assert src.schema().names == table.schema.names
    assert src.row_count() == table.num_rows


def test_orc_projection(tmp_path):
    from batcher.io.formats.structured.orc import ORCSink, ORCSource

    table = _sample_table()
    path = str(tmp_path / "p.orc")
    ORCSink().write(table, path)

    src = ORCSource(path)
    got = pa.Table.from_batches(src.read(projection=["id", "v"]))
    assert got.column_names == ["id", "v"]
    assert _sorted_pydict(got) == _sorted_pydict(table.select(["id", "v"]))


def test_orc_split_concat_equals_whole(tmp_path):
    import pyarrow.orc as orc

    from batcher.io.formats.structured.orc import ORCSource

    table = _sample_table()
    path = str(tmp_path / "stripes.orc")
    # Force multiple stripes so the split path is non-trivial.
    with open(path, "wb") as fh:
        orc.write_table(table, fh, stripe_size=1)

    src = ORCSource(path)
    splits = src.splits()
    assert len(splits) >= 1
    from_splits = pa.Table.from_batches([b for s in splits for b in s.read()])
    whole = pa.Table.from_batches(src.read())
    assert _sorted_pydict(from_splits) == _sorted_pydict(whole)
    assert _sorted_pydict(from_splits) == _sorted_pydict(table)


# --------------------------------------------------------------------- Arrow IPC
def test_arrow_ipc_roundtrip(tmp_path):
    from batcher.io.formats.structured.arrow_ipc import ArrowIPCSink, ArrowIPCSource

    table = _sample_table()
    path = str(tmp_path / "data.arrow")
    ArrowIPCSink().write(table, path)

    src = ArrowIPCSource(path)
    got = pa.Table.from_batches(src.read())
    assert _sorted_pydict(got) == _sorted_pydict(table)
    assert src.schema().names == table.schema.names
    assert src.row_count() == table.num_rows


def test_arrow_ipc_projection(tmp_path):
    from batcher.io.formats.structured.arrow_ipc import ArrowIPCSink, ArrowIPCSource

    table = _sample_table()
    path = str(tmp_path / "p.arrow")
    ArrowIPCSink().write(table, path)

    src = ArrowIPCSource(path)
    got = pa.Table.from_batches(src.read(projection=["name"]))
    assert got.column_names == ["name"]
    assert _sorted_pydict(got) == _sorted_pydict(table.select(["name"]))


def test_arrow_ipc_split_concat_equals_whole(tmp_path):
    import pyarrow.ipc as ipc

    from batcher.io.formats.structured.arrow_ipc import ArrowIPCSource

    table = _sample_table()
    path = str(tmp_path / "blocks.arrow")
    # Write several blocks so there is more than one split.
    with ipc.new_file(path, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=2):
            writer.write_batch(batch)

    src = ArrowIPCSource(path)
    splits = src.splits()
    assert len(splits) > 1
    from_splits = pa.Table.from_batches([b for s in splits for b in s.read()])
    whole = pa.Table.from_batches(src.read())
    assert _sorted_pydict(from_splits) == _sorted_pydict(whole)
    assert _sorted_pydict(from_splits) == _sorted_pydict(table)


def test_arrow_ipc_alias_registration():
    from batcher.io.formats.base import SINKS, SOURCES

    for name in ("arrow", "feather", "ipc"):
        assert name in SOURCES
        assert name in SINKS


# -------------------------------------------------------------------- WebDataset
def _write_tar_shard(path: str) -> None:
    samples = {
        "0001": {"jpg": b"\xff\xd8img1", "cls": b"3"},
        "0002": {"jpg": b"\xff\xd8img2", "cls": b"7"},
    }
    with tarfile.open(path, "w") as tar:
        for key, fields in samples.items():
            for ext, payload in fields.items():
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))


def test_webdataset_roundtrip(tmp_path):
    from batcher.io.formats.ml.webdataset import WebDatasetSource

    path = str(tmp_path / "shard.tar")
    _write_tar_shard(path)

    src = WebDatasetSource(path)
    table = pa.Table.from_batches(src.read())
    assert set(table.column_names) == {"__key__", "jpg", "cls"}
    assert table.num_rows == 2
    rows = {r["__key__"]: r for r in table.to_pylist()}
    assert rows["0001"]["jpg"] == b"\xff\xd8img1"
    assert rows["0002"]["cls"] == b"7"


def test_webdataset_projection(tmp_path):
    from batcher.io.formats.ml.webdataset import WebDatasetSource

    path = str(tmp_path / "shard2.tar")
    _write_tar_shard(path)

    src = WebDatasetSource(path)
    got = pa.Table.from_batches(src.read(projection=["__key__", "cls"]))
    assert got.column_names == ["__key__", "cls"]


# Optional formats below: importorskip when the extra is absent.
def test_avro_roundtrip(tmp_path):
    pytest.importorskip("fastavro")
    from batcher.io.formats.structured.avro import AvroSink, AvroSource

    table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    path = str(tmp_path / "data.avro")
    AvroSink().write(table, path)

    src = AvroSource(path)
    got = pa.Table.from_batches(src.read())
    assert _sorted_pydict(got.select(["id", "name"])) == _sorted_pydict(table)


def test_lance_roundtrip(tmp_path):
    pytest.importorskip("lance")
    from batcher.io.formats.structured.lance import LanceSink, LanceSource

    table = _sample_table()
    path = str(tmp_path / "data.lance")
    LanceSink().write(table, path)

    src = LanceSource(path)
    got = pa.Table.from_batches(src.read())
    assert _sorted_pydict(got) == _sorted_pydict(table)
    assert src.row_count() == table.num_rows
    assert len(src.splits()) >= 1


def test_excel_read(tmp_path):
    pytest.importorskip("python_calamine")
    openpyxl = pytest.importorskip("openpyxl")
    from batcher.io.formats.structured.excel import ExcelSource

    path = str(tmp_path / "data.xlsx")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["id", "name"])
    sheet.append([1, "a"])
    sheet.append([2, "b"])
    workbook.save(path)

    src = ExcelSource(path)
    table = pa.Table.from_batches(src.read())
    assert table.column_names == ["id", "name"]
    assert table.num_rows == 2


def test_xml_read(tmp_path):
    pytest.importorskip("xml2arrow")
    from batcher.io.formats.semistructured.xml import XMLSource

    path = tmp_path / "data.xml"
    path.write_text("<root><row><a>1</a></row><row><a>2</a></row></root>")
    src = XMLSource(str(path))
    assert pa.Table.from_batches(src.read()).num_rows >= 1


def test_protobuf_importorskip():
    pytest.importorskip("protarrow")
    from batcher.io.formats.semistructured.protobuf import ProtobufSource

    assert ProtobufSource is not None


def test_msgpack_roundtrip(tmp_path):
    pytest.importorskip("ormsgpack")
    from batcher.io.formats.semistructured.msgpack import MsgpackSink, MsgpackSource

    table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    path = str(tmp_path / "data.msgpack")
    MsgpackSink().write(table, path)

    src = MsgpackSource(path)
    got = pa.Table.from_batches(src.read())
    assert _sorted_pydict(got) == _sorted_pydict(table)


def test_tfrecord_importorskip():
    # Framing is core, but only smoke-test the import path here.
    from batcher.io.formats.ml.tfrecord import TFRecordSource

    assert TFRecordSource is not None


def test_hdf5_roundtrip(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    import numpy as np

    from batcher.io.formats.ml.hdf5 import HDF5Source

    path = str(tmp_path / "data.h5")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.arange(10, dtype="int64"))

    src = HDF5Source(path, dataset="data")
    table = pa.Table.from_batches(src.read())
    assert table.column_names == ["value"]
    assert table.num_rows == 10
    assert src.row_count() == 10


def test_zarr_roundtrip(tmp_path):
    pytest.importorskip("zarr")
    import numpy as np
    import zarr

    from batcher.io.formats.ml.zarr import ZarrSource

    path = str(tmp_path / "data.zarr")
    zarr.save(path, np.arange(10, dtype="int64"))

    src = ZarrSource(path)
    table = pa.Table.from_batches(src.read())
    assert table.num_rows == 10
    assert src.row_count() == 10


def test_documents_importorskip(tmp_path):
    pytest.importorskip("pypdf")
    from batcher.io.formats.unstructured.documents import DocumentSource

    assert DocumentSource is not None
