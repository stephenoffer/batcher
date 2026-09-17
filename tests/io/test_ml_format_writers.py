"""The NumPy, WebDataset and TFRecord writers, which Ray Data has and Batcher did not.

Each is the inverse of a reader Batcher already had, so the core check is the round trip:
write, read back through `bt.read.<format>`, and compare values *and* Arrow types. The rest
pins what each format cannot hold (a null in a `.npy`, a nested cell in a tar member) as a
typed refusal, and, for TFRecord, that TensorFlow itself parses what was written, since a
record whose checksum or protobuf is wrong reads back fine through a reader that does not
check either.
"""

from __future__ import annotations

import tarfile

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, FormatError, SchemaError
from batcher.io.formats.base import SINKS
from batcher.io.formats.ml.numpy import NumpySink
from batcher.io.formats.ml.tfrecord import TFRecordSink
from batcher.io.formats.ml.webdataset import WebDatasetSink

pytestmark = pytest.mark.io


def test_the_three_sinks_are_registered():
    assert SINKS.get("numpy") is NumpySink
    assert SINKS.get("webdataset") is WebDatasetSink
    assert SINKS.get("tfrecord") is TFRecordSink


# --- numpy ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("array", "arrow_type"),
    [
        (np.array([1.5, -2.0, 3.25]), pa.float64()),
        (np.array([1, 2, 3], dtype=np.int64), pa.int64()),
        (np.array([True, False]), pa.bool_()),
        (np.arange(12, dtype=np.float32).reshape(4, 3), pa.list_(pa.float32(), 3)),
    ],
    ids=["float", "int", "bool", "fixed-size-list"],
)
def test_numpy_round_trips_values_and_types(tmp_path, array, arrow_type):
    out = str(tmp_path / "a.npy")
    bt.from_numpy(array).write.numpy(out)
    np.testing.assert_array_equal(np.load(out), array)
    back = bt.read.numpy(out)
    assert back.schema.field("data").type == arrow_type
    assert back.to_pydict()["data"] == array.tolist()


def test_numpy_round_trips_a_tensor_column_with_its_shape(tmp_path):
    array = np.arange(24, dtype=np.int32).reshape(3, 2, 2, 2)
    out = str(tmp_path / "t.npy")
    bt.from_numpy(array).write.numpy(out)
    np.testing.assert_array_equal(np.load(out), array)
    back = bt.read.numpy(out)
    assert isinstance(back.schema.field("data").type, pa.FixedShapeTensorType)
    assert back.schema.field("data").type.shape == [2, 2, 2]


def test_numpy_picks_the_named_column_and_writes_an_empty_array(tmp_path):
    ds = bt.from_pydict({"id": [1, 2, 3], "x": [0.5, 1.5, 2.5]})
    out = str(tmp_path / "x.npy")
    ds.write.numpy(out, column="x")
    np.testing.assert_array_equal(np.load(out), [0.5, 1.5, 2.5])

    empty = str(tmp_path / "empty.npy")
    ds.filter(bt.col("id") > 10).write.numpy(empty, column="id")
    assert np.load(empty).shape == (0,)
    assert bt.read.numpy(empty).count() == 0


def test_numpy_directory_write_reads_back_as_one_column(tmp_path):
    out = str(tmp_path / "parts")
    manifest = bt.from_pydict({"v": list(range(10))}).write.numpy(out, max_rows_per_file=4)
    assert [f.rows for f in manifest.files] == [4, 4, 2]
    assert sorted(bt.read.numpy(out).to_pydict()["data"]) == list(range(10))


def test_numpy_refuses_nulls_strings_and_an_ambiguous_column(tmp_path):
    out = str(tmp_path / "bad.npy")
    with pytest.raises(SchemaError, match="null"):
        bt.from_pydict({"x": [1.0, None]}).write.numpy(out)
    with pytest.raises(SchemaError, match="cannot write column 's'"):
        bt.from_pydict({"s": ["a" * 100_000]}).write.numpy(out)
    with pytest.raises(SchemaError, match=r"column=\.\.\."):
        bt.from_pydict({"a": [1], "b": [2]}).write.numpy(out)
    with pytest.raises(ColumnNotFoundError):
        bt.from_pydict({"a": [1]}).write.numpy(out, column="nope")


# --- webdataset -----------------------------------------------------------------------------


def test_webdataset_round_trips_bytes_text_numbers_and_missing_members(tmp_path):
    out = str(tmp_path / "shard.tar")
    ds = bt.from_pydict(
        {
            "__key__": ["dir/s0", "dir/s1", "dir/s2"],
            "jpg": [b"\xff\xd8\x00", b"", None],
            "txt": ["café", None, "x" * 2_000_000],
            "cls": [3, 4, None],
            "flag": [True, False, None],
        }
    )
    manifest = ds.write.webdataset(out)
    assert manifest.files[0].rows == 3
    with tarfile.open(out) as tar:
        assert tar.getnames()[:4] == ["dir/s0.jpg", "dir/s0.txt", "dir/s0.cls", "dir/s0.flag"]
        assert all(m.mtime == 0 for m in tar.getmembers())

    back = bt.read.webdataset(out)
    assert back.schema == pa.schema(
        [("__key__", pa.string())] + [(n, pa.binary()) for n in ("jpg", "txt", "cls", "flag")]
    )
    got = back.to_pydict()
    assert got["__key__"] == ["dir/s0", "dir/s1", "dir/s2"]
    assert got["jpg"] == [b"\xff\xd8\x00", b"", None]
    assert got["txt"] == ["café".encode(), None, b"x" * 2_000_000]
    assert got["cls"] == [b"3", b"4", None]
    assert got["flag"] == [b"1", b"0", None]


def test_webdataset_empty_result_is_an_empty_tar(tmp_path):
    out = str(tmp_path / "empty.tar")
    ds = bt.from_pydict({"__key__": ["a"], "txt": ["x"]}).filter(bt.col("txt") == "no")
    ds.write.webdataset(out)
    with tarfile.open(out) as tar:
        assert tar.getnames() == []


def test_webdataset_refuses_a_missing_key_a_dotted_key_and_a_nested_cell(tmp_path):
    out = str(tmp_path / "bad.tar")
    with pytest.raises(SchemaError, match="__key__"):
        bt.from_pydict({"txt": ["x"]}).write.webdataset(out)
    with pytest.raises(SchemaError, match=r"sample key 'a\.b'"):
        bt.from_pydict({"__key__": ["a.b"], "txt": ["x"]}).write.webdataset(out)
    with pytest.raises(SchemaError, match="cannot write column 'meta'"):
        bt.from_pydict({"__key__": ["a"], "meta": [{"k": 1}]}).write.webdataset(out)
    with pytest.raises(SchemaError, match="repeats"):
        bt.from_pydict({"__key__": ["a", "a"], "txt": ["x", "y"]}).write.webdataset(out)


def test_webdataset_refuses_a_key_repeated_across_a_batch_boundary(tmp_path):
    sink = WebDatasetSink()
    table = pa.table({"__key__": ["a", "b", "b", "c"], "txt": ["1", "2", "3", "4"]})
    with pytest.raises(SchemaError, match="repeats"):
        sink.write_stream(iter(table.to_batches(max_chunksize=2)), str(tmp_path / "s.tar"))
    # Positive control: the same batches with distinct keys write and read back whole.
    ok = pa.table({"__key__": ["a", "b", "c", "d"], "txt": ["1", "2", "3", "4"]})
    sink.write_stream(iter(ok.to_batches(max_chunksize=2)), str(tmp_path / "ok.tar"))
    assert bt.read.webdataset(str(tmp_path / "ok.tar")).count() == 4


# --- tfrecord -------------------------------------------------------------------------------


def _requires_crc() -> None:
    """Writing a TFRecord needs a CRC32C implementation; skip where neither is installed."""
    try:
        import google_crc32c  # noqa: F401
    except ImportError:
        pytest.importorskip("crc32c")


def test_tfrecord_raw_round_trips_the_record_payloads(tmp_path):
    _requires_crc()
    payloads = [b"", b"\x00\x01", b"z" * 3_000_000]
    out = str(tmp_path / "raw.tfrecord")
    bt.from_pydict({"record": payloads}).write.tfrecord(out, record_format="raw")
    back = bt.read.tfrecord(out)
    assert back.schema == pa.schema([("record", pa.binary())])
    assert back.to_pydict()["record"] == payloads


def test_tfrecord_example_records_are_parsed_by_tensorflow(tmp_path):
    _requires_crc()
    tf = pytest.importorskip("tensorflow")
    out = str(tmp_path / "ex.tfrecord")
    ds = bt.from_pydict(
        {
            "label": [7, -5, None],
            "score": [0.5, None, 2.25],
            "text": ["no", "yés", None],
            "ids": [[1, 2, 3], None, []],
            "blob": [b"\x00", b"", None],
            "flag": [True, False, None],
        }
    )
    assert ds.write.tfrecord(out).files[0].rows == 3
    parsed = []
    for raw in tf.data.TFRecordDataset(out):  # verifies every length and payload CRC
        features = tf.train.Example.FromString(raw.numpy()).features.feature
        parsed.append(
            {name: list(getattr(f, f.WhichOneof("kind")).value) for name, f in features.items()}
        )
    assert parsed[0] == {
        "label": [7],
        "score": [0.5],
        "text": [b"no"],
        "ids": [1, 2, 3],
        "blob": [b"\x00"],
        "flag": [1],
    }
    assert parsed[1]["label"] == [-5] and parsed[1]["score"] == [] and parsed[1]["ids"] == []
    assert parsed[1]["text"] == ["yés".encode()] and parsed[1]["blob"] == [b""]
    assert parsed[2] == {
        "label": [],
        "score": [2.25],
        "text": [],
        "ids": [],
        "blob": [],
        "flag": [],
    }
    # And the source reads the same records back as raw payloads.
    assert bt.read.tfrecord(out).count() == 3


def test_tfrecord_empty_result_is_an_empty_file(tmp_path):
    out = str(tmp_path / "empty.tfrecord")
    ds = bt.from_pydict({"x": [1]}).filter(bt.col("x") > 1)
    ds.write.tfrecord(out)
    assert (tmp_path / "empty.tfrecord").read_bytes() == b""
    assert bt.read.tfrecord(out).count() == 0


def test_tfrecord_refuses_what_an_example_cannot_hold(tmp_path):
    _requires_crc()
    out = str(tmp_path / "bad.tfrecord")
    with pytest.raises(SchemaError, match="cannot write column 'meta'"):
        bt.from_pydict({"meta": [{"k": 1}]}).write.tfrecord(out)
    with pytest.raises(SchemaError, match="exactly one binary column"):
        bt.from_pydict({"a": [b"x"], "b": [b"y"]}).write.tfrecord(out, record_format="raw")
    with pytest.raises(FormatError, match="record_format"):
        TFRecordSink(record_format="proto")
