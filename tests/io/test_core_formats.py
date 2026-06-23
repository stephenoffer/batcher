"""Core (pyarrow-native) Phase-3 format coverage: ParquetDataset, text, binary,
numpy. No optional dependencies; runs without the native engine.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

import batcher as bt
from batcher.io.formats.ml.numpy import NumpySource
from batcher.io.formats.structured.parquet import ParquetDatasetSource
from batcher.io.formats.unstructured.binary import BinarySource
from batcher.io.formats.unstructured.text import TextSource


def test_parquet_dataset_recovers_hive_partitions(tmp_path):
    out = str(tmp_path / "part")
    bt.from_arrow(pa.table({"k": [0, 0, 1, 1], "v": [1, 2, 3, 4]})).write.parquet(
        out, partition_by=["k"]
    )
    ds = ParquetDatasetSource(out)
    assert set(ds.schema().names) == {"k", "v"}
    assert ds.row_count() == 4
    table = pa.Table.from_batches(ds.read())
    assert sorted(table.column("v").to_pylist()) == [1, 2, 3, 4]
    assert sorted(set(table.column("k").to_pylist())) == [0, 1]


def test_text_line_mode(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("l1\nl2\nl3\n")
    src = TextSource(str(f))
    table = pa.Table.from_batches(src.read())
    assert table.column_names == ["path", "line_number", "text"]
    assert table.column("text").to_pylist() == ["l1", "l2", "l3"]
    assert table.column("line_number").to_pylist() == [1, 2, 3]


def test_text_file_mode(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello\nworld\n")
    table = pa.Table.from_batches(TextSource(str(f), mode="file").read())
    assert table.column_names == ["path", "text"]
    assert table.num_rows == 1
    assert table.column("text")[0].as_py() == "hello\nworld\n"


def test_binary_source_columns_and_mime(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"\x00\x01\x02")
    table = pa.Table.from_batches(BinarySource(str(f)).read())
    assert table.column_names == ["uri", "bytes", "size", "mime"]
    assert table.column("bytes")[0].as_py() == b"\x00\x01\x02"
    assert table.column("size")[0].as_py() == 3


def test_numpy_ndarray_to_fixed_size_list(tmp_path):
    f = tmp_path / "a.npy"
    np.save(f, np.arange(12).reshape(4, 3))
    table = pa.Table.from_batches(NumpySource(str(f)).read())
    assert table.num_rows == 4
    assert pa.types.is_fixed_size_list(table.schema.field("data").type)
    assert table.column("data")[0].as_py() == [0, 1, 2]


def test_binary_split_cover(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.bin").write_bytes(bytes([i]))
    src = BinarySource(str(tmp_path), suffix=".bin", batch_files=2)
    whole = pa.Table.from_batches(src.read())
    cover = pa.concat_tables(
        [pa.Table.from_batches(s.read(), schema=src.schema()) for s in src.splits()]
    )
    assert cover.num_rows == whole.num_rows == 5
