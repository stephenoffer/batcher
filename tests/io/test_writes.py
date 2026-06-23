"""Partitioned + manifest write coverage for the IO sink bases.

Exercises single-file writes (returning a manifest), Hive-partitioned directory
writes (partition columns dropped from the data, encoded in the path), and the
round-trip back through the matching source. Runs without the native engine.
"""

from __future__ import annotations

import os

import pyarrow as pa

from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.io.sink import ParquetSink
from batcher.io.source import ParquetSource


def _sorted_rows(table: pa.Table) -> list[dict]:
    return sorted(table.to_pylist(), key=lambda r: tuple(str(r[c]) for c in table.column_names))


def test_single_file_write_returns_manifest(tmp_path):
    table = pa.table({"a": [1, 2, 3]})
    path = str(tmp_path / "out.parquet")
    manifest = WriteManifest((ParquetSink().write(table, path),))
    assert isinstance(manifest.files[0], WrittenFile)
    assert manifest.total_rows == 3
    assert manifest.num_files == 1
    assert ParquetSource(path).row_count() == 3


def test_hive_partitioned_write_layout_and_roundtrip(tmp_path):
    table = pa.table(
        {
            "country": ["US", "US", "CA", "CA"],
            "v": [1, 2, 3, 4],
        }
    )
    out = str(tmp_path / "t")
    files = ParquetSink().write_partitioned(table, out, partition_by=["country"])
    manifest = WriteManifest(tuple(files))

    assert manifest.num_files == 2
    assert manifest.total_rows == 4
    # Hive directory layout, partition column dropped from data files.
    assert os.path.isdir(os.path.join(out, "country=US"))
    assert os.path.isdir(os.path.join(out, "country=CA"))
    us = ParquetSource(os.path.join(out, "country=US"))
    assert us.schema().names == ["v"]  # partition col not in data
    assert us.row_count() == 2
    # Partition values are recorded on the manifest.
    assert {tuple(f.partition_values.items()) for f in files} == {
        (("country", "US"),),
        (("country", "CA"),),
    }


def test_multi_column_partition(tmp_path):
    table = pa.table(
        {
            "y": [2023, 2023, 2024],
            "m": [1, 2, 1],
            "v": [10, 20, 30],
        }
    )
    out = str(tmp_path / "p")
    files = ParquetSink().write_partitioned(table, out, partition_by=["y", "m"])
    assert len(files) == 3
    assert os.path.isfile(os.path.join(out, "y=2023", "m=1", "part-00000.parquet"))
    recovered = pa.concat_tables(
        [pa.Table.from_batches(ParquetSource(f.path).read()) for f in files]
    )
    assert recovered.num_rows == 3
