"""Regression tests for two IO defects found in the bug hunt.

Both are "the split path disagrees with the whole-source read" defects — the kind
that only surfaces on the distributed / multi-split execution path:

- An NDJSON `LineRangeSplit` inferred each byte range's schema independently, so an
  all-integer range of a column parsed as ``int64`` while a range holding a float
  parsed it as ``double`` (and a field absent from a range vanished from that range's
  schema). Batches from different ranges of one file then failed to concatenate, and
  disagreed with the source's advertised schema. `CSVRangeSplit` already pins
  ``column_types``; NDJSON did not.
- A Hive-partitioned Parquet dataset read via its distributed `PartitionDirSplit`s
  recovered the partition value from the RAW ``col=val`` directory name and never
  URL-decoded it, so ``x/y`` came back as ``x%2Fy`` — while the single-node
  `pyarrow.dataset` `read()` path decoded it correctly. A distributed read produced
  different data than a single-node read of the same directory.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

from batcher.io.formats.semistructured.json import JSONSource
from batcher.io.formats.structured.parquet.dataset import ParquetDatasetSource
from batcher.io.formats.structured.parquet.sink import ParquetSink

pytestmark = pytest.mark.unit


def test_json_range_splits_share_the_file_schema(tmp_path):
    # First half of the rows carry an integer 'v'; the second half a float 'v'. The
    # whole-file schema unifies to double; a range holding only the integer rows would
    # infer int64 without a pinned schema.
    path = str(tmp_path / "data.json")
    lines = [f'{{"v": {i + 1}}}' for i in range(2000)]
    lines += [f'{{"v": {i + 0.5}}}' for i in range(2000)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    src = JSONSource(path)
    size = os.path.getsize(path)
    splits = src.splits(target_size=size // 4)
    assert len(splits) > 1  # the file really did fan into ranges

    batches = [b for s in splits for b in s.read()]
    # Every range must produce the source's advertised schema, so the batches
    # concatenate (they used to raise here) and no value is silently retyped.
    assert {b.schema.field("v").type for b in batches} == {src.schema().field("v").type}
    table = pa.Table.from_batches(batches)
    assert table.num_rows == 4000
    assert pa.types.is_floating(table.schema.field("v").type)


def test_json_range_split_missing_field_keeps_null_column(tmp_path):
    # 'b' is present only in the first rows; a later range lacks it entirely. With a
    # pinned schema the range still yields column 'b' (as nulls), matching every other
    # range so the splits concatenate.
    path = str(tmp_path / "data.json")
    lines = [f'{{"a": {i}, "b": {i}}}' for i in range(1500)]
    lines += [f'{{"a": {i}}}' for i in range(1500)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    src = JSONSource(path)
    size = os.path.getsize(path)
    splits = src.splits(target_size=size // 4)
    assert len(splits) > 1

    batches = [b for s in splits for b in s.read()]
    assert {tuple(b.schema.names) for b in batches} == {("a", "b")}
    table = pa.Table.from_batches(batches)
    assert table.num_rows == 3000


def test_partition_dir_splits_url_decode_like_the_dataset_read(tmp_path):
    # Values needing URL-encoding on write must decode identically on both read paths.
    values = ["x/y", "a=b", "hello world", "p%q", "normal", None]
    table = pa.table({"c": values, "v": list(range(len(values)))})
    ParquetSink().write_partitioned(table, str(tmp_path), partition_by=["c"])

    src = ParquetDatasetSource(str(tmp_path))
    single = pa.Table.from_batches(src.read())
    distributed = pa.Table.from_batches([b for s in src.splits() for b in s.read()])

    def pairs(t: pa.Table) -> list[tuple[object, object]]:
        return sorted(
            zip(t.column("c").to_pylist(), t.column("v").to_pylist(), strict=True),
            key=lambda kv: (kv[0] is None, kv[0], kv[1]),
        )

    # The distributed split path must produce byte-identical data to the single-node
    # read (single-node == distributed). Pre-fix it returned 'x%2Fy' etc.
    assert pairs(distributed) == pairs(single)
    got = {c for c, _ in pairs(distributed)}
    assert {"x/y", "a=b", "hello world", "p%q", "normal", None} == got
