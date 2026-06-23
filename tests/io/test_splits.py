"""Split-cover correctness for the IO source bases.

A source's `splits()` must be a disjoint, exhaustive cover of its rows: reading
every split and concatenating must equal reading the whole source. This is the
invariant the distributed read path relies on (one task per split, mergeable
combine), so it is tested directly at the source layer without the engine.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from batcher.io.source import CSVSource, InMemorySource, JSONSource, ParquetSource
from batcher.io.splits import RowGroupSplit, WholeSourceSplit


def _cover(source) -> pa.Table:
    """Concatenate the rows of every split of `source`."""
    return pa.concat_tables(
        [pa.Table.from_batches(s.read(), schema=source.schema()) for s in source.splits()]
    )


def test_parquet_row_group_splits_cover_whole(tmp_path):
    path = str(tmp_path / "m.parquet")
    table = pa.table({"a": list(range(1000)), "b": [f"r{i}" for i in range(1000)]})
    pq.write_table(table, path, row_group_size=100)  # 10 row groups

    source = ParquetSource(path)
    splits = source.splits()
    assert len(splits) == 10
    assert all(isinstance(s, RowGroupSplit) for s in splits)
    assert _cover(source).equals(table)


def test_parquet_splits_pack_to_target(tmp_path):
    path = str(tmp_path / "m.parquet")
    pq.write_table(pa.table({"a": list(range(1000))}), path, row_group_size=100)
    source = ParquetSource(path)
    # A target larger than the whole file packs all row-groups into one split.
    packed = source.splits(target_size=10_000_000)
    assert len(packed) == 1
    assert _cover(source).num_rows == 1000


def test_parquet_split_respects_projection(tmp_path):
    path = str(tmp_path / "m.parquet")
    pq.write_table(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), path)
    split = ParquetSource(path).splits()[0]
    out = pa.Table.from_batches(split.read(["a"]))
    assert out.column_names == ["a"]


def test_csv_default_one_split_per_file(tmp_path):
    path = str(tmp_path / "d.csv")
    (tmp_path / "d.csv").write_text("a,b\n1,x\n2,y\n")
    source = CSVSource(path)
    assert len(source.splits()) == 1
    assert _cover(source).equals(pa.Table.from_batches(source.read()))


def test_json_default_one_split_per_file(tmp_path):
    path = str(tmp_path / "d.json")
    (tmp_path / "d.json").write_text('{"a":1}\n{"a":2}\n')
    source = JSONSource(path)
    assert len(source.splits()) == 1
    assert _cover(source).equals(pa.Table.from_batches(source.read()))


def test_parquet_dataset_distributed_listing_splits(tmp_path):
    """A partitioned dataset yields one split per top-level partition directory
    (distributed listing — each worker lists only its own subtree, the driver never
    lists the whole tree). Splits are picklable, recover the partition column, and
    cover the whole table exactly.
    """
    import pickle

    import batcher as bt
    from batcher.io.formats.structured.parquet import (
        ParquetDatasetSource,
        PartitionDirSplit,
    )

    out = str(tmp_path / "p")
    bt.from_arrow(
        pa.table({"k": [i % 5 for i in range(200)], "v": list(range(200))})
    ).write.parquet(out, partition_by=["k"])
    source = ParquetDatasetSource(out)
    splits = source.splits()
    assert len(splits) == 5  # one per partition dir, not a whole-source split
    assert all(isinstance(s, PartitionDirSplit) for s in splits)
    # Picklable (ships to a worker as locators + the schema only).
    assert all(pickle.loads(pickle.dumps(s)).subdir for s in splits)
    cover = pa.concat_tables([pa.Table.from_batches(s.read()) for s in splits])
    assert cover.num_rows == 200
    assert set(cover.schema.names) == {"k", "v"}  # partition column recovered
    assert sorted(set(cover.column("k").to_pylist())) == [0, 1, 2, 3, 4]


def test_parquet_fragment_split_pushes_projection_and_predicate(tmp_path):
    import batcher as bt
    from batcher.io.formats.structured.parquet import ParquetDatasetSource

    out = str(tmp_path / "p")
    bt.from_arrow(pa.table({"k": [0, 0, 1, 1], "v": [1, 2, 3, 4]})).write.parquet(
        out, partition_by=["k"]
    )
    split = ParquetDatasetSource(out).splits()[0]
    batches = split.read(["v"], (bt.col("v") > 1).to_ir())
    table = pa.Table.from_batches(batches, schema=batches[0].schema)
    assert table.column_names == ["v"]
    assert all(v > 1 for v in table.column("v").to_pylist())


def test_ndjson_byte_range_splits_exact_once(tmp_path):
    """A large NDJSON file splits into newline-aligned byte ranges that cover every
    line exactly once (no loss or duplication across split boundaries), so one huge
    file fans across workers.
    """
    import json
    import pickle

    from batcher.io.formats.semistructured.json import JSONSource
    from batcher.io.splits import LineRangeSplit

    path = str(tmp_path / "big.jsonl")
    with open(path, "w") as f:
        for i in range(1000):
            f.write(json.dumps({"id": i, "v": i * 2}) + "\n")

    source = JSONSource(path)
    splits = source.splits(target_size=2000)  # tiny target → many byte ranges
    assert len(splits) > 1
    assert all(isinstance(s, LineRangeSplit) for s in splits)
    assert all(pickle.loads(pickle.dumps(s)).path == path for s in splits)

    cover = pa.concat_tables(
        [pa.Table.from_batches(s.read(), schema=source.schema()) for s in splits]
    )
    assert sorted(cover.column("id").to_pylist()) == list(range(1000))  # exactly once


def test_ndjson_small_file_single_split(tmp_path):
    from batcher.io.formats.semistructured.json import JSONSource
    from batcher.io.splits import FileSplit

    path = str(tmp_path / "small.jsonl")
    (tmp_path / "small.jsonl").write_text('{"a":1}\n{"a":2}\n')
    splits = JSONSource(path).splits()  # below the default split size → one FileSplit
    assert len(splits) == 1
    assert isinstance(splits[0], FileSplit)


def test_csv_byte_range_splits_exact_once(tmp_path):
    """A large CSV splits into newline-aligned byte ranges that cover every data row
    exactly once, with the header prepended to each mid-file range for column names.
    """
    from batcher.io.formats.structured.csv import CSVRangeSplit, CSVSource

    path = str(tmp_path / "big.csv")
    with open(path, "w") as f:
        f.write("id,v\n")
        for i in range(1000):
            f.write(f"{i},{i * 2}\n")

    source = CSVSource(path)
    splits = source.splits(target_size=1500)  # tiny target → many byte ranges
    assert len(splits) > 1
    assert all(isinstance(s, CSVRangeSplit) for s in splits)
    cover = pa.concat_tables(
        [pa.Table.from_batches(s.read(), schema=source.schema()) for s in splits]
    )
    assert set(cover.schema.names) == {"id", "v"}  # header recovered for every range
    assert sorted(cover.column("id").to_pylist()) == list(range(1000))  # exactly once


def test_in_memory_source_whole_split():
    source = InMemorySource(pa.table({"a": [1, 2, 3]}).to_batches())
    splits = source.splits()
    assert len(splits) == 1
    assert isinstance(splits[0], WholeSourceSplit)
    assert pa.Table.from_batches(splits[0].read()).num_rows == 3
