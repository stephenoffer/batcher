"""The shard layer's two hot-path contracts: a streaming writer and a vectorized gather.

Both halves of `batcher.io.formats.ml.shards` sit under a training loop, so both are held
here to properties a training loop depends on: the writer must not need the corpus in
memory, and `take` must return exactly the rows asked for, in the order asked for, in one
chunk — fast enough to feed an accelerator.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.io.formats.ml.shards import ShardReader, read_shard_index, write_shards


def _table(n: int, width: int = 8) -> pa.Table:
    return pa.table(
        {
            "id": np.arange(n, dtype=np.int64),
            "f": pa.FixedSizeListArray.from_arrays(
                pa.array(np.arange(n * width, dtype=np.float32)), width
            ),
        }
    )


def test_writer_publishes_shards_before_consuming_the_whole_input(tmp_path):
    # The writer used to do `pa.Table.from_batches(list(batches))` — pulling the ENTIRE input
    # into one table before writing anything, so a larger-than-RAM corpus needed
    # larger-than-RAM. The observable consequence is that no shard file exists until the last
    # batch has been read. Streaming means shards land while the source is still producing.
    shards_seen_midway: list[int] = []

    def _batches():
        for i in range(0, 200, 4):
            yield pa.record_batch({"id": np.arange(i, i + 4, dtype=np.int64)})
            if i == 100:  # halfway through the source
                shards_seen_midway.append(len(list(tmp_path.glob("shard-*.arrow"))))

    index = write_shards(_batches(), str(tmp_path), rows_per_shard=8)
    assert index.total_rows == 200
    assert index.shard_rows == tuple([8] * 25)
    assert shards_seen_midway[0] >= 10, (
        f"only {shards_seen_midway[0]} shards written after 104 of 200 rows; "
        "the writer is buffering the whole corpus"
    )
    # And the rows are all there, in order, once it is done.
    assert ShardReader(str(tmp_path), cache_size=32).take(list(range(200))).column(
        "id"
    ).to_pylist() == list(range(200))


def test_index_carries_the_schema_so_metadata_needs_no_data_read(tmp_path):
    write_shards(_table(20), str(tmp_path), rows_per_shard=8)
    index = read_shard_index(str(tmp_path))
    assert index.schema is not None
    assert index.schema.names == ["id", "f"]
    # The tensor-column type survives the round trip, which a field-name encoding would lose.
    assert pa.types.is_fixed_size_list(index.schema.field("f").type)
    # An empty gather answers from the index, with the columns intact.
    empty = ShardReader(str(tmp_path)).take([])
    assert empty.num_rows == 0
    assert empty.schema.names == ["id", "f"]


def test_writer_rejects_batches_that_disagree_on_schema(tmp_path):
    batches = [
        pa.record_batch({"id": pa.array([1, 2], type=pa.int64())}),
        pa.record_batch({"id": pa.array(["a", "b"])}),
    ]
    with pytest.raises(ValueError, match="differing schemas"):
        write_shards(batches, str(tmp_path), rows_per_shard=2)


@pytest.mark.parametrize("rows_per_shard", [1, 3, 7, 64])
def test_take_returns_exactly_the_requested_rows_in_order(tmp_path, rows_per_shard):
    n = 64
    write_shards(_table(n), str(tmp_path), rows_per_shard=rows_per_shard)
    reader = ShardReader(str(tmp_path), cache_size=2)
    rng = np.random.default_rng(0)
    wanted = rng.integers(0, n, 40).tolist()
    out = reader.take(wanted)
    # Order-sensitive on purpose: the gather reorders shard-grouped rows back, and an
    # order-independent comparison could not see that inverse permutation being wrong.
    assert out.column("id").to_pylist() == wanted
    assert out.column("f").to_pylist() == [list(range(i * 8, i * 8 + 8)) for i in wanted], (
        "the feature row must follow its id"
    )


def test_take_returns_one_chunk_per_column(tmp_path):
    # Building the result as one single-row table per sample left `batch_size` chunks per
    # column, so every downstream tensor conversion paid for the fragmentation again.
    write_shards(_table(500), str(tmp_path), rows_per_shard=50)
    out = ShardReader(str(tmp_path), cache_size=16).take(list(range(0, 500, 3)))
    assert out.column("id").num_chunks == 1
    assert out.column("f").num_chunks == 1


def test_take_handles_duplicates_and_rejects_out_of_range(tmp_path):
    write_shards(_table(20), str(tmp_path), rows_per_shard=8)
    reader = ShardReader(str(tmp_path))
    # Duplicates are legitimate: a padded (`drop_last=False`) epoch tail repeats samples.
    assert reader.take([3, 3, 19, 3]).column("id").to_pylist() == [3, 3, 19, 3]
    with pytest.raises(IndexError, match=r"out of range \[0, 20\)"):
        reader.take([20])
    with pytest.raises(IndexError):
        reader.take([-1])


def test_take_batch_is_one_contiguous_record_batch(tmp_path):
    write_shards(_table(30), str(tmp_path), rows_per_shard=7)
    reader = ShardReader(str(tmp_path), cache_size=8)
    batch = reader.take_batch([5, 1, 29, 12])
    assert isinstance(batch, pa.RecordBatch)
    assert batch.column("id").to_pylist() == [5, 1, 29, 12]
    assert reader.take_batch([]).num_rows == 0
    assert reader.take_batch([]).schema.names == ["id", "f"]


def test_reader_cache_stays_bounded_while_serving_a_shuffled_read(tmp_path):
    write_shards(_table(200), str(tmp_path), rows_per_shard=10)  # 20 shards
    reader = ShardReader(str(tmp_path), cache_size=3)
    rng = np.random.default_rng(1)
    for _ in range(10):
        wanted = rng.integers(0, 200, 32).tolist()
        assert reader.take(wanted).column("id").to_pylist() == wanted
        assert len(reader._cache) <= 3
    reader.clear_cache()
    assert len(reader._cache) == 0


def test_a_shard_corpus_reads_back_as_a_relation(tmp_path):
    # The return leg: the loader reads the corpus by sample index, but every question asked
    # *around* a training run — class balance, null labels, does this match the source
    # table — is relational, and used to need the corpus re-derived from scratch.
    import batcher as bt

    ds = bt.from_pydict({"x": list(range(100)), "lab": ["a" if i % 2 else "b" for i in range(100)]})
    ds.ml.write_shards(str(tmp_path), rows_per_shard=16)

    corpus = bt.read.training_shards(str(tmp_path))
    assert corpus.schema.names == ["x", "lab"]
    # Order-sensitive: a scan must reproduce the order the corpus was written in, which is
    # what makes a row index mean the same thing to the loader and to a query.
    assert corpus.to_pydict()["x"] == list(range(100))
    counted = corpus.group_by("lab").agg(n=bt.col("x").count()).sort("lab")
    assert counted.to_pydict() == {"lab": ["a", "b"], "n": [50, 50]}


def test_the_row_count_comes_from_the_index_not_a_scan(tmp_path):
    import batcher as bt
    from batcher.io.formats.base import SOURCES

    bt.from_pydict({"x": list(range(50))}).ml.write_shards(str(tmp_path), rows_per_shard=8)
    source = SOURCES.get("training_shards")(str(tmp_path))
    assert source.row_count() == 50
    stats = source.statistics()
    assert (stats.row_count, stats.exact_rows) == (50, True)
    # Deleting the shards leaves the index — a metadata answer must not have touched them.
    for shard in tmp_path.glob("shard-*.arrow"):
        shard.unlink()
    assert SOURCES.get("training_shards")(str(tmp_path)).row_count() == 50


def test_splits_cover_the_corpus_once_and_survive_a_pickle(tmp_path):
    # A split is the unit of distributed read parallelism: it must carry locators only, so
    # it pickles to a worker that reads its slice straight from storage.
    import pickle

    import batcher as bt
    from batcher.io.formats.base import SOURCES

    bt.from_pydict({"x": list(range(100))}).ml.write_shards(str(tmp_path), rows_per_shard=16)
    splits = SOURCES.get("training_shards")(str(tmp_path)).splits()
    assert len(splits) == 7  # one per shard
    rows = [
        v
        for split in splits
        for batch in pickle.loads(pickle.dumps(split)).read()
        for v in batch.column("x").to_pylist()
    ]
    assert sorted(rows) == list(range(100)), "splits must cover the corpus exactly once"


def test_projection_reaches_the_shard_reader(tmp_path):
    import batcher as bt
    from batcher.io.formats.base import SOURCES

    bt.from_pydict({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]}).ml.write_shards(
        str(tmp_path), rows_per_shard=2
    )
    source = SOURCES.get("training_shards")(str(tmp_path))
    assert [b.schema.names for b in source.read(["y"])] == [["y"], ["y"]]


def test_a_shard_directory_is_detected_without_naming_the_format(tmp_path):
    import batcher as bt
    from batcher.io.detect import _training_shards_at

    bt.from_pydict({"x": [1, 2, 3]}).ml.write_shards(str(tmp_path), rows_per_shard=2)
    assert _training_shards_at(str(tmp_path)) == "training_shards"
    assert bt.read(str(tmp_path)).count() == 3


def test_a_plain_directory_holding_an_index_json_is_not_claimed(tmp_path):
    # `index.json` is a name any directory may carry, so the format is claimed only when a
    # shard it would be describing is there too. Claiming on the manifest alone would
    # hijack an ordinary directory read.
    from batcher.io.detect import _training_shards_at

    (tmp_path / "index.json").write_text("{}")
    (tmp_path / "part-0.parquet").write_bytes(b"")
    assert _training_shards_at(str(tmp_path)) is None
