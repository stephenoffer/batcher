"""Training-loader failure modes that a green suite could not see.

Each test here pins a defect that was silent: a batch width that changed mid-epoch, rows
dropped throughout a stream rather than at its tail, a column that vanished from the batch
dict partway through, a rank that crashed because it legitimately drew no rows, and a shard
loader whose shuffle made its own cache useless.
"""

from __future__ import annotations

import warnings

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.io.formats.ml.shards import write_shards
from batcher.ml.loader import (
    iter_torch_batches,
    shard_stream_loader,
    stream_loader,
    streaming_split,
)
from batcher.ml.loader.sharding import gather_rank_shard


def _ds(n: int) -> bt.Dataset:
    return bt.from_pydict({"x": list(range(n)), "y": [float(i) for i in range(n)]})


def _shuffled(source_rows: int, buffer_rows: int, out_rows: int):
    """The local-shuffle stage over batches that do not divide the buffer evenly.

    Exercised at this level on purpose. A block is cut when it reaches `buffer_rows` **or**
    `_SHUFFLE_BLOCK_MAX_BYTES`, and the byte cap is the one that bites in training: it is
    there precisely because 50,000 decoded images is ~30 GB. A byte-cut block is not a whole
    number of output batches, which is the shape reproduced here with a few hundred rows
    instead of a few gigabytes.
    """
    from batcher.ml.loader.lazy import _shuffle_to_numpy

    batches = [
        pa.record_batch({"x": np.arange(i, i + source_rows, dtype=np.int64)})
        for i in range(0, 10 * source_rows, source_rows)
    ]
    return list(_shuffle_to_numpy(iter(batches), buffer_rows, out_rows, 7, None))


def test_local_shuffle_yields_uniform_batches_not_a_short_one_per_block():
    # The shuffle emitted each block in `out_rows` chunks and let the block's remainder go
    # out as its own short batch, so the batch width changed every few steps mid-epoch.
    # Only the very last batch of the whole stream may be short.
    widths = [len(b["x"]) for b in _shuffled(source_rows=100, buffer_rows=250, out_rows=64)]
    assert widths[:-1] == [64] * (len(widths) - 1), widths
    assert sum(widths) == 1000


def test_local_shuffle_conserves_every_row_across_block_boundaries():
    # Carrying a block's remainder into the next block must not drop or duplicate it.
    rows = sorted(
        int(v) for b in _shuffled(source_rows=100, buffer_rows=250, out_rows=64) for v in b["x"]
    )
    assert rows == list(range(1000))


def test_drop_last_with_a_local_shuffle_drops_only_the_tail():
    # `drop_last` discards short batches. While the shuffle emitted one short batch per
    # block, that discarded rows throughout the epoch rather than a single ragged tail.
    from batcher.ml.loader.lazy import _full_batches_only

    kept = sum(
        len(b["x"])
        for b in _full_batches_only(
            iter(_shuffled(source_rows=100, buffer_rows=250, out_rows=64)), 64
        )
    )
    assert kept == 960, "drop_last must cost only the final partial batch (1000 -> 15*64)"


def test_a_nullable_boolean_column_does_not_vanish_mid_epoch():
    # A boolean column converted cleanly until the first batch that happened to hold a null,
    # and then disappeared from the yielded dict — a KeyError on a key the loop had been
    # reading all epoch. Tensorizability is a property of the type, not of this batch's rows.
    ds = bt.from_pydict({"x": list(range(8)), "flag": [True] * 4 + [None] * 4})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        keys = [sorted(b.keys()) for b in iter_torch_batches(ds, batch_size=4, device=None)]
    assert keys == [["flag", "x"], ["flag", "x"]], keys


def test_the_widening_of_a_nullable_column_is_announced():
    ds = bt.from_pydict({"label": [1, 2, None, 4]})
    with pytest.warns(UserWarning, match="nullable .* has no NumPy/tensor equivalent"):
        list(iter_torch_batches(ds, batch_size=4, device=None))


def test_a_dropped_string_column_is_announced_once():
    ds = bt.from_pydict({"x": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})
    with pytest.warns(UserWarning, match=r"dropping non-numeric column\(s\) \['name'\]"):
        batches = list(iter_torch_batches(ds, batch_size=2, device=None))
    assert [sorted(b) for b in batches] == [["x"], ["x"]]


def test_a_rank_that_draws_no_rows_yields_an_empty_epoch_rather_than_crashing():
    # world_size above the corpus size trims the epoch to zero usable positions, so every
    # rank's index list is empty. Building the (empty) shard raised a KeyError from pyarrow.
    table, order = gather_rank_shard(
        _ds(3),
        num_rows=3,
        world_size=4,
        rank=0,
        epoch=0,
        seed=0,
        shuffle=True,
        drop_last=True,
        global_consumed=0,
    )
    assert table.num_rows == 0
    assert table.schema.names == ["x", "y"]
    assert len(order) == 0
    loader = stream_loader(_ds(3), batch_size=2, world_size=4, rank=0)
    assert list(loader) == []
    assert len(loader) == 0


def test_the_shard_loader_covers_the_epoch_exactly_once_per_rank(tmp_path):
    write_shards(pa.table({"id": np.arange(240, dtype=np.int64)}), str(tmp_path), rows_per_shard=16)
    world = 4
    per_rank = [
        [
            int(v)
            for b in shard_stream_loader(
                str(tmp_path), batch_size=5, world_size=world, rank=r, seed=1, cache_size=4
            )
            for v in b["id"].tolist()
        ]
        for r in range(world)
    ]
    assert len({len(ids) for ids in per_rank}) == 1, "ranks must stay balanced for DDP"
    assert sorted(v for ids in per_rank for v in ids) == list(range(240))


def test_the_shard_loader_sample_order_does_not_depend_on_cache_size(tmp_path):
    # The shuffle block is derived from how the corpus was written, never from a caching
    # knob: two ranks configured with different caches must still train on the same order.
    write_shards(pa.table({"id": np.arange(120, dtype=np.int64)}), str(tmp_path), rows_per_shard=8)
    read = lambda cache: [  # noqa: E731
        int(v)
        for b in shard_stream_loader(str(tmp_path), batch_size=4, seed=3, cache_size=cache)
        for v in b["id"].tolist()
    ]
    assert read(2) == read(9)


def test_the_shard_loader_warns_when_the_shuffle_window_exceeds_the_cache(tmp_path):
    write_shards(pa.table({"id": np.arange(120, dtype=np.int64)}), str(tmp_path), rows_per_shard=8)
    with pytest.warns(UserWarning, match="spans more shards than cache_size"):
        shard_stream_loader(str(tmp_path), batch_size=4, cache_size=2, shuffle_block_size=64)


def test_writing_shards_from_a_dataset_round_trips_through_the_loader(tmp_path):
    # The public on-ramp: `ds.ml.write_shards` is what makes the larger-than-memory loader
    # reachable without importing `batcher.io.formats.ml` by hand.
    index = _ds(100).ml.write_shards(str(tmp_path), rows_per_shard=16)
    assert index.total_rows == 100
    assert index.shard_rows == (16, 16, 16, 16, 16, 16, 4)
    seen = [
        int(v)
        for b in shard_stream_loader(str(tmp_path), batch_size=10, seed=2, cache_size=8)
        for v in b["x"].tolist()
    ]
    assert sorted(seen) == list(range(100))
    # And the shards stay ordinary Arrow IPC, readable relationally.
    assert bt.read.arrow(f"{tmp_path}/*.arrow").count() == 100


def test_an_abandoned_streaming_split_winds_its_reader_down():
    # A training script that early-stops, raises, or just stops iterating used to strand the
    # fan-out producer on a full queue for the life of the process, holding the engine stream
    # open and queue_depth * world_size batches resident — one leak per loader built.
    import gc
    import threading
    import time

    def _producers() -> list[str]:
        return [t.name for t in threading.enumerate() if "loader-producer" in t.name]

    assert _producers() == []
    ranks = streaming_split(
        bt.from_pydict({"x": list(range(10_000))}),
        world_size=2,
        batch_size=8,
        device=None,
        queue_depth=2,
    )
    next(ranks[0])  # start rank 0, never touch rank 1, then walk away from both
    del ranks
    gc.collect()
    for _ in range(50):
        if not _producers():
            break
        time.sleep(0.1)
    assert _producers() == [], "the fan-out reader outlived every consumer"


def _ids(loader) -> list[int]:
    return [int(v) for b in loader for v in b["x"].tolist()]


def test_set_epoch_reshuffles_both_indexed_loaders(tmp_path):
    # `DistributedSampler.set_epoch` is the call every DDP loop already makes once per epoch.
    # Without it a caller had to rebuild the loader with `epoch=` by hand, and the ones who
    # did not replayed one order every epoch — which costs convergence, silently.
    _ds(120).ml.write_shards(str(tmp_path), rows_per_shard=16)
    for loader in (
        stream_loader(_ds(120), batch_size=8, seed=4),
        shard_stream_loader(str(tmp_path), batch_size=8, seed=4, cache_size=8),
    ):
        first = _ids(loader)
        loader.set_epoch(1)
        second = _ids(loader)
        assert sorted(first) == sorted(second), "an epoch must still cover the corpus"
        assert first != second, f"{type(loader).__name__} replayed epoch 0's order"
        assert loader.state_dict()["epoch"] == 1


def test_state_dict_resumes_mid_epoch_with_nothing_repeated_or_skipped(tmp_path):
    _ds(120).ml.write_shards(str(tmp_path), rows_per_shard=16)
    for build in (
        lambda **kw: stream_loader(_ds(120), batch_size=8, seed=6, **kw),
        lambda **kw: shard_stream_loader(str(tmp_path), batch_size=8, seed=6, cache_size=8, **kw),
    ):
        loader = build()
        whole = _ids(loader)
        loader.set_epoch(0)
        seen: list[int] = []
        for i, batch in enumerate(loader):
            seen.extend(int(v) for v in batch["x"].tolist())
            if i == 4:  # checkpoint after five steps
                break
        state = loader.state_dict()
        assert state["global_consumed"] == 5 * 8, state
        resumed = build()
        resumed.load_state_dict(state)
        assert seen + _ids(resumed) == whole, "resume must continue the same epoch exactly"


def test_the_loaders_report_their_remaining_length_after_a_resume(tmp_path):
    _ds(120).ml.write_shards(str(tmp_path), rows_per_shard=16)
    for loader in (
        stream_loader(_ds(120), batch_size=8, seed=6),
        shard_stream_loader(str(tmp_path), batch_size=8, seed=6, cache_size=8),
    ):
        assert len(loader) == 15
        loader.load_state_dict({"epoch": 0, "global_consumed": 40})
        assert len(loader) == 10, f"{type(loader).__name__} reported a stale step count"


def test_a_misnamed_column_is_refused_at_the_edge_by_every_loader(tmp_path):
    # A name that is not a column used to surface as a bare pyarrow KeyError on the first
    # `next()` — after the engine had begun reading, and with no suggestion of what was
    # meant. `columns=` is the most-typed argument these loaders take.
    from batcher._internal.errors import ColumnNotFoundError

    _ds(40).ml.write_shards(str(tmp_path), rows_per_shard=8)
    for build in (
        lambda: shard_stream_loader(str(tmp_path), batch_size=4, columns=["nope"]),
        lambda: stream_loader(_ds(40), batch_size=4, columns=["nope"]),
        lambda: iter_torch_batches(_ds(40), batch_size=4, device=None, columns=["nope"]),
    ):
        with pytest.raises(ColumnNotFoundError, match="Unknown column 'nope'"):
            result = build()
            next(iter(result))  # the lazy path raises on construction; consume to be sure


@pytest.mark.parametrize(("world_size", "rank"), [(2, 5), (0, 0), (1, -1), (4, 4)], ids=str)
def test_a_bad_rank_placement_fails_at_construction(tmp_path, world_size, rank):
    # In DDP a rank that raises mid-epoch while the others wait at the all-reduce is a
    # *hang*, not a failure — an hour of wall clock before anyone reads the traceback.
    _ds(40).ml.write_shards(str(tmp_path), rows_per_shard=8)
    with pytest.raises(PlanError):
        shard_stream_loader(str(tmp_path), batch_size=4, world_size=world_size, rank=rank)
    with pytest.raises(PlanError):
        stream_loader(_ds(40), batch_size=4, world_size=world_size, rank=rank)


def test_a_directory_that_is_not_a_corpus_says_so(tmp_path):
    from batcher._internal.errors import FormatError

    with pytest.raises(FormatError, match="not a training-shard corpus"):
        shard_stream_loader(str(tmp_path), batch_size=4)
    (tmp_path / "index.json").write_text('{"a": 1}')
    with pytest.raises(FormatError, match="not a shard manifest"):
        shard_stream_loader(str(tmp_path), batch_size=4)


def test_a_checkpoint_is_refused_against_a_corpus_it_did_not_come_from(tmp_path):
    # `global_consumed` is a *position*. Replay it against a different corpus or a different
    # seed and the position still resolves, but to different samples — so the epoch both
    # repeats and skips, invisibly, and a loss curve will not say so.
    grown = tmp_path / "grown"
    original = tmp_path / "original"
    _ds(120).ml.write_shards(str(original), rows_per_shard=8)
    _ds(200).ml.write_shards(str(grown), rows_per_shard=8)

    loader = shard_stream_loader(str(original), batch_size=8, seed=7)
    for step, _ in enumerate(loader):
        if step == 2:
            break
    checkpoint = loader.state_dict()
    assert checkpoint == {"epoch": 0, "global_consumed": 24, "num_samples": 120, "seed": 7}

    with pytest.raises(PlanError, match="num_samples"):
        shard_stream_loader(str(grown), batch_size=8, seed=7).load_state_dict(checkpoint)
    with pytest.raises(PlanError, match="seed"):
        shard_stream_loader(str(original), batch_size=8, seed=99).load_state_dict(checkpoint)
    with pytest.raises(PlanError, match="num_samples"):
        stream_loader(_ds(200), batch_size=8, seed=7).load_state_dict(checkpoint)

    # The matching loader still accepts it, and a checkpoint predating these fields does too.
    resumed = shard_stream_loader(str(original), batch_size=8, seed=7)
    resumed.load_state_dict(checkpoint)
    assert len(resumed) == 12
    legacy = shard_stream_loader(str(original), batch_size=8, seed=7)
    legacy.load_state_dict({"epoch": 0, "global_consumed": 24})
    assert len(legacy) == 12
