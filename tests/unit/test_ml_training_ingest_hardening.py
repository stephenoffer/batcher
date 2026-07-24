"""Regressions for the training-data ingestion path (`ml/loader`, `ml/streaming_sampler`).

Every test here failed before the fix it pins. The defects share a shape: each one produced a
*plausible* training run — batches arrived, the loss went down, nothing raised — while quietly
corrupting the data the model saw or the memory the job needed.

* pinned staging buffers freed while their asynchronous host-to-device copy was still running;
* string label/id columns vanishing from every batch with no warning and no escape hatch;
* `streaming_split` silently dropping up to ``world_size - 1`` batches per epoch;
* `stream_loader` collecting the whole corpus in every rank, plus a Python list of every index;
* `elastic_shard` materializing the full position list on the resume path;
* every epoch over the same dataset seeing an identical order;
* a ragged final batch reaching DDP;
* `ResumableSampler` replaying its whole stream once per DataLoader worker.

Ordering assertions here are deliberately exact. An order-independent comparison cannot see an
ordering bug, which is the whole failure mode of a shuffled sample sequence.
"""

from __future__ import annotations

import gc
import inspect
import warnings
import weakref
from typing import Any, ClassVar

import pyarrow as pa
import pytest

from batcher.ml.loader.indexed import stream_loader
from batcher.ml.loader.lazy import iter_torch_batches, streaming_split
from batcher.ml.loader.sharding import gather_rank_shard
from batcher.ml.loader.tensors import DeviceMover
from batcher.ml.streaming_sampler import ResumableSampler, elastic_shard, epoch_order

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch", reason="torch not installed")


# --------------------------------------------------------------------------------------
# Fakes: a Dataset-shaped stand-in, so the ingest contract is testable with no engine,
# no GPU and no cluster.
# --------------------------------------------------------------------------------------


class _SpyDataset:
    """The slice of the `Dataset` surface the loaders use, recording how it was consumed."""

    def __init__(self, table: pa.Table, batch_rows: int = 8) -> None:
        self._table = table
        self._batch_rows = batch_rows
        self.collect_calls = 0
        self.iter_calls = 0
        self.rows_streamed = 0

    def collect(self) -> pa.Table:
        self.collect_calls += 1
        return self._table

    def count(self) -> int:
        return self._table.num_rows

    @property
    def columns(self) -> list[str]:
        return list(self._table.column_names)

    def iter_batches(self, batch_size: int | None = None):
        self.iter_calls += 1
        for batch in self._table.to_batches(max_chunksize=batch_size or self._batch_rows):
            self.rows_streamed += batch.num_rows
            yield batch


class _FakeTensor:
    """A tensor-shaped object that records the staging buffers `pin_memory` hands out."""

    staged: ClassVar[list[weakref.ref]] = []

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def pin_memory(self) -> _FakeTensor:
        pinned = _FakeTensor(f"{self.tag}:pinned")
        _FakeTensor.staged.append(weakref.ref(pinned))
        return pinned

    def to(self, device: Any, non_blocking: bool = False) -> _FakeTensor:
        # The device copy is asynchronous: it does NOT read `self` before returning.
        return _FakeTensor(f"{self.tag}:on-{device}")


def _table(n: int = 40) -> pa.Table:
    return pa.table({"id": list(range(n)), "v": [float(i) for i in range(n)]})


def _ids(loader) -> list[int]:
    return [int(x) for batch in loader for x in batch["id"].tolist()]


# --------------------------------------------------------------------------------------
# 1. pin_memory + non_blocking was a use-after-free.
# --------------------------------------------------------------------------------------


def test_pinned_staging_buffer_outlives_the_async_copy():
    """`DeviceMover` must keep the DMA source alive; the naive recipe frees it mid-copy."""
    _FakeTensor.staged = []
    source = _FakeTensor("x")

    # The old recipe, verbatim: nothing holds the pinned tensor once `.to()` returns.
    source.pin_memory().to("cuda", non_blocking=True)
    gc.collect()
    assert _FakeTensor.staged[0]() is None, "the naive recipe should drop the staging buffer"

    _FakeTensor.staged = []
    mover = DeviceMover("cuda", pin_memory=True, depth=2)
    moved = mover({"x": _FakeTensor("x")})
    gc.collect()

    assert moved["x"].tag == "x:pinned:on-cuda"
    # The copy may still be in flight, so the staging buffer must still be reachable.
    assert _FakeTensor.staged[0]() is not None, "staging buffer freed while the copy was live"


def test_staging_buffers_are_retired_once_the_look_ahead_has_passed():
    """The keepalive is bounded — it must not turn into an unbounded pinned-memory leak."""
    _FakeTensor.staged = []
    mover = DeviceMover("cuda", pin_memory=True, depth=2)
    for i in range(6):
        mover({"x": _FakeTensor(str(i))})
    gc.collect()

    live = [ref for ref in _FakeTensor.staged if ref() is not None]
    assert len(live) <= 3, f"pinned buffers accumulating: {len(live)} still held"


def test_device_mover_without_pin_memory_stages_nothing():
    _FakeTensor.staged = []
    moved = DeviceMover("cuda", pin_memory=False)({"x": _FakeTensor("x")})
    assert moved["x"].tag == "x:on-cuda"
    assert _FakeTensor.staged == []


# --------------------------------------------------------------------------------------
# 2 + 8. Non-tensorizable columns vanished silently, with no way to keep them.
# --------------------------------------------------------------------------------------


def test_stream_loader_warns_about_columns_it_cannot_tensorize():
    ds = _SpyDataset(pa.table({"id": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]}))
    with pytest.warns(UserWarning, match="non-tensorizable column"):
        loader = stream_loader(ds, batch_size=2, shuffle=False)
    assert "label" not in next(iter(loader))  # still dropped, but no longer silently


def test_stream_loader_collate_fn_keeps_a_string_column():
    ds = _SpyDataset(pa.table({"id": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]}))
    loader = stream_loader(
        ds, batch_size=2, shuffle=False, collate_fn=lambda t: t.to_pydict(), world_size=1
    )
    batches = list(loader)
    assert batches[0]["label"] == ["a", "b"]  # exact order, not a set comparison
    assert batches[1]["label"] == ["c", "d"]


def test_collate_fn_suppresses_the_drop_warning():
    """A user-supplied collate takes the whole table, so no column is dropped to warn about."""
    ds = _SpyDataset(pa.table({"id": [1, 2], "label": ["a", "b"]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing is dropped, so nothing may be warned about
        loader = stream_loader(ds, batch_size=2, shuffle=False, collate_fn=lambda t: t)
        batches = list(loader)
    assert [b.column_names for b in batches] == [["id", "label"]]


def test_shard_stream_loader_takes_a_collate_fn(tmp_path):
    from batcher.io.formats.ml import write_shards
    from batcher.ml.loader.indexed import shard_stream_loader

    write_shards(
        pa.table({"id": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]}),
        str(tmp_path),
        rows_per_shard=2,
    )
    loader = shard_stream_loader(
        str(tmp_path), batch_size=2, shuffle=False, collate_fn=lambda t: t.to_pydict()
    )
    assert next(iter(loader))["label"] == ["a", "b"]


# --------------------------------------------------------------------------------------
# 3. streaming_split dropped a trailing partial round with no count and no opt-out.
# --------------------------------------------------------------------------------------


def _drain_fleet(splits: list) -> list[list[int]]:
    """Drain every rank concurrently, which the fleet split requires by design."""
    iters = [iter(s) for s in splits]
    out: list[list[int]] = [[] for _ in iters]
    while True:
        advanced = False
        for r, it in enumerate(iters):
            batch = next(it, None)
            if batch is not None:
                out[r].extend(int(v) for v in batch["id"].tolist())
                advanced = True
        if not advanced:
            return out


def test_streaming_split_reports_the_batches_it_drops():
    ds = _SpyDataset(_table(40), batch_rows=40)
    with pytest.warns(UserWarning, match="dropped 2 trailing batch"):
        one = list(streaming_split(ds, 3, rank=0, batch_size=8, device=None))
    assert len(one) == 1  # 5 batches -> one complete round of 3, 2 dropped


def test_streaming_split_drop_last_false_keeps_the_trailing_round():
    ds = _SpyDataset(_table(40), batch_rows=40)
    kept = _drain_fleet(streaming_split(ds, 3, batch_size=8, drop_last=False, device=None))

    assert len({len(rank) for rank in kept}) == 1, "ranks must stay balanced"
    # 5 batches of 8: rounds [0,1,2] and the partial [3,4], completed cyclically as [3,4,3].
    assert kept[0] == list(range(0, 8)) + list(range(24, 32))
    assert kept[2] == list(range(16, 24)) + list(range(24, 32))
    # Every sample is present at least once — nothing is silently discarded.
    assert set(range(40)).issubset({v for rank in kept for v in rank})


def test_streaming_split_single_rank_drop_last_false_matches_the_fleet():
    ds = _SpyDataset(_table(40), batch_rows=40)
    fleet = _drain_fleet(streaming_split(ds, 3, batch_size=8, drop_last=False, device=None))
    for rank in range(3):
        solo = streaming_split(
            _SpyDataset(_table(40), batch_rows=40),
            3,
            rank=rank,
            batch_size=8,
            drop_last=False,
            device=None,
        )
        assert _ids(solo) == fleet[rank]


# --------------------------------------------------------------------------------------
# 4. stream_loader collected the whole corpus per rank, and listed every index.
# --------------------------------------------------------------------------------------


def test_stream_loader_never_collects_the_whole_dataset():
    ds = _SpyDataset(_table(40))
    loader = stream_loader(ds, batch_size=4, world_size=4, rank=1, seed=5)
    _ids(loader)

    assert ds.collect_calls == 0, "the corpus must be streamed, not collected"
    assert ds.rows_streamed == 40, "and streamed exactly once"


def test_a_rank_retains_only_its_own_shard():
    n, world_size = 64, 8
    table, order = gather_rank_shard(
        _SpyDataset(_table(n)),
        num_rows=n,
        world_size=world_size,
        rank=3,
        epoch=0,
        seed=5,
        shuffle=True,
        drop_last=True,
        global_consumed=0,
    )
    # The whole point: 1/world_size of the corpus resident, not all of it.
    assert table.num_rows == n // world_size
    assert len(order) == n // world_size


@pytest.mark.parametrize("world_size,rank", [(1, 0), (3, 2), (4, 1), (8, 7)])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_streamed_shard_reproduces_the_materialized_sample_order(
    world_size, rank, drop_last, shuffle
):
    """The order must be bit-identical to the list-materializing implementation it replaced."""
    n, seed, epoch = 37, 11, 2
    expected = elastic_shard(
        epoch_order(n, epoch=epoch, seed=seed, shuffle=shuffle),
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
    )
    loader = stream_loader(
        _SpyDataset(_table(n), batch_rows=5),
        batch_size=1,
        world_size=world_size,
        rank=rank,
        epoch=epoch,
        seed=seed,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    assert _ids(loader) == expected


def test_streamed_shard_reproduces_the_resume_order():
    n, world_size, rank = 40, 4, 2
    expected = elastic_shard(
        epoch_order(n, seed=3), world_size=world_size, rank=rank, global_consumed=8
    )
    loader = stream_loader(
        _SpyDataset(_table(n), batch_rows=7),
        batch_size=1,
        world_size=world_size,
        rank=rank,
        seed=3,
        global_consumed=8,
    )
    assert _ids(loader) == expected


# --------------------------------------------------------------------------------------
# 5. elastic_shard materialized the full position list on the resume path.
# --------------------------------------------------------------------------------------


class _NoSliceOrder:
    """A lazy order that refuses to be sliced or copied — as `epoch_permutation` effectively is.

    Materializing the epoch's positions means slicing or concatenating the order, so this
    stands in for the corpus that does not fit in driver memory: any attempt to build the whole
    position list raises instead of quietly costing O(num_samples).
    """

    def __init__(self, values: list[int]) -> None:
        self._values = values
        self.reads = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: Any) -> int:
        if isinstance(index, slice):
            raise AssertionError("the whole order was materialized")
        self.reads += 1
        return self._values[index]

    def __add__(self, other: Any) -> Any:
        raise AssertionError("the whole order was materialized")


@pytest.mark.parametrize("drop_last", [True, False])
def test_elastic_shard_builds_only_this_ranks_share(drop_last):
    order = epoch_order(64, seed=4)
    lazy = _NoSliceOrder(list(order))

    got = elastic_shard(lazy, world_size=8, rank=5, global_consumed=16, drop_last=drop_last)

    assert got == elastic_shard(
        list(order), world_size=8, rank=5, global_consumed=16, drop_last=drop_last
    )
    assert lazy.reads == len(got), "read more of the order than this rank needs"
    assert lazy.reads <= 64 // 8


# --------------------------------------------------------------------------------------
# 6 + 7 + 9 + 11. iter_torch_batches: per-epoch reseeding, drop_last, prefetch, docs.
# --------------------------------------------------------------------------------------


def test_successive_epochs_shuffle_differently():
    ds = _SpyDataset(_table(64), batch_rows=16)
    kwargs = {"batch_size": 8, "local_shuffle_buffer_size": 32, "seed": 1, "device": None}

    first = _ids(iter_torch_batches(ds, epoch=0, **kwargs))
    second = _ids(iter_torch_batches(_SpyDataset(_table(64), batch_rows=16), epoch=1, **kwargs))

    assert sorted(first) == sorted(second) == list(range(64))  # same samples
    assert first != second, "every epoch saw an identical order"
    # ... and the same epoch still reproduces exactly.
    again = _ids(iter_torch_batches(_SpyDataset(_table(64), batch_rows=16), epoch=0, **kwargs))
    assert again == first


def test_drop_last_removes_the_ragged_final_batch():
    ds = _SpyDataset(_table(30), batch_rows=30)
    kept = list(iter_torch_batches(ds, batch_size=8, drop_last=True, device=None))
    assert [len(b["id"]) for b in kept] == [8, 8, 8]  # the 6-row tail is gone

    ragged = list(
        iter_torch_batches(_SpyDataset(_table(30), batch_rows=30), batch_size=8, device=None)
    )
    assert [len(b["id"]) for b in ragged] == [8, 8, 8, 6]  # default is unchanged


def test_drop_last_without_a_batch_size_is_an_error():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="drop_last requires an explicit batch_size"):
        list(iter_torch_batches(_SpyDataset(_table(8)), drop_last=True, device=None))


def test_prefetch_default_overlaps_the_device_copy():
    """One batch of look-ahead cannot cover a copy plus the next read; 2-4 is the band."""
    default = inspect.signature(iter_torch_batches).parameters["prefetch_batches"].default
    assert 2 <= default <= 4


def test_local_shuffle_buffer_documents_its_cost():
    doc = iter_torch_batches.__doc__ or ""
    assert "5-10x" in doc, "the local-shuffle buffer/correlation trade is undocumented"
    assert "100-500" in doc


# --------------------------------------------------------------------------------------
# 10. ResumableSampler replayed its whole stream in every DataLoader worker.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_resumable_sampler_partitions_across_dataloader_workers(monkeypatch, num_workers):
    reference = list(ResumableSampler(64, world_size=2, rank=0, seed=7))

    per_worker = []
    for worker in range(num_workers):
        monkeypatch.setattr(
            "batcher.ml.streaming_sampler.resumable._worker_stride", lambda w=worker: (w, num_workers)
        )
        per_worker.append(list(ResumableSampler(64, world_size=2, rank=0, seed=7)))

    flat = [x for worker in per_worker for x in worker]
    # A partition, not a broadcast: before the fix this was `reference * num_workers`.
    assert len(flat) == len(reference)
    assert sorted(flat) == sorted(reference)
    assert per_worker[0] == reference[::num_workers]  # order preserved within a worker


def test_worker_striding_does_not_disturb_the_resume_position(monkeypatch):
    monkeypatch.setattr("batcher.ml.streaming_sampler.resumable._worker_stride", lambda: (1, 2))
    sampler = ResumableSampler(16, world_size=2, rank=0, seed=7)
    list(sampler)
    # The global position counts skipped samples too, so a checkpoint means the same thing
    # in every worker: 16 usable positions, 8 of them this rank's, at world_size each.
    assert sampler.global_consumed == 16
    assert len(sampler) == 0
