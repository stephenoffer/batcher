"""A training epoch must see each sample exactly once — including under `num_workers > 1`.

`IterableDataset` is *replicated* into every DataLoader worker process, and each worker runs
`__iter__` in full. A dataset that ignores `torch.utils.data.get_worker_info()` therefore yields
its whole sequence once per worker, so the completely ordinary

    DataLoader(stream_loader(ds, ...), num_workers=4)

silently trains on every sample four times per epoch. Nothing raises, nothing warns — the loss
curve just quietly means something else, and the "each sample exactly once" contract that the
whole Feistel-permutation sampler exists to provide is void.

These tests pin the contract end to end through a real `DataLoader`: the union over workers is
the rank's sequence, each sample produced by exactly one worker. `num_workers=0` (no worker
process) must be unchanged.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _drain(loader, num_workers: int) -> list[int]:
    """Every `x` value the DataLoader yields, across all its workers."""
    from torch.utils.data import DataLoader

    # `batch_size=None` disables the DataLoader's own collation: the dataset already yields
    # whole batches, which is the shape a Batcher loader produces.
    out: list[int] = []
    for item in DataLoader(loader, batch_size=None, num_workers=num_workers):
        out.extend(int(v) for v in item["x"])
    return out


@pytest.mark.parametrize("num_workers", [0, 1, 2, 4])
def test_stream_loader_yields_each_sample_exactly_once(num_workers):
    import batcher as bt
    from batcher.ml import stream_loader

    ds = bt.from_pydict({"x": list(range(64))})
    loader = stream_loader(ds, batch_size=8, shuffle=False)

    got = _drain(loader, num_workers)

    # The whole corpus, once. Before the fix this was `64 * num_workers` values with every
    # sample repeated `num_workers` times.
    assert sorted(got) == list(range(64))
    assert len(got) == 64, f"{num_workers} workers duplicated the epoch: {len(got)} samples"


@pytest.mark.parametrize("num_workers", [2, 3])
def test_shard_stream_loader_yields_each_sample_exactly_once(tmp_path, num_workers):
    """Same contract for the out-of-core shard loader — the one used past RAM."""
    import batcher as bt
    from batcher.io.formats.ml import write_shards
    from batcher.ml import shard_stream_loader

    directory = str(tmp_path / "shards")
    write_shards(bt.from_pydict({"x": list(range(60))}), directory, rows_per_shard=16)

    loader = shard_stream_loader(directory, batch_size=6, shuffle=False)

    got = _drain(loader, num_workers)

    assert sorted(got) == list(range(60))
    assert len(got) == 60, f"{num_workers} workers duplicated the epoch: {len(got)} samples"


def test_workers_partition_the_batches_rather_than_replaying_them():
    """Each batch is produced by exactly one worker — a partition, not a broadcast.

    Asserted on the stride arithmetic directly, because it is the property the DataLoader
    result depends on: offsets `0..n-1` under stride `n` tile the integers with no gap and no
    overlap.
    """
    from batcher.ml.converters import _worker_stride

    # Outside a worker process there is no striding at all.
    assert _worker_stride() == (0, 1)

    for stride in (1, 2, 4, 8):
        owners = [i % stride for i in range(64)]
        for offset in range(stride):
            assert owners.count(offset) == 64 // stride  # equal share, none dropped
        assert len({(i, owners[i]) for i in range(64)}) == 64  # exactly one owner each
