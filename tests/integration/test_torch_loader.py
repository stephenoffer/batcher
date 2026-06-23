"""`iter_torch_batches` / `streaming_split` — the lazy PyTorch dataloader path.

Streams the dataset in bounded memory (no `collect`), dropping non-numeric columns,
with optional column subset, prefetch, local-shuffle buffer, collate_fn, and a
batch-sharded `streaming_split`. Equivalence: prefetch must not change the data, the
shuffle must be a permutation, and the splits must partition the rows.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed")

import batcher as bt


def _ds(n: int = 100):
    return bt.from_pydict(
        {"x": list(range(n)), "y": [float(i) * 2 for i in range(n)], "name": ["a"] * n}
    )


def _all_x(batches) -> list[int]:
    return [int(v) for b in batches for v in b["x"].tolist()]


def test_streams_all_rows_and_drops_non_numeric():
    batches = list(_ds().ml.iter_torch_batches(batch_size=32))
    assert set(batches[0].keys()) == {"x", "y"}  # "name" dropped
    assert sorted(_all_x(batches)) == list(range(100))


def test_column_subset():
    batches = list(_ds().ml.iter_torch_batches(batch_size=50, columns=["x"]))
    assert set(batches[0].keys()) == {"x"}


def test_prefetch_matches_no_prefetch():
    a = _all_x(_ds().ml.iter_torch_batches(batch_size=16, prefetch_batches=0))
    b = _all_x(_ds().ml.iter_torch_batches(batch_size=16, prefetch_batches=4))
    assert a == b == list(range(100))


def test_local_shuffle_is_a_permutation():
    batches = list(_ds().ml.iter_torch_batches(batch_size=16, local_shuffle_buffer_size=40, seed=7))
    assert sorted(_all_x(batches)) == list(range(100))


def test_collate_fn():
    def collate(arrays):
        return {"s": arrays["x"] + arrays["y"]}

    batches = list(_ds().ml.iter_torch_batches(batch_size=100, collate_fn=collate))
    assert list(batches[0].keys()) == ["s"]


def test_streaming_split_partitions_rows():
    splits = bt.ml.streaming_split(_ds(), 4, batch_size=10)
    collected = [_all_x(s) for s in splits]
    flat = sorted(v for shard in collected for v in shard)
    assert flat == list(range(100))  # disjoint and complete
    assert all(len(c) > 0 for c in collected)


def test_streaming_split_single_rank():
    one = bt.ml.streaming_split(_ds(), 2, rank=0, batch_size=10)
    assert len(_all_x(one)) == 50
