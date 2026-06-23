"""`ds.ml.stream_loader` — distributed-training data ingest (F).

Verifies the loader honors the sampler contract end to end and emits torch tensors:
deterministic, balanced across ranks (no DDP straggler hang), full coverage with no
dup/skip, mid-epoch resume, FixedShapeTensor → shaped tensor, strings dropped.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

import torch

import batcher as bt


def _ds(n: int = 100) -> bt.Dataset:
    return bt.from_arrow(pa.table({"id": list(range(n)), "v": [float(i) for i in range(n)]}))


def _ids(loader) -> list[int]:
    out: list[int] = []
    for batch in loader:
        out.extend(int(x) for x in batch["id"].tolist())
    return out


def test_deterministic_balanced_and_full_coverage():
    W = 4
    per_rank = [
        _ids(_ds(100).ml.stream_loader(batch_size=5, world_size=W, rank=r, seed=1))
        for r in range(W)
    ]
    assert len({len(x) for x in per_rank}) == 1  # balanced → no DDP hang
    assert sorted(x for r in per_rank for x in r) == list(range(100))  # every id once
    # Deterministic: same (seed, epoch, rank) reproduces exactly.
    assert (
        _ids(_ds(100).ml.stream_loader(batch_size=5, world_size=W, rank=0, seed=1)) == per_rank[0]
    )


def test_epoch_changes_order():
    a = _ids(_ds(64).ml.stream_loader(batch_size=8, epoch=0, seed=3))
    b = _ids(_ds(64).ml.stream_loader(batch_size=8, epoch=1, seed=3))
    assert sorted(a) == sorted(b)  # same samples
    assert a != b  # different order


def test_tensor_types_and_strings_dropped():
    ds = bt.from_arrow(
        pa.table({"id": [1, 2, 3, 4], "v": [1.0, 2.0, 3.0, 4.0], "name": ["a", "b", "c", "d"]})
    )
    batch = next(iter(ds.ml.stream_loader(batch_size=2, seed=0)))
    assert isinstance(batch["id"], torch.Tensor)
    assert isinstance(batch["v"], torch.Tensor)
    assert "name" not in batch  # non-numeric columns are not tensorized


def test_fixed_shape_tensor_yields_shaped_tensor():
    arr = np.arange(6 * 4, dtype=np.float32).reshape(6, 2, 2)
    ds = bt.from_arrow(
        pa.table({"id": list(range(6)), "t": pa.FixedShapeTensorArray.from_numpy_ndarray(arr)})
    )
    batch = next(iter(ds.ml.stream_loader(batch_size=3, shuffle=False, seed=0)))
    assert batch["t"].shape == (3, 2, 2)
    assert batch["t"].dtype == torch.float32


def test_resume_skips_consumed_samples():
    W = 2
    full = _ids(_ds(40).ml.stream_loader(batch_size=4, world_size=W, rank=0, seed=2))
    # global_consumed=8 → positions 0..7 done; rank 0's strided positions there are
    # {0,2,4,6} → 4 of its samples already processed.
    resumed = _ids(
        _ds(40).ml.stream_loader(batch_size=4, world_size=W, rank=0, seed=2, global_consumed=8)
    )
    assert resumed == full[4:]


def test_loader_tensors_are_writable():
    # A training loop mutates batches in place (augmentation/normalization); the
    # yielded tensors must own writable memory, not a read-only Arrow-buffer view.
    batch = next(iter(_ds(8).ml.stream_loader(batch_size=4, seed=0)))
    for t in batch.values():
        assert t.flags["WRITEABLE"] if hasattr(t, "flags") else t.is_contiguous()
        t += 1  # must not raise / corrupt source


def test_shard_stream_loader_streams_from_disk(tmp_path):
    # Write a corpus to shards, then stream it (bounded memory) with the SAME
    # deterministic/balanced/elastic contract as the in-memory loader.
    import pyarrow as pa

    from batcher.io.formats.ml import write_shards
    from batcher.ml import shard_stream_loader

    tbl = pa.table({"id": list(range(100)), "v": [float(i) for i in range(100)]})
    write_shards(tbl, str(tmp_path), rows_per_shard=16)

    W = 4
    per_rank = []
    for r in range(W):
        ld = shard_stream_loader(
            str(tmp_path), batch_size=5, world_size=W, rank=r, seed=1, cache_size=2
        )
        ids = []
        for batch in ld:
            ids.extend(int(x) for x in batch["id"].tolist())
        per_rank.append(ids)
    # Balanced (no DDP straggler) and full coverage, each id once.
    assert len({len(x) for x in per_rank}) == 1
    assert sorted(x for r in per_rank for x in r) == list(range(100))
    # Deterministic: same args reproduce rank 0 exactly.
    again = shard_stream_loader(str(tmp_path), batch_size=5, world_size=W, rank=0, seed=1)
    assert [int(x) for b in again for x in b["id"].tolist()] == per_rank[0]


def test_shard_stream_loader_tensors_writable_and_typed(tmp_path):
    import pyarrow as pa
    import torch

    from batcher.io.formats.ml import write_shards
    from batcher.ml import shard_stream_loader

    write_shards(
        pa.table({"id": [1, 2, 3, 4], "v": [1.0, 2.0, 3.0, 4.0]}), str(tmp_path), rows_per_shard=2
    )
    batch = next(iter(shard_stream_loader(str(tmp_path), batch_size=2, shuffle=False)))
    assert isinstance(batch["id"], torch.Tensor)
    batch["id"] += 1  # writable (no error / corruption)
