"""The disk shuffle compresses what it writes to a network mount, and only there.

The Flight transport has compressed its wire since it existed
(`DistributedConfig.flight_compression`), so the one shuffle that *did* send raw bytes was
the disk one — which is also the only one whose scratch can be a cluster-shared network
filesystem, where every byte crosses the wire twice. These tests pin the two halves of
`shuffle_ipc_options`: a shared mount always compresses, node-local scratch honors the
configured codec (and so stays uncompressed under the `"auto"` default), and a file written
either way reads back identically because Arrow IPC records its own codec.
"""

from __future__ import annotations

import dataclasses
import os

import pyarrow as pa
import pytest

from batcher.config import active_config, config_context
from batcher.dist.shuffle_io import (
    read_ipc,
    shuffle_ipc_options,
    write_ipc,
    write_ipc_round_robin,
)

pytestmark = pytest.mark.unit


def _batch(n: int = 2000) -> pa.RecordBatch:
    """Rows a real shuffle carries: a low-cardinality key and a repetitive string."""
    return pa.RecordBatch.from_pydict(
        {"k": [i % 32 for i in range(n)], "s": ["a moderately repetitive value"] * n}
    )


def _shared_scratch(path: str):
    """Make `path` read as cluster-shared scratch, the way an operator-set spill dir does."""
    cfg = active_config()
    return config_context(
        dataclasses.replace(cfg, memory=dataclasses.replace(cfg.memory, spill_dir=path))
    )


def test_node_local_scratch_is_uncompressed_by_default(tmp_path):
    assert shuffle_ipc_options(str(tmp_path / "m0_r0.arrow")) is None


def test_shared_mount_scratch_is_always_compressed(tmp_path):
    with _shared_scratch(str(tmp_path)):
        opts = shuffle_ipc_options(str(tmp_path / "m0_r0.arrow"))
    assert opts is not None
    assert opts.compression == "lz4"


def test_a_sibling_directory_is_not_mistaken_for_the_shared_root(tmp_path):
    """The prefix test is on path *components*, so `/mnt/x` must not claim `/mnt/xy`."""
    shared = tmp_path / "shuffle"
    shared.mkdir()
    sibling = tmp_path / "shuffle_other"
    sibling.mkdir()
    with _shared_scratch(str(shared)):
        assert shuffle_ipc_options(str(sibling / "m0_r0.arrow")) is None


def test_an_explicit_codec_is_honored_on_local_scratch(tmp_path):
    cfg = active_config()
    with config_context(
        dataclasses.replace(cfg, memory=dataclasses.replace(cfg.memory, spill_compression="zstd"))
    ):
        opts = shuffle_ipc_options(str(tmp_path / "m0_r0.arrow"))
    assert opts is not None
    assert opts.compression == "zstd"


def test_compression_shrinks_the_bucket_and_the_rows_survive(tmp_path):
    batches = [_batch()] * 8
    plain = str(tmp_path / "plain.arrow")
    write_ipc(batches, plain)

    packed_dir = tmp_path / "shared"
    packed_dir.mkdir()
    packed = str(packed_dir / "packed.arrow")
    with _shared_scratch(str(packed_dir)):
        write_ipc(batches, packed)

    assert os.path.getsize(packed) < os.path.getsize(plain)
    # Read back with no codec argument anywhere: the IPC message carries its own.
    expected = pa.Table.from_batches(batches)
    assert pa.Table.from_batches(read_ipc(packed)).equals(expected)
    assert pa.Table.from_batches(read_ipc(plain)).equals(expected)


def test_round_robin_partitioning_compresses_on_a_shared_mount(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    paths = [str(shared / f"p{j}.arrow") for j in range(3)]
    batches = [_batch(600) for _ in range(6)]
    with _shared_scratch(str(shared)):
        write_ipc_round_robin(iter(batches), batches[0].schema, paths)

    plain_paths = [str(tmp_path / f"p{j}.arrow") for j in range(3)]
    write_ipc_round_robin(iter(batches), batches[0].schema, plain_paths)

    assert sum(os.path.getsize(p) for p in paths) < sum(os.path.getsize(p) for p in plain_paths)
    # Round-robin preserves the row multiset regardless of the codec.
    got = sum(b.num_rows for p in paths for b in read_ipc(p))
    assert got == sum(b.num_rows for b in batches)
