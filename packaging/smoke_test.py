"""Prove an installed Batcher works, the way a user's first session would exercise it.

Run against an *installed* package from outside the source tree, so the checkout cannot shadow
it: the release workflow runs it on every wheel, in Alpine for the musllinux wheels, and inside
the Docker image. It covers the four things a broken install has actually got wrong here — an
undeclared import (NumPy), the native extension failing to load, a host probe crashing where no
cgroup mount exists, and the parallel path, which only engages past `MIN_ROWS_TO_SHARD`.

Past those it exercises the native paths a platform can break without breaking a group-by: a
hash join, a descending sort checked in order, an aggregate that actually spills to disk, the
Cranelift JIT (whose code generation is per-architecture and per-OS), and a streaming query.
Each is small; the point is that it ran on this wheel, on this platform.

Exits non-zero on the first failure.
"""

from __future__ import annotations

import os
import platform
import sys
import sysconfig
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import batcher as bt


def main() -> None:
    """Run the checks and print one line describing the install that passed them."""
    here = os.path.dirname(os.path.abspath(bt.__file__))
    # An editable install or a PYTHONPATH pointing at the checkout would pass every check below
    # while proving nothing about the wheel, so refuse a package that sits in a source tree.
    if os.path.exists(os.path.join(here, os.pardir, os.pardir, "Cargo.toml")):
        sys.exit(f"batcher was imported from a source checkout ({here}), not an installed wheel")

    ds = bt.from_pydict({"a": [1, 2, 3]})
    assert ds.agg(s=bt.col("a").sum()).to_pydict() == {"s": [6]}
    assert bt.sql("select 1 + 1 as x").to_pydict() == {"x": [2]}

    with tempfile.TemporaryDirectory() as tmp:
        table = pa.table({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
        pq.write_table(table, os.path.join(tmp, "p.parquet"))
        out = bt.read(tmp).group_by("k").agg(t=bt.col("v").sum()).sort("k").to_pydict()
        assert out == {"k": [1, 2], "t": [3.0, 3.0]}, out

    # 300,000 rows clears the 65,536-row sharding threshold, so this runs the parallel path.
    n = 300_000
    big = bt.from_pydict({"g": [i % 7 for i in range(n)], "x": list(range(n))})
    counts = big.group_by("g").agg(c=bt.col("x").count()).sort("g").to_pydict()
    assert counts["g"] == list(range(7)) and sum(counts["c"]) == n, counts

    _check_join_and_sort()
    _check_spill()
    _check_jit()
    _check_streaming()

    versions = bt.versions()
    print(
        f"ok: batcher {versions['batcher']} ({versions['engine_profile']}) on "
        f"{sysconfig.get_platform()} {platform.libc_ver()[0] or 'musl'} "
        f"python {platform.python_version()} from {here}"
    )


def _check_join_and_sort() -> None:
    """A hash join, then a descending sort compared in order, not as a multiset."""
    left = bt.from_pydict({"id": [3, 1, 2, 4], "l": ["c", "a", "b", "d"]})
    right = bt.from_pydict({"id": [2, 3, 5], "r": [20, 30, 50]})
    joined = left.join(right, on="id").sort("id", descending=True).to_pydict()
    assert joined == {"id": [3, 2], "l": ["c", "b"], "r": [30, 20]}, joined


def _check_spill() -> None:
    """An aggregate forced out of core must write spill buckets and match the in-memory run."""
    from batcher import Config, MemoryConfig
    from batcher.carbonite.spill.store import TieredSpillStore

    n = 50_000
    ds = bt.from_pydict({"k": [i % 2000 for i in range(n)], "v": list(range(n))})
    query = ds.group_by("k").agg(s=bt.col("v").sum()).sort("k")
    opened = [0]
    original = TieredSpillStore.writer

    def counting(self, name):
        opened[0] += 1
        return original(self, name)

    TieredSpillStore.writer = counting
    try:
        cfg = Config().replace(memory=MemoryConfig(spill_bucket_max_bytes=1))
        with bt.config_context(cfg):
            spilled = query.collect(spill=True, num_partitions=4).to_pydict()
    finally:
        TieredSpillStore.writer = original
    assert opened[0] > 0, "nothing spilled, so the out-of-core path never ran"
    assert spilled == query.to_pydict(), "the spilled aggregate disagrees with the in-memory one"


def _check_jit() -> None:
    """An aggregate over arithmetic must run on the Cranelift JIT and agree with Python."""
    n = 200_000
    ds = bt.from_pydict({"k": [i % 4 for i in range(n)], "a": list(range(n))})
    query = ds.group_by("k").agg(s=(bt.col("a") * 2 + 1).sum()).sort("k")
    run = query.stats()
    backends = {op.kind: op.backend for op in run.ops}
    assert backends.get("aggregate") in {"jit", "interp+jit"}, backends
    expected = [sum(2 * i + 1 for i in range(k, n, 4)) for k in range(4)]
    assert query.to_pydict()["s"] == expected


def _check_streaming() -> None:
    """A streaming query runs to completion on its loop thread and counts every row."""
    stream = bt.read.rate(rows_per_second=5, num_rows=12, pace=False)
    query = stream.write.memory("smoke_stream", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert query.exception() is None, query.exception()
    assert query.status.total_input_rows == 12, query.status


if __name__ == "__main__":
    main()
