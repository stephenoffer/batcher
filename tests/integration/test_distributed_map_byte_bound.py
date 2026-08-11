"""A map fanned out by its *byte* budget still equals single-node.

The distributed map's partition count is the maximum of two terms: a parallelism term clamped
to the cluster's cores, and a memory term (`target_bytes_per_task`) that is a bound and so is
deliberately *not* clamped. Above `cores x target_bytes_per_task` the memory term wins and the
stage runs more tasks than there are cores — the configuration that keeps a large scan's
per-task input flat instead of growing with the dataset.

Nothing else exercises that shape: every other distributed test runs at or below one task per
core, so the branch where the byte term is the binding one had no end-to-end coverage at all.
These tests force it with a small `target_bytes_per_task` rather than a large corpus, so they
cost a few seconds instead of a terabyte.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import tempfile

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher.config import active_config, config_context
from batcher.dist.executors import map as mapmod

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

_ROWS = 120_000
# Enough files that the source has splits to spread over: the partition count is capped at
# `len(source.splits())`, so a byte budget can only bind if there are splits to give it.
_FILES = 160
# Small enough that the byte term lands well ABOVE the cluster's core count, which is the
# only regime where the two formulas differ: below it, clamping the byte term to the cores is
# a no-op and the test would pass against the very bug it exists for.
_TIGHT_BUDGET = 4 << 10


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _corpus_root(tmp_path_factory):
    """A directory every worker of the attached cluster can read.

    The byte term only binds for a *splittable* source, so this needs real files on disk — and
    `init_test_ray` attaches to whatever cluster is running, which on a real deployment is
    multi-node. A driver-local `tmp_path` is then invisible to the workers and every task dies
    with `FileNotFoundError`, which reads as a batcher fault and is not one. Prefer a shared
    scratch directory when the cluster has more than one node, and say so plainly when there
    is none rather than failing obscurely.
    """
    import ray

    alive = [n for n in ray.nodes() if n.get("Alive")]
    if len(alive) <= 1:
        return tmp_path_factory.mktemp("bytebound")
    shared = os.environ.get("BATCHER_TEST_SHARED_DIR") or "/mnt/cluster_storage"
    if not os.path.isdir(shared) or not os.access(shared, os.W_OK):
        pytest.skip(
            f"a {len(alive)}-node Ray cluster is attached but no shared scratch directory is "
            "available; set BATCHER_TEST_SHARED_DIR to one the workers can read"
        )
    return pathlib.Path(tempfile.mkdtemp(prefix="batcher-bytebound-", dir=shared))


@pytest.fixture
def corpus(tmp_path_factory) -> list[str]:
    """Several Parquet files, so the source has splits to spread over many tasks."""
    import pyarrow.parquet as pq

    root = _corpus_root(tmp_path_factory)
    rng = np.random.default_rng(11)
    paths = []
    per_file = _ROWS // _FILES
    for i in range(_FILES):
        table = pa.table(
            {
                "price": rng.random(per_file) * 100.0,
                "qty": rng.integers(1, 9, per_file).astype("float64"),
                "flag": pa.array(rng.choice(["A", "N", "R"], per_file)),
            }
        )
        path = str(root / f"part-{i:03d}.parquet")
        pq.write_table(table, path, row_group_size=per_file)
        paths.append(path)
    return paths


def _pipeline(paths: list[str]):
    """The pipeline under test, with its UDF defined *locally*.

    A module-level `fn` is pickled by reference, and a Ray worker cannot import a pytest test
    module — the failure is a worker-side `ModuleNotFoundError`, not a batcher fault. A local
    closure is pickled by value, which is what every other distributed `map_batches` test here
    relies on too.
    """

    def charge(batch: pa.RecordBatch) -> pa.RecordBatch:
        price = batch.column("price").to_numpy(zero_copy_only=False)
        qty = batch.column("qty").to_numpy(zero_copy_only=False)
        return pa.record_batch(
            {"flag": batch.column("flag"), "charge": pa.array(np.sqrt(price * price + qty))}
        )

    return (
        bt.read.parquet(paths)
        .map_batches(
            charge, input_columns=["price", "qty", "flag"], output_columns=["flag", "charge"]
        )
        .group_by("flag")
        .agg(total=bt.col("charge").sum(), n=bt.col("charge").count())
    )


def _rows(table: pa.Table) -> list[tuple]:
    return sorted((r["flag"], r["n"], round(r["total"], 6)) for r in table.to_pylist())


@dataclasses.dataclass
class _Observed:
    partitions: int = 0


@pytest.fixture
def observed(monkeypatch) -> _Observed:
    """Record the partition count the sizing actually chose."""
    seen = _Observed()
    original = mapmod._adaptive_partition_count

    def spy(*args, **kwargs):
        seen.partitions = original(*args, **kwargs)
        return seen.partitions

    monkeypatch.setattr(mapmod, "_adaptive_partition_count", spy)
    return seen


def _tiny_byte_budget(byte_budget: int):
    cfg = active_config()
    return config_context(
        cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, target_bytes_per_task=byte_budget))
    )


def _row_term() -> int:
    """The count the *parallelism* term alone would choose, clamped as the code clamps it."""
    import math

    from batcher.config import active_config

    rows_per_cpu = max(1, active_config().optimizer.target_rows_per_task // 2)
    return max(1, min(math.ceil(_ROWS / rows_per_cpu), int(mapmod._cluster_cores())))


def _byte_term(paths: list[str], byte_budget: int) -> int:
    """The count the *memory* term needs, capped by the splits that actually exist.

    This is the number the assertions below are written against, because it is the one the
    old formula destroyed: it took `min(max(row_term, byte_term), cluster_cores)`, so any byte
    term above the core count came back as the core count. Asserting merely "more tasks than
    the row term" does not catch that — both formulas clear it — which is why the first
    version of this test passed against the bug it was written for.
    """
    ds = bt.read.parquet(paths)
    source = ds._sources[0]
    with _tiny_byte_budget(byte_budget):
        needed = mapmod._byte_partition_count(source, ds._plan, _ROWS)
    return min(needed, len(source.splits()))


def test_the_byte_budget_and_not_the_row_count_decides_the_fan_out(corpus, observed) -> None:
    """The memory term is a bound, so it must survive being larger than the row term.

    Stated as "the byte term is what is binding" rather than "more tasks than cores": both
    describe the same branch, but the core count belongs to whatever cluster the suite is
    attached to, and a corpus small enough to write in a test cannot exceed 128 cores. What
    the fix has to guarantee is that the larger of the two terms wins, and that is checkable
    anywhere.
    """
    needed = _byte_term(corpus, _TIGHT_BUDGET)
    with _tiny_byte_budget(_TIGHT_BUDGET):
        _pipeline(corpus).collect(distributed=True)

    cores = int(mapmod._cluster_cores())
    assert needed > cores, (
        f"the fixture must need more tasks ({needed}) than the cluster has cores ({cores}) or "
        "the core clamp is a no-op here and this test cannot see the defect"
    )
    assert observed.partitions >= needed, (
        f"the byte budget needs {needed} tasks to hold each one's input inside it, but "
        f"{observed.partitions} were used — the memory bound was clamped away"
    )


def test_a_byte_bounded_fan_out_equals_single_node(corpus, observed) -> None:
    """Sharding by bytes must not change a single row.

    Partition count only shards, so this is the invariant the whole sizing change rests on.
    """
    needed = _byte_term(corpus, _TIGHT_BUDGET)
    with _tiny_byte_budget(_TIGHT_BUDGET):
        distributed = _pipeline(corpus).collect(distributed=True)
    single = _pipeline(corpus).collect(distributed=False)

    assert observed.partitions >= needed
    assert _rows(distributed) == _rows(single)
    assert sum(r[1] for r in _rows(distributed)) == _ROWS


def test_a_generous_byte_budget_leaves_the_fan_out_to_the_parallelism_term(corpus, observed):
    """The other side of the maximum: a budget nothing exceeds must not inflate the count."""
    with _tiny_byte_budget(1 << 30):  # 1 GiB — the corpus is far smaller
        _pipeline(corpus).collect(distributed=True)

    assert observed.partitions <= _row_term()
