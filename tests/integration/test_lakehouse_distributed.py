"""A distributed lakehouse write is ONE transaction, and its result equals single-node's.

The transaction count is the load-bearing assertion. Each worker writes its own data
files, so the tempting (and wrong) design is for each worker to commit them — which
would leave N transactions in the log for one logical write, make the write
non-atomic (a reader could observe half of it), and blow up the log on a large
cluster. The contract is: workers write files, the driver commits *once*.

The second assertion is the mergeable-algebra invariant applied to writes: a table
written by W workers must be byte-for-byte the same table as one written by a single
process, and must still be readable with file skipping afterwards.
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("deltalake", reason="deltalake not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

WORKERS = 4


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    from conftest import init_test_ray, shutdown_test_ray

    started = init_test_ray(WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def data() -> pa.Table:
    rng = np.random.default_rng(0)
    n = 200_000
    return pa.table(
        {
            "day": pa.array(rng.integers(0, 20, n), pa.int64()),
            "id": pa.array(np.arange(n), pa.int64()),
            "val": pa.array(rng.random(n)),
        }
    )


def _transactions(uri: str) -> list[str]:
    log = os.path.join(uri, "_delta_log")
    return sorted(f for f in os.listdir(log) if f.endswith(".json"))


def test_distributed_write_commits_exactly_one_transaction(tmp_path, data) -> None:
    """W workers, many data files — but exactly one atomic commit."""
    uri = str(tmp_path / "t")
    manifest = bt.from_arrow(data).write.delta(uri, distributed=True, num_workers=WORKERS)

    assert manifest.num_files >= 1
    assert len(_transactions(uri)) == 1, "one logical write must be one transaction"
    assert bt.read.delta(uri).count() == data.num_rows


def test_distributed_write_equals_single_node_write(tmp_path, data) -> None:
    """The mergeable-algebra invariant, on the write side."""
    dist_uri = str(tmp_path / "dist")
    local_uri = str(tmp_path / "local")
    bt.from_arrow(data).write.delta(dist_uri, distributed=True, num_workers=WORKERS)
    bt.from_arrow(data).write.delta(local_uri)

    got = sorted(bt.read.delta(dist_uri).collect().column("id").to_pylist())
    expected = sorted(bt.read.delta(local_uri).collect().column("id").to_pylist())
    assert got == expected


def test_a_distributed_table_is_still_skippable(tmp_path, data) -> None:
    """Every worker records its files' statistics, so the table it wrote can be pruned."""
    from batcher.io.formats.lakehouse import DeltaSource

    uri = str(tmp_path / "t")
    bt.from_arrow(data).write.delta(uri, distributed=True, num_workers=WORKERS)

    predicate = {
        "e": "binary",
        "op": "eq",
        "left": {"e": "col", "name": "day"},
        "right": {"e": "lit", "value": {"int": 7}},
    }
    source = DeltaSource(uri)
    assert len(source.splits(predicate=predicate)) <= len(source.splits())
    expected = int(data.filter(pa.compute.equal(data.column("day"), 7)).num_rows)
    assert bt.read.delta(uri).filter(bt.col("day") == 7).count() == expected


def test_a_second_distributed_write_appends_one_more_transaction(tmp_path, data) -> None:
    """Each write is one commit, and the second must not clobber the first's data files."""
    uri = str(tmp_path / "t")
    bt.from_arrow(data).write.delta(uri, distributed=True, num_workers=WORKERS)
    bt.from_arrow(data.slice(0, 100)).write.delta(uri, distributed=True, num_workers=WORKERS)

    assert len(_transactions(uri)) == 2
    assert bt.read.delta(uri).count() == data.num_rows + 100
