"""The broadcast build side must be *measured* against its budget without being read whole.

`_materialize_build_side` is the guard that makes an estimated broadcast decision safe to
act on: the planner marks a join broadcast on a byte estimate, and this re-checks the real
size before replicating it to every worker. The guard is not optional — a mis-estimated
build side replicated cluster-wide is an OOM rather than a slow query.

What it must not do is cost more than the strategy it guards. Reading the whole relation
first meant a declined broadcast paid a full single-threaded driver scan of the build side:
measured on TPC-H sf100 `lineitem join orders` over 8 workers, **10.2 s spent to learn the
answer and then discard the data**, turning an 11.7 s query into 23.5 s.

So the two properties below are a pair, and neither alone is the contract: a build side
that fits must come back byte-for-byte what the whole-relation read produced, and one that
does not must be refused *early*, without draining the source.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.native import engine
from batcher.dist.executors.partition_io import source_pushdown
from batcher.dist.flight_broadcast import _bounded_build_side, _splittable_build_reads

pytestmark = pytest.mark.unit

_FILES = 8
_ROWS_PER_FILE = 2_000


@pytest.fixture
def parquet_dir(tmp_path):
    """A multi-file Parquet source, so it has more than one split to stop between."""
    d = tmp_path / "t"
    d.mkdir()
    for f in range(_FILES):
        base = f * _ROWS_PER_FILE
        pq.write_table(
            pa.table(
                {
                    "k": pa.array(range(base, base + _ROWS_PER_FILE), pa.int64()),
                    "v": pa.array([float(i) for i in range(_ROWS_PER_FILE)], pa.float64()),
                }
            ),
            d / f"part-{f}.parquet",
        )
    return str(d)


def _plan_and_source(uri: str):
    ds = bt.read.parquet(uri).select("k", "v")
    return ds._plan, ds._sources[0]


def test_a_build_side_that_fits_is_identical_to_the_whole_relation_read(parquet_dir):
    """Chunked execution of a row-wise plan is the same relation, in the same order.

    That equality is the whole licence for reading it a piece at a time — the plan is
    scan/filter/project, so its output over a concatenation of chunks *is* the
    concatenation of its output over each chunk.
    """
    from batcher.io.source import read_source

    plan, source = _plan_and_source(parquet_dir)
    proj, pred = source_pushdown(plan, 0)
    nat = engine()
    ir = json.dumps(plan.to_ir())

    whole = pa.Table.from_batches(nat.execute_plan(ir, [read_source(source, proj, pred)], ""))
    chunked = _bounded_build_side(nat, plan, source, proj, pred, "", budget=1 << 40)
    assert chunked is not None
    assert pa.Table.from_batches(chunked).equals(whole)
    assert whole.num_rows == _FILES * _ROWS_PER_FILE


def test_an_oversized_build_side_is_refused_without_draining_the_source(parquet_dir, monkeypatch):
    """The point of the change: refusing must not cost a full read of the relation.

    Counted in splits actually consumed rather than in wall time, because a timing on a
    2,000-row fixture would measure the harness. A budget of one byte can be answered by
    the first chunk, so anything past the first split is work the guard did not need.
    """
    import batcher.dist.executors.scan_read as scan_read

    plan, source = _plan_and_source(parquet_dir)
    proj, pred = source_pushdown(plan, 0)
    assert len(_splittable_build_reads(plan, source) or ()) > 1  # the fixture must split

    read = {"batches": 0}
    real = scan_read._read_split_batches

    def counting(*a, **k):
        for batch in real(*a, **k):
            read["batches"] += 1
            yield batch

    monkeypatch.setattr(scan_read, "_read_split_batches", counting)

    assert _bounded_build_side(engine(), plan, source, proj, pred, "", budget=1) is None
    assert read["batches"] < _FILES, f"drained {read['batches']} batches of {_FILES} splits"


def test_a_plan_with_a_breaker_is_read_whole(parquet_dir):
    """Chunking is licensed by row-wiseness, so an aggregate build side must not take it.

    `None` here means "no bounded path"; the caller then reads the relation whole, which is
    the behaviour before this existed.
    """
    ds = bt.read.parquet(parquet_dir).group_by("k").agg(n=bt.count())
    assert _splittable_build_reads(ds._plan, ds._sources[0]) is None


def test_an_unsplittable_source_is_read_whole():
    """One whole-source split cannot be read incrementally, so it keeps the plain path."""
    ds = bt.from_pydict({"k": [1, 2, 3]}).select("k")
    assert _splittable_build_reads(ds._plan, ds._sources[0]) is None
