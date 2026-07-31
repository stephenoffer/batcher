"""The GPU worker task bodies, exercised on the host against the CPU engine.

A GPU task reads its own shard from storage and replays a translated chain on it. Both halves
matter and neither used to be checkable without a device: the *read* is what keeps a large
relation off the driver, and the *replay* is what has to agree with the engine. Parameterizing
the task body by dataframe backend makes both testable here, on pandas, with the CPU engine as
the oracle — a task that can only be exercised on a GPU is a task nothing checks.

`nest_ops` gets its own cases because the CPU fallback depends on it: a shard whose device is
lost is recomputed by the native engine from the *same* chain, rebuilt into the nested IR the
engine reads. If that rebuild were wrong, the fallback would answer a different question than
the shard it replaced, and the combined result would be quietly incoherent.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_join_spec, gpu_plan_ops
from batcher.dist.gpu.tasks import run_shard_chain, run_shard_join
from batcher.plan.distribution import nest_ops

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


def _table():
    rng = np.random.default_rng(3)
    return pa.table(
        {
            "k": rng.integers(0, 6, 120).astype("int64"),
            "v": rng.random(120) * 10.0,
        }
    )


def _descriptor(table: pa.Table) -> dict:
    """A batch-list descriptor, the in-memory form `partition_descriptors` produces."""
    return {"batches": table.to_batches()}


def _rows(table: pa.Table) -> list[tuple]:
    def canon(v):
        return float(f"{v:.12e}") if isinstance(v, float) and v == v else v

    return sorted(
        (tuple(canon(v) for v in r) for r in zip(*table.to_pydict().values(), strict=True)),
        key=repr,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.filter(col("v") > 3.0),
        lambda ds: ds.group_by("k").agg(s=col("v").sum(), n=bt.count()),
        lambda ds: ds.with_columns(w=col("v") * 2.0).filter(col("w") < 12.0),
        lambda ds: ds.sort("v", descending=True).limit(5),
    ],
)
def test_shard_task_body_matches_the_cpu_engine(build, be):
    table = _table()
    ds = build(bt.from_arrow(table))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    got = run_shard_chain(_descriptor(table), spec[1], be)
    assert _rows(got.select(ds.collect().column_names)) == _rows(ds.collect())


def test_shard_task_reports_an_empty_shard_as_none(be):
    """An empty shard returns `None`, not an empty table.

    A fan-out concatenates its shards' results, and an empty table carries a schema that need
    not match the others'. Dropping it at the source is what keeps the concatenation total.
    """
    ds = bt.from_arrow(_table()).filter(col("v") > 3.0)
    ops = gpu_plan_ops(ds._plan)[1]
    assert run_shard_chain({"batches": []}, ops, be) is None


def test_join_task_body_matches_the_cpu_engine(be):
    fact = pa.table({"id": np.array([1, 2, 3, 1, 2], "int64"), "v": np.array([1.0, 2, 3, 4, 5])})
    dim = pa.table({"id": np.array([1, 2, 3], "int64"), "w": np.array([10, 20, 30], "int64")})
    ds = bt.from_arrow(fact).join(bt.from_arrow(dim), on="id").filter(col("w") > 10)
    (ls, lops), (rs, rops), jir, ops = gpu_join_spec(ds._plan)
    lt = fact if ls.source_id == 0 else dim
    rt = dim if rs.source_id == 1 else fact
    got = run_shard_join(_descriptor(lt), _descriptor(rt), lops, rops, jir, ops, be)
    expected = ds.collect()
    assert _rows(got.select(expected.column_names)) == _rows(expected)


def test_join_task_reports_an_empty_side_as_none(be):
    fact = pa.table({"id": np.array([1], "int64"), "v": np.array([1.0])})
    dim = pa.table({"id": np.array([1], "int64"), "w": np.array([10], "int64")})
    ds = bt.from_arrow(fact).join(bt.from_arrow(dim), on="id")
    (_ls, lops), (_rs, rops), jir, ops = gpu_join_spec(ds._plan)
    assert run_shard_join({"batches": []}, _descriptor(dim), lops, rops, jir, ops, be) is None


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.filter(col("v") > 3.0),
        lambda ds: ds.filter(col("v") > 3.0).group_by("k").agg(s=col("v").sum()),
        lambda ds: ds.with_columns(w=col("v") * 2.0).sort("w").limit(3),
    ],
)
def test_nested_ops_round_trip_to_the_engine(build):
    """The flat chain rebuilt as nested IR is the plan the engine would have run itself.

    This is the CPU fallback's contract: a shard whose device is lost is recomputed from this
    document, so it must denote the same query the device was given.
    """
    table = _table()
    ds = build(bt.from_arrow(table))
    ops = gpu_plan_ops(ds._plan)[1]
    nested = nest_ops(ops)
    assert nested == ds._plan.to_ir()


def test_nested_ops_bottoms_out_at_the_scan():
    ds = bt.from_arrow(_table()).filter(col("v") > 1.0).group_by("k").agg(s=col("v").sum())
    nested = nest_ops(gpu_plan_ops(ds._plan)[1], source_id=0)
    node = nested
    depth = 0
    while node.get("op") != "scan":
        node = node["input"]
        depth += 1
    assert depth == 2  # filter, aggregate
    assert node == {"op": "scan", "source_id": 0}


def test_union_task_body_matches_the_cpu_engine(be):
    from batcher.core.gpu_plan import gpu_union_spec
    from batcher.dist.gpu.tasks import run_shard_union

    a = pa.table({"x": np.array([1, 2, 3], "int64"), "y": np.array([1.0, 2, 3])})
    b = pa.table({"x": np.array([3, 4], "int64"), "y": np.array([3.0, 4])})
    ds = bt.from_arrow(a).union(bt.from_arrow(b)).group_by("x").agg(s=col("y").sum())
    inputs, distinct, ops = gpu_union_spec(ds._plan)
    tables = [a if s.source_id == 0 else b for s, _ in inputs]
    got = run_shard_union(
        [_descriptor(t) for t in tables], [o for _, o in inputs], distinct, ops, be
    )
    expected = ds.collect()
    assert _rows(got.select(expected.column_names)) == _rows(expected)


def test_union_task_reports_all_empty_inputs_as_none(be):
    from batcher.core.gpu_plan import gpu_union_spec
    from batcher.dist.gpu.tasks import run_shard_union

    a = pa.table({"x": np.array([1], "int64")})
    ds = bt.from_arrow(a).union(bt.from_arrow(a))
    inputs, distinct, ops = gpu_union_spec(ds._plan)
    empty = [{"batches": []} for _ in inputs]
    assert run_shard_union(empty, [o for _, o in inputs], distinct, ops, be) is None
