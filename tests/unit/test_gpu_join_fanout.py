"""Splitting a join's probe side across devices equals the single-node join.

A join was the last relational shape pinned to one device, which is backwards: it is the shape
whose whole premise is that one input is large. Splitting the **probe** side and giving every
worker the whole **build** side is correct for the join types whose output is driven by left
rows, and these cases check that against the CPU engine at several shard counts, with chains
above the join folding through the same split every other chain uses.

The exclusion has its own case, and it is the important one: a `right` or `full` join must emit
an unmatched build row exactly once, and every shard sees the whole build side, so each would
emit it. The case demonstrates the duplication concretely rather than asserting a rule, because
a rule with no counter-example tends to get "simplified" back out.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_join_spec
from batcher.dist.gpu.tasks import run_shard_join
from batcher.plan.distribution import BROADCAST_SAFE_JOINS, ShardSplit, shard_plan

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


def _tables():
    """Unmatched keys on BOTH sides, so every join type has work to do.

    Probe keys 0..2 have no dimension row, which is what `left` and `anti` turn on; dimension
    keys 6..9 have no fact row, which is what `right` and `outer` turn on — and which is the
    thing a per-shard broadcast would emit once per shard.
    """
    rng = np.random.default_rng(11)
    fact = pa.table({"id": rng.integers(0, 6, 500).astype("int64"), "v": rng.random(500) * 50.0})
    dim = pa.table(
        {"id": np.arange(3, 10, dtype="int64"), "w": (np.arange(3, 10) * 10).astype("int64")}
    )
    return fact, dim


def _shards(table: pa.Table, k: int) -> list[pa.Table]:
    step = max(1, -(-table.num_rows // k))
    return [table.slice(i, min(step, table.num_rows - i)) for i in range(0, table.num_rows, step)]


def _rows(table: pa.Table) -> list[tuple]:
    def canon(v):
        if isinstance(v, float):
            return "__nan__" if v != v else float(f"{v:.12e}")
        return v

    return sorted(
        (tuple(canon(v) for v in r) for r in zip(*table.to_pydict().values(), strict=True)),
        key=repr,
    )


def _descriptor(table: pa.Table) -> dict:
    return {"batches": table.to_batches()}


def _fan_out(ds, fact, dim, shard_count, be):
    """Run the join with the probe side split and the build side whole on every shard."""
    from batcher.core.gpu_plan.execute import run_chain

    spec = gpu_join_spec(ds._plan)
    assert spec is not None
    (ls, lops), (rs, rops), jir, ops = spec
    probe = fact if ls.source_id == 0 else dim
    build = dim if rs.source_id == 1 else fact
    assert jir["join_type"] in BROADCAST_SAFE_JOINS
    above = shard_plan(ops) if ops else ShardSplit([], [], [])
    assert above is not None

    pieces = []
    for shard in _shards(probe, shard_count):
        out = run_shard_join(
            _descriptor(shard), _descriptor(build), lops, rops, jir, above.shard_ops, be
        )
        if out is not None and out.num_rows:
            pieces.append(out)
    if not pieces:
        pieces = [
            run_shard_join(
                _descriptor(probe), _descriptor(build), lops, rops, jir, above.shard_ops, be
            )
        ]
    merged = pa.concat_tables([p for p in pieces if p is not None])
    tail = [*above.merge_ops, *above.tail_ops]
    return be.to_arrow(run_chain(merged, tail, be)) if tail else merged


@pytest.mark.parametrize("shard_count", [1, 2, 3, 7, 500])
@pytest.mark.parametrize(
    "build",
    [
        lambda a, b: a.join(b, on="id", how="inner"),
        lambda a, b: a.join(b, on="id", how="left"),
        lambda a, b: a.join(b, on="id", how="semi"),
        lambda a, b: a.join(b, on="id", how="anti"),
        lambda a, b: a.join(b, on="id").filter(col("w") > 10),
        lambda a, b: a.join(b, on="id").select("w", "v"),
        # a fold above the join — the star-schema shape the fan-out exists for
        lambda a, b: a.join(b, on="id").group_by("w").agg(s=col("v").sum()),
        lambda a, b: (
            a.join(b, on="id")
            .group_by("w")
            .agg(s=col("v").sum())
            .sort("s", descending=True)
            .limit(3)
        ),
        lambda a, b: a.join(b, on="id").select("w").distinct(),
        lambda a, b: a.filter(col("v") > 10.0).join(b.filter(col("w") < 50), on="id"),
    ],
)
def test_probe_side_shards_match_the_single_node_join(build, shard_count, be):
    fact, dim = _tables()
    ds = build(bt.from_arrow(fact), bt.from_arrow(dim))
    got = _fan_out(ds, fact, dim, shard_count, be)
    expected = ds.collect()
    assert _rows(got.select(expected.column_names)) == _rows(expected)


@pytest.mark.parametrize("how", ["right", "outer"])
def test_right_and_outer_joins_are_excluded(how):
    """These are the join types the fan-out must refuse, and the reason is duplication."""
    assert how not in BROADCAST_SAFE_JOINS


def test_broadcasting_an_outer_join_would_duplicate_unmatched_build_rows(be):
    """The concrete failure the exclusion prevents.

    An unmatched build row must appear exactly once. Every shard sees the whole build side, so
    every shard emits it — two shards, two copies. Shown rather than asserted as a rule,
    because a rule with no counter-example is the kind that gets simplified back out.
    """
    fact, dim = _tables()
    ds = bt.from_arrow(fact).join(bt.from_arrow(dim), on="id", how="outer")
    spec = gpu_join_spec(ds._plan)
    assert spec is not None
    (ls, lops), (rs, rops), jir, ops = spec
    probe = fact if ls.source_id == 0 else dim
    build = dim if rs.source_id == 1 else fact
    assert jir["join_type"] not in BROADCAST_SAFE_JOINS  # ...which is why this is never run

    single = run_shard_join(
        _descriptor(probe), _descriptor(build), lops, rops, jir, ops, be
    ).num_rows
    split = sum(
        run_shard_join(_descriptor(shard), _descriptor(build), lops, rops, jir, ops, be).num_rows
        for shard in _shards(probe, 2)
    )
    assert split > single, "the whole point of the exclusion"
