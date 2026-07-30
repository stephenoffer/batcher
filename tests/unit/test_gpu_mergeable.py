"""The multi-GPU aggregate's mergeable invariant, checked on the host with no GPU or cluster.

`decompose` splits a group-by aggregate into `partial → combine → finalize`, all three
expressed as plan IR. The property that makes a multi-device run trustworthy is that folding
the shards' partials reproduces the single-node answer **for any shard count** — the mergeable
contract every stateful operator in this engine owes. That is what these cases assert, against
the CPU engine's own result for the whole table.

They also pin the other half of the contract: a reduction with no mergeable partial form must
be *declined*, not approximated. A `median` whose shard medians were averaged would look
plausible on uniform data and be wrong on real data, which is the failure mode a scale-out path
must never have.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain
from batcher.plan.distribution import decompose

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


def _table():
    rng = np.random.default_rng(7)
    return pa.table(
        {
            "k": rng.integers(0, 12, 400).astype("int64"),
            "j": rng.integers(0, 3, 400).astype("int64"),
            "v": rng.random(400) * 100.0,
        }
    )


def _nulls():
    """A null group key, an all-null group, and a group of one — the cases a naive fold breaks."""
    return pa.table(
        {
            "k": pa.array([1, 1, None, 2, 2, None, 3], type=pa.int64()),
            "v": pa.array([1.0, None, 3.0, 4.0, 5.0, None, None], type=pa.float64()),
        }
    )


def _shards(table: pa.Table, k: int) -> list[pa.Table]:
    step = max(1, -(-table.num_rows // k))
    return [table.slice(i, min(step, table.num_rows - i)) for i in range(0, table.num_rows, step)]


def _rows(table: pa.Table) -> list[tuple]:
    def canon(v):
        # Twelve significant digits, not an absolute rounding: a mergeable fold re-associates
        # float arithmetic, and an absolute tolerance fails on large magnitudes for a
        # difference of one part in 1e15.
        if isinstance(v, float):
            return "__nan__" if v != v else float(f"{v:.12e}")
        return v

    return sorted(
        (tuple(canon(v) for v in row) for row in zip(*table.to_pydict().values(), strict=True)),
        key=repr,
    )


def _fold(table: pa.Table, ops: list[dict], shard_count: int, be) -> pa.Table:
    """Reduce each shard independently, then fold the partials — the distributed shape."""
    parts = decompose(ops[-1])
    assert parts is not None, "this aggregate should be mergeable"
    partial_ir, combine_ir, finalize_ir = parts
    below = ops[:-1]
    partials = [
        be.to_arrow(run_chain(s, [*below, partial_ir], be)) for s in _shards(table, shard_count)
    ]
    merged = pa.concat_tables(partials)
    return be.to_arrow(run_chain(merged, [combine_ir, finalize_ir], be))


@pytest.mark.parametrize("shard_count", [1, 2, 3, 7, 400])
@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.group_by("k").agg(s=col("v").sum()),
        lambda ds: ds.group_by("k").agg(n=col("v").count(), mn=col("v").min(), mx=col("v").max()),
        # a mean is not itself mergeable; the sum/count pair it is a ratio of is
        lambda ds: ds.group_by("k").agg(m=col("v").mean()),
        lambda ds: ds.group_by("k", "j").agg(s=col("v").sum(), m=col("v").mean(), c=bt.count()),
        lambda ds: ds.group_by(k2=col("k") + 1).agg(s=col("v").sum()),
        # a keyless aggregate folds through the same decomposition
        lambda ds: ds.agg(s=col("v").sum(), m=col("v").mean(), c=bt.count()),
        lambda ds: (
            ds.filter(col("v") > 20.0).group_by("k").agg(s=col("v").sum(), m=col("v").mean())
        ),
        lambda ds: ds.group_by("j").agg(p=col("v").product()),
    ],
)
def test_sharded_partials_fold_to_the_single_node_answer(build, shard_count, be):
    table = _table()
    ds = build(bt.from_arrow(table))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    got = _fold(table, spec[1], shard_count, be)
    expected = ds.collect()
    assert _rows(got.select(expected.column_names)) == _rows(expected)


@pytest.mark.parametrize("shard_count", [1, 2, 5])
@pytest.mark.parametrize(
    "build",
    [
        # a null key is a group, and it must survive both the partial and the combine
        lambda ds: ds.group_by("k").agg(s=col("v").sum(), c=col("v").count()),
        # an all-null group sums to null, and folding must not turn that into 0.0
        lambda ds: ds.group_by("k").agg(s=col("v").sum(), mn=col("v").min()),
        # a mean over a group with no non-null values is null, not a division by zero
        lambda ds: ds.group_by("k").agg(m=col("v").mean()),
    ],
)
def test_null_groups_survive_the_fold(build, shard_count, be):
    table = _nulls()
    ds = build(bt.from_arrow(table))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    got = _fold(table, spec[1], shard_count, be)
    expected = ds.collect()
    assert _rows(got.select(expected.column_names)) == _rows(expected)


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.group_by("k").agg(m=col("v").median()),
        lambda ds: ds.group_by("k").agg(q=col("v").quantile(0.9)),
        lambda ds: ds.group_by("k").agg(v=col("v").var()),
        lambda ds: ds.group_by("k").agg(sd=col("v").std()),
        lambda ds: ds.group_by("k").agg(d=col("v").count_distinct()),
        # one non-mergeable reduction disqualifies the whole node: the shards' partials for the
        # others would be fine, but there is no single fan-out that answers all of them
        lambda ds: ds.group_by("k").agg(s=col("v").sum(), m=col("v").median()),
    ],
)
def test_non_mergeable_reductions_are_declined(build):
    ds = build(bt.from_arrow(_table()))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the shape translates; only the fan-out does not apply"
    assert decompose(spec[1][-1]) is None


def test_partial_stage_is_smaller_than_its_input():
    """The fold is only worth doing because a partial is one row per group per shard.

    If a decomposition ever emitted something per *row*, the fan-out would move as much data as
    it saved and the scale argument would be gone.
    """
    table = _table()
    ds = bt.from_arrow(table).group_by("k").agg(s=col("v").sum(), m=col("v").mean())
    spec = gpu_plan_ops(ds._plan)
    partial_ir, _combine, _finalize = decompose(spec[1][-1])
    import pandas as pd

    be = DfBackend(pd)
    partial = be.to_arrow(run_chain(table, [partial_ir], be))
    assert partial.num_rows < table.num_rows
    assert partial.num_rows == len(set(table.column("k").to_pylist()))


# --- the generalized split: which operators may run per shard -------------------------


def _fold_split(table: pa.Table, ops: list[dict], shard_count: int, be) -> pa.Table:
    """Run `shard_ops` per shard, then `merge_ops + tail_ops` once — the distributed shape."""
    from batcher.plan.distribution import shard_plan

    split = shard_plan(ops)
    assert split is not None, "this chain should be shardable"
    shard_ops, merge_ops, tail_ops = split
    pieces = [be.to_arrow(run_chain(s, shard_ops, be)) for s in _shards(table, shard_count)]
    merged = pa.concat_tables([p for p in pieces if p.num_rows] or pieces[:1])
    return be.to_arrow(run_chain(merged, [*merge_ops, *tail_ops], be))


@pytest.mark.parametrize("shard_count", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.select("k", "j").distinct(),
        lambda ds: ds.filter(col("v") > 10.0).select("k").distinct(),
        # operators ABOVE the reducer run once on the folded result; requiring the reducer to
        # be the chain's last operator excluded this, the ordinary analytical shape
        lambda ds: ds.group_by("k").agg(s=col("v").sum()).filter(col("s") > 100.0),
        lambda ds: ds.group_by("k").agg(s=col("v").sum()).sort("s", descending=True).limit(3),
    ],
)
def test_generalized_split_matches_the_single_node_answer(build, shard_count, be):
    table = _table()
    ds = build(bt.from_arrow(table))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    got = _fold_split(table, spec[1], shard_count, be)
    expected = ds.collect()
    assert _rows(got.select(expected.column_names)) == _rows(expected)


@pytest.mark.parametrize("shard_count", [1, 2, 3, 5])
def test_sharded_top_n_keeps_the_single_node_order(shard_count, be):
    """A top-N is compared row-for-row: its order is the whole point of the operator."""
    table = _table()
    ds = bt.from_arrow(table).sort("v", descending=True).limit(9)
    spec = gpu_plan_ops(ds._plan)
    got = _fold_split(table, spec[1], shard_count, be)
    expected = ds.collect()

    def in_order(t):
        return [
            tuple(float(f"{v:.12e}") if isinstance(v, float) else v for v in row)
            for row in zip(*t.select(expected.column_names).to_pydict().values(), strict=True)
        ]

    assert in_order(got) == in_order(expected)


def test_only_row_local_operators_run_below_the_cut():
    """The split must cut at the FIRST non-row-local operator, not the last reducer.

    `sort(...).limit(10)` under an aggregate is the trap: each shard would contribute its own
    ten rows to a global ten that may all have come from one shard, and the aggregate over that
    is a confidently wrong number. Cutting at the sort makes the aggregate part of the tail,
    which runs once on the merged top-N.
    """
    from batcher.plan.distribution import shard_plan

    ds = (
        bt.from_arrow(_table())
        .filter(col("v") > 5.0)
        .sort("v", descending=True)
        .limit(10)
        .group_by("k")
        .agg(s=col("v").sum())
    )
    shard_ops, _merge, tail_ops = shard_plan(gpu_plan_ops(ds._plan)[1])
    assert [op["op"] for op in shard_ops] == ["filter", "sort", "limit"]
    assert [op["op"] for op in tail_ops] == ["aggregate"]


def test_a_map_only_chain_has_nothing_to_fold():
    """No reducer means no fan-out: the result is every input row, so folding buys nothing."""
    from batcher.plan.distribution import shard_plan

    ds = bt.from_arrow(_table()).filter(col("v") > 5.0).with_columns(w=col("v") * 2.0)
    assert shard_plan(gpu_plan_ops(ds._plan)[1]) is None


def test_a_limit_with_an_offset_does_not_shard():
    """A shard's rows 10..20 are not the global rows 10..20."""
    from batcher.plan.distribution import shard_plan

    ds = bt.from_arrow(_table()).sort("v").limit(5, offset=10)
    assert shard_plan(gpu_plan_ops(ds._plan)[1]) is None
