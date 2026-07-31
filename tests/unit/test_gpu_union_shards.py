"""The union fan-out: what it divides into, what it declines, and that the pieces reassemble.

A union was the last relational shape pinned to one device. The properties worth pinning are
the two that a fan-out can get wrong without erroring: the shards must reassemble into the
single-node answer *in order*, and the shapes whose algebra does not divide must decline rather
than produce something plausible.

No Ray and no GPU here. The planning half — which inputs shard, into how many pieces, carrying
which operator chain — is a pure function of the spec, and the executing half is the chain
fan-out's, which its own tests cover.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_union_spec
from batcher.core.gpu_plan.execute import run_union
from batcher.dist.gpu.union import _shard_plan_per_input, sharded_gpu_union
from batcher.plan.distribution import ShardSplit, shard_plan

pytestmark = pytest.mark.unit


@pytest.fixture
def be():
    import pandas as pd

    return DfBackend(pd)


A = pa.table({"k": ["a", "b", "a"], "v": [1.0, 2.0, 3.0]})
B = pa.table({"k": ["b", "c"], "v": [4.0, 5.0]})
C = pa.table({"k": ["a", "d"], "v": [6.0, 7.0]})


def _spec(build, *tables):
    """The translator's union spec for `build` over `tables`."""
    dss = [bt.from_arrow(t) for t in tables]
    return gpu_union_spec(build(dss[0].union(*dss[1:]))._plan)


# --- what the fan-out declines --------------------------------------------------------


def test_a_deduplicating_union_declines_the_fan_out():
    """Slice-wise dedup is exact only while nothing reduces above it.

    With an aggregate on top, a row appearing in two shards survives both slice-wise dedups
    and is counted twice — which the merge can no longer see. The honest fix is a hash shuffle
    on the whole row, so this declines and the single-device path runs it.
    """
    assert sharded_gpu_union([], [], True, [], gpu_count=8, sharded=True) is None


def test_a_union_with_no_inputs_has_nothing_to_shard():
    assert sharded_gpu_union([], [], False, [], gpu_count=8, sharded=True) is None


def test_an_unshardable_chain_above_the_union_declines(monkeypatch):
    """`shard_plan` returning `None` means the chain has no split; the union inherits that."""
    monkeypatch.setattr("batcher.plan.distribution.shard_plan", lambda ops: None)
    ops = [{"op": "limit", "n": 5}]
    assert sharded_gpu_union(["s"], [[]], False, ops, gpu_count=8, sharded=True) is None


# --- what the fan-out plans -----------------------------------------------------------


class _Source:
    """A stand-in source whose splits are known, so the planning half needs no storage."""

    def __init__(self, name: str, shards: int):
        self.name = name
        self.shards = shards


def _fake_descriptors(monkeypatch, per_source: dict[str, int] | None = None):
    """Make `shard_descriptors` answer from the stand-in rather than from Ray."""
    calls: list[tuple[str, int, bool]] = []

    def _fake(source, gpu_count, *, sharded, preserve_order):
        calls.append((source.name, gpu_count, preserve_order))
        n = (per_source or {}).get(source.name, source.shards)
        return None if n == 0 else [{"src": source.name, "i": i} for i in range(n)]

    monkeypatch.setattr("batcher.dist.gpu.aggregate.shard_descriptors", _fake)
    return calls


def test_every_input_contributes_its_shards_in_input_order(monkeypatch):
    _fake_descriptors(monkeypatch)
    above = ShardSplit([], [], [], ordered=True)
    plan, _bytes = _shard_plan_per_input(
        [_Source("a", 2), _Source("b", 3)], [[], []], above, 8, sharded=True
    )
    assert [d["src"] for d, _ in plan] == ["a", "a", "b", "b", "b"]
    assert [d["i"] for d, _ in plan] == [0, 1, 0, 1, 2]


def test_each_shard_carries_its_own_inputs_chain_then_the_shared_one(monkeypatch):
    """One task body serves every input, so the chain has to travel with the shard."""
    _fake_descriptors(monkeypatch)
    left_chain = [{"op": "filter", "predicate": {"e": "col", "name": "x"}}]
    above = ShardSplit([{"op": "project", "exprs": []}], [], [], ordered=False)
    plan, _bytes = _shard_plan_per_input(
        [_Source("a", 1), _Source("b", 1)], [left_chain, []], above, 8, sharded=True
    )
    assert plan[0][1] == [*left_chain, *above.shard_ops]
    assert plan[1][1] == list(above.shard_ops)


def test_the_device_budget_is_divided_across_the_inputs(monkeypatch):
    """Four inputs each asking for the whole cluster would ask for four clusters."""
    calls = _fake_descriptors(monkeypatch)
    above = ShardSplit([], [], [], ordered=True)
    _shard_plan_per_input([_Source(n, 1) for n in "abcd"], [[]] * 4, above, 8, sharded=True)
    assert [gpu_count for _, gpu_count, _ in calls] == [2, 2, 2, 2]


def test_the_budget_never_falls_below_one_device(monkeypatch):
    calls = _fake_descriptors(monkeypatch)
    above = ShardSplit([], [], [], ordered=True)
    _shard_plan_per_input([_Source(n, 1) for n in "abcd"], [[]] * 4, above, 1, sharded=True)
    assert all(gpu_count >= 1 for _, gpu_count, _ in calls)


def test_one_unsplittable_input_declines_the_whole_fan_out(monkeypatch):
    """Its size is exactly why the fan-out was wanted, so running it whole beside the rest
    is not the compromise it looks like."""
    _fake_descriptors(monkeypatch, per_source={"b": 0})
    above = ShardSplit([], [], [], ordered=True)
    plan = _shard_plan_per_input(
        [_Source("a", 2), _Source("b", 0)], [[], []], above, 8, sharded=True
    )
    assert plan is None


def test_an_ordered_merge_asks_for_ordered_splits(monkeypatch):
    """A concatenation reproduces the single-node answer only if the slices are contiguous."""
    calls = _fake_descriptors(monkeypatch)
    _shard_plan_per_input(
        [_Source("a", 1)], [[]], ShardSplit([], [], [], ordered=True), 4, sharded=True
    )
    assert calls[-1][2] is True

    calls.clear()
    _shard_plan_per_input(
        [_Source("a", 1)], [[]], ShardSplit([], [], [], ordered=False), 4, sharded=True
    )
    assert calls[-1][2] is False


# --- the pieces reassemble ------------------------------------------------------------


def _run_sharded(tables, input_ops, ops, be, slices: int):
    """Compute the fan-out's answer by hand: each input sliced, then merged the way it merges."""
    from batcher.core.gpu_plan.execute import run_ops
    from batcher.dist.gpu.aggregate import merge_shards

    above = shard_plan(ops) if ops else ShardSplit([], [], [], ordered=True)
    shards = []
    for table, chain in zip(tables, input_ops, strict=True):
        step = max(1, -(-table.num_rows // slices))
        for start in range(0, table.num_rows, step):
            piece = table.slice(start, step)
            out = run_ops(be.from_arrow(piece), [*chain, *above.shard_ops], be)
            shards.append(be.to_arrow(out))
    return merge_shards(shards, [*above.merge_ops, *above.tail_ops])


@pytest.mark.parametrize("slices", [1, 2, 3])
@pytest.mark.parametrize(
    ("build", "ordered"),
    [
        # `ordered` is stated per case rather than inferred: a fold may collect its shards in
        # any order and a concatenation may not, and reading one as the other is exactly the
        # bug an order-independent comparison cannot see.
        (lambda ds: ds, True),
        (lambda ds: ds.filter(col("v") > 2.0), True),
        (lambda ds: ds.agg(total=col("v").sum(), n=col("v").count()), False),
        (lambda ds: ds.group_by("k").agg(total=col("v").sum()), False),
    ],
)
def test_sharding_a_union_equals_running_it_whole(be, build, ordered, slices):
    """The property the fan-out exists to preserve, across every shard count."""
    spec = _spec(build, A, B, C)
    assert spec is not None, "shape should be GPU-translatable"
    inputs, distinct, ops = spec
    assert not distinct
    input_ops = [o for _, o in inputs]

    whole = be.to_arrow(run_union([A, B, C], input_ops, distinct, ops, be))
    fanned = _run_sharded([A, B, C], input_ops, ops, be, slices)

    got = [repr(r) for r in fanned.to_pylist()]
    want = [repr(r) for r in whole.select(fanned.column_names).to_pylist()]
    assert (got == want) if ordered else (sorted(got) == sorted(want))


@pytest.mark.parametrize("slices", [1, 2, 3])
def test_a_row_local_union_reassembles_in_order(be, slices):
    """`UNION ALL` with no reducer is a concatenation, and its order is the answer."""
    spec = _spec(lambda d: d.filter(col("v") > 0.0), A, B, C)
    inputs, distinct, ops = spec
    input_ops = [o for _, o in inputs]
    fanned = _run_sharded([A, B, C], input_ops, ops, be, slices)
    whole = be.to_arrow(run_union([A, B, C], input_ops, distinct, ops, be))
    assert [repr(r) for r in fanned.to_pylist()] == [
        repr(r) for r in whole.select(fanned.column_names).to_pylist()
    ]
