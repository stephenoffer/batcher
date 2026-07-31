"""Folding a fan-out's partials in waves gives the same answer as folding them at once.

The fan-out bounds device memory by dividing the input, and then handed every shard's output to
one driver-side concatenation — so a group-by over many groups across many shards materialized
all of them in one process, which is exactly the failure the sharding exists to prevent, moved
to the host. The wave fold keeps one accumulator instead.

The property that matters is not that it is smaller but that it is **exact**: the same rows,
for every wave size, every shard count, and every reduction the algebra admits. So these
compare against the single-shot merge and against the single-node answer rather than against
values written by hand — the single-shot merge is the thing already checked against DuckDB, and
equality with it inherits that.

Wave sizes are chosen to hit the shapes that break a naive implementation: a wave that divides
the shard count exactly, one that leaves a remainder of one, and one small enough to force
several rounds of re-folding, where a combine applied twice would look for columns its first
application consumed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.config import active_config, set_config
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain
from batcher.dist.gpu.aggregate import fold_shards, merge_shards
from batcher.plan.distribution import recombine, shard_plan

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


@pytest.fixture
def wave():
    """Set `gpu_merge_wave`, restoring whatever the session had."""
    saved = active_config()

    def _apply(size: int):
        cfg = active_config()
        set_config(
            cfg.replace(distributed=dataclasses.replace(cfg.distributed, gpu_merge_wave=size))
        )

    yield _apply
    set_config(saved)


def _table(rows: int = 600, groups: int = 17):
    rng = np.random.default_rng(11)
    return pa.table(
        {
            "k": rng.integers(0, groups, rows).astype("int64"),
            "j": rng.integers(0, 3, rows).astype("int64"),
            "v": rng.random(rows) * 100.0,
        }
    )


def _shards(table: pa.Table, k: int) -> list[pa.Table]:
    step = max(1, -(-table.num_rows // k))
    return [table.slice(i, min(step, table.num_rows - i)) for i in range(0, table.num_rows, step)]


def _rows(table: pa.Table) -> list[tuple]:
    """Rows canonicalized for comparison, order-independent and float-tolerant.

    Twelve significant digits rather than an absolute tolerance: a wave fold re-associates the
    float arithmetic, and an absolute epsilon fails on large magnitudes for a difference of one
    part in 1e15.
    """

    def canon(v):
        if isinstance(v, float):
            return "__nan__" if v != v else float(f"{v:.12e}")
        return v

    return sorted(
        (tuple(canon(v) for v in row) for row in zip(*table.to_pydict().values(), strict=True)),
        key=repr,
    )


def _partials(table: pa.Table, split, shard_count: int, be) -> list[pa.Table]:
    """Every shard's partial, exactly as the fan-out's tasks would produce them."""
    return [be.to_arrow(run_chain(s, split.shard_ops, be)) for s in _shards(table, shard_count)]


BUILDS = [
    lambda ds: ds.group_by("k").agg(s=col("v").sum()),
    lambda ds: ds.group_by("k").agg(n=col("v").count(), mn=col("v").min(), mx=col("v").max()),
    # a mean is a ratio of two mergeable parts; the division must run exactly once
    lambda ds: ds.group_by("k").agg(m=col("v").mean()),
    lambda ds: ds.group_by("k", "j").agg(s=col("v").sum(), m=col("v").mean(), c=bt.count()),
    # a keyless aggregate folds through the same decomposition
    lambda ds: ds.agg(s=col("v").sum(), m=col("v").mean(), c=bt.count()),
    lambda ds: ds.group_by("j").agg(p=col("v").product()),
]


@pytest.mark.parametrize("wave_size", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("shard_count", [1, 2, 5, 9, 16])
@pytest.mark.parametrize("build", BUILDS)
def test_a_wave_fold_equals_the_single_node_answer(build, shard_count, wave_size, wave, be):
    wave(wave_size)
    table = _table()
    ds = build(bt.from_arrow(table))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    split = shard_plan(spec[1])
    assert split is not None and split.foldable

    got = fold_shards(_partials(table, split, shard_count, be), split)
    expected = ds.collect()
    assert _rows(got.select(expected.column_names)) == _rows(expected)


@pytest.mark.parametrize("wave_size", [2, 3, 7])
@pytest.mark.parametrize("build", BUILDS)
def test_a_wave_fold_equals_folding_every_partial_at_once(build, wave_size, wave, be):
    """The wave form and the single-shot form are the same function, not merely close."""
    table = _table()
    ds = build(bt.from_arrow(table))
    split = shard_plan(gpu_plan_ops(ds._plan)[1])
    partials = _partials(table, split, 11, be)

    wave(0)
    once = fold_shards(partials, split)
    wave(wave_size)
    waved = fold_shards(partials, split)
    assert _rows(waved.select(once.column_names)) == _rows(once)


def test_a_distinct_folds_in_waves_without_a_second_form(wave, be):
    """Deduplication is idempotent, so its fold repeats with no rewriting at all."""
    wave(2)
    table = _table(200, groups=5)
    ds = bt.from_arrow(table).select("k", "j").distinct()
    split = shard_plan(gpu_plan_ops(ds._plan)[1])
    assert split.foldable
    assert split.refold_ops == split.fold_ops

    got = fold_shards(_partials(table, split, 7, be), split)
    assert _rows(got.select(ds.collect().column_names)) == _rows(ds.collect())


def test_a_row_local_chain_is_not_wave_folded_because_its_order_is_the_answer(wave, be):
    """A concatenation may not be reassociated: the shards' order *is* the result."""
    wave(2)
    table = _table(120)
    ds = bt.from_arrow(table).filter(col("v") > 20.0)
    split = shard_plan(gpu_plan_ops(ds._plan)[1])
    assert not split.foldable, "a row-local chain has no repeatable fold"

    partials = _partials(table, split, 9, be)
    assert fold_shards(partials, split).equals(
        merge_shards(partials, [*split.merge_ops, *split.tail_ops])
    )


def test_a_fan_out_smaller_than_one_wave_takes_the_single_shot_path(wave, be):
    wave(32)
    table = _table(100)
    ds = bt.from_arrow(table).group_by("k").agg(s=col("v").sum())
    split = shard_plan(gpu_plan_ops(ds._plan)[1])
    partials = _partials(table, split, 4, be)
    assert _rows(fold_shards(partials, split)) == _rows(
        merge_shards(partials, [*split.merge_ops, *split.tail_ops])
    )


def test_the_finalize_is_the_part_of_the_merge_the_fold_does_not_cover() -> None:
    """A mean's division belongs after the last fold, never inside it."""
    ds = bt.from_pydict({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]}).group_by("k").agg(m=col("v").mean())
    split = shard_plan(gpu_plan_ops(ds._plan)[1])
    assert split.fold_ops == split.merge_ops[: len(split.fold_ops)]
    assert split.finalize_ops == split.merge_ops[len(split.fold_ops) :]
    assert [op["op"] for op in split.finalize_ops] == ["project"]


def test_a_recombine_reads_the_columns_its_combine_wrote() -> None:
    """The one way the wave fold could be wrong is reading the wrong column names."""
    combine = {
        "op": "aggregate",
        "group_keys": [{"expr": {"e": "col", "name": "k"}, "alias": "k"}],
        "aggregates": [
            {"func": "sum", "alias": "total", "input": {"e": "col", "name": "__bt_pa0"}},
            {"func": "min", "alias": "low", "input": {"e": "col", "name": "__bt_pa1"}},
        ],
    }
    again = recombine(combine)
    assert again["group_keys"] == combine["group_keys"]
    assert [a["input"]["name"] for a in again["aggregates"]] == ["total", "low"]
    assert [a["func"] for a in again["aggregates"]] == ["sum", "min"]
    assert [a["alias"] for a in again["aggregates"]] == ["total", "low"]
