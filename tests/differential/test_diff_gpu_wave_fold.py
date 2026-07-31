"""The GPU fan-out's wave-folded merge, checked against DuckDB rather than against itself.

`tests/unit/test_gpu_merge_waves.py` proves the wave fold equals the single-shot fold. That is
necessary and not sufficient: both could be wrong the same way, because both are built from the
same decomposition. This runs the whole sharded shape — partial per shard, fold in waves,
finalize once — and compares it against the oracle, which is what makes the wave form as
trustworthy as the merge it replaces.

The shard bodies run on the translator's **host** backend rather than on a device, exactly as
the existing mergeable tests do. What is under test is the driver's merge, which is host code on
every run; a GPU would change which kernel produced a partial and not what the fold does with
it, so requiring one here would mean the merge is checked nowhere.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher.config import active_config, set_config
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain
from batcher.dist.gpu.aggregate import fold_shards
from batcher.plan.distribution import shard_plan

pytestmark = pytest.mark.differential


@pytest.fixture(scope="module")
def be():
    pd = pytest.importorskip("pandas")
    return DfBackend(pd)


@pytest.fixture
def wave():
    saved = active_config()

    def _apply(size: int):
        cfg = active_config()
        set_config(
            cfg.replace(distributed=dataclasses.replace(cfg.distributed, gpu_merge_wave=size))
        )

    yield _apply
    set_config(saved)


def _table(rows: int = 900):
    """Random data whose float column sums *exactly*, whatever order it is added in.

    A mergeable fold re-associates the addition: it sums each shard, then sums the sums, so the
    last bit of a long double sum differs from the oracle's single left-to-right pass. That is a
    property of the algebra rather than of this change, and comparing against DuckDB through a
    tolerance would blunt the only test that can catch a real fold bug. Quarters are exactly
    representable in binary and stay so under any grouping at this magnitude, so equality here
    means equality, and the float re-association itself is covered to twelve significant digits
    by `tests/unit/test_gpu_merge_waves.py`.
    """
    rng = np.random.default_rng(3)
    return pa.table(
        {
            "k": rng.integers(0, 23, rows).astype("int64"),
            "g": rng.integers(0, 4, rows).astype("int64"),
            "v": rng.integers(0, 40_000, rows).astype("float64") / 4.0,
            "n": rng.integers(-50, 50, rows).astype("int64"),
        }
    )


def _shards(table: pa.Table, k: int):
    step = max(1, -(-table.num_rows // k))
    return [table.slice(i, min(step, table.num_rows - i)) for i in range(0, table.num_rows, step)]


def _fan_out(table: pa.Table, ds, shard_count: int, be) -> pa.Table:
    """The whole sharded shape: partial per shard, wave fold, finalize once."""
    split = shard_plan(gpu_plan_ops(ds._plan)[1])
    assert split is not None, "this chain should shard"
    partials = [be.to_arrow(run_chain(s, split.shard_ops, be)) for s in _shards(table, shard_count)]
    return fold_shards(partials, split)


QUERIES = [
    (
        lambda ds: ds.group_by("k").agg(s=col("v").sum()),
        "SELECT k, sum(v) AS s FROM t GROUP BY k",
    ),
    (
        lambda ds: ds.group_by("k").agg(c=col("v").count(), lo=col("v").min(), hi=col("v").max()),
        "SELECT k, count(v) AS c, min(v) AS lo, max(v) AS hi FROM t GROUP BY k",
    ),
    (
        lambda ds: ds.group_by("k").agg(m=col("v").mean()),
        "SELECT k, avg(v) AS m FROM t GROUP BY k",
    ),
    (
        lambda ds: ds.group_by("k", "g").agg(s=col("n").sum(), m=col("v").mean(), c=bt.count()),
        "SELECT k, g, sum(n) AS s, avg(v) AS m, count(*) AS c FROM t GROUP BY k, g",
    ),
    (
        lambda ds: ds.agg(s=col("v").sum(), c=bt.count()),
        "SELECT sum(v) AS s, count(*) AS c FROM t",
    ),
    (
        lambda ds: ds.filter(col("v") > 2000.0).group_by("g").agg(s=col("v").sum()),
        "SELECT g, sum(v) AS s FROM t WHERE v > 2000.0 GROUP BY g",
    ),
]


@pytest.mark.parametrize("wave_size", [2, 3, 8])
@pytest.mark.parametrize("shard_count", [1, 4, 13])
@pytest.mark.parametrize(("build", "sql"), QUERIES)
def test_the_wave_folded_fan_out_matches_duckdb(build, sql, shard_count, wave_size, wave, be, duck):
    wave(wave_size)
    table = _table()
    duck.register("t", table)
    got = _fan_out(table, build(bt.from_arrow(table)), shard_count, be)
    assert_same(got, duck.sql(sql))


@pytest.mark.parametrize("wave_size", [2, 5])
def test_a_wave_fold_over_null_keys_and_all_null_groups_matches_duckdb(wave_size, wave, be, duck):
    """A null key is a group, and an all-null group's sum is null, not zero — through every fold."""
    wave(wave_size)
    table = pa.table(
        {
            "k": pa.array([1, 1, None, 2, 2, None, 3, 3, 4], type=pa.int64()),
            "v": pa.array([1.0, None, 3.0, 4.0, 5.0, None, None, None, 6.0], type=pa.float64()),
        }
    )
    duck.register("t", table)
    ds = bt.from_arrow(table).group_by("k").agg(s=col("v").sum(), c=col("v").count())
    assert_same(
        _fan_out(table, ds, 5, be),
        duck.sql("SELECT k, sum(v) AS s, count(v) AS c FROM t GROUP BY k"),
    )


@pytest.mark.parametrize("wave_size", [2, 4])
def test_a_wave_fold_over_one_row_per_shard_matches_duckdb(wave_size, wave, be, duck):
    """The degenerate fan-out: every shard is one row, so every wave is a real re-fold."""
    wave(wave_size)
    table = _table(24)
    duck.register("t", table)
    ds = bt.from_arrow(table).group_by("k").agg(s=col("v").sum(), m=col("v").mean())
    assert_same(
        _fan_out(table, ds, 24, be),
        duck.sql("SELECT k, sum(v) AS s, avg(v) AS m FROM t GROUP BY k"),
    )
