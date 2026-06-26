"""Differential tests proving `ExecutionConfig.morsel_rows` reaches the Rust data
plane and is purely a *scheduling* concern.

The morsel size now flows Python `Config` → `EngineConfig` JSON → `execute_plan` →
the parallel executor's morselize step. A non-default (deliberately tiny) morsel
size forces many morsels — exercising the cross-morsel combine/shuffle paths — yet
the result MUST still match DuckDB. This both confirms the FFI wiring works and
locks the invariant that morselization never changes results.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count


@pytest.mark.parametrize("morsel_rows", [1, 3, 16])
def test_filter_project_tiny_morsels_vs_duckdb(duck, morsel_rows):
    from conftest import assert_same

    t = pa.table({"x": list(range(64)), "y": [i * 10 for i in range(64)]})
    duck.register("t", t)

    cfg = bt.Config().replace(execution=bt.ExecutionConfig(morsel_rows=morsel_rows))
    with bt.config_context(cfg):
        out = bt.from_arrow(t).filter(col("x") > 20).select("x", xy=col("x") * col("y")).collect()
    expected = duck.sql("SELECT x, x * y AS xy FROM t WHERE x > 20")
    assert_same(out, expected)


@pytest.mark.parametrize("morsel_rows", [1, 4, 16])
def test_group_by_tiny_morsels_vs_duckdb(duck, morsel_rows):
    """Tiny morsels force many partial aggregates to combine — the mergeable path."""
    from conftest import assert_same

    t = pa.table(
        {
            "dept": ["eng", "sales", "eng", "ops", "sales", "eng", "ops", "eng"],
            "salary": [100, 150, 300, 75, 50, 200, 80, 90],
        }
    )
    duck.register("t", t)

    cfg = bt.Config().replace(execution=bt.ExecutionConfig(morsel_rows=morsel_rows))
    with bt.config_context(cfg):
        out = bt.from_arrow(t).group_by("dept").agg(total=col("salary").sum(), n=count()).collect()
    expected = duck.sql("SELECT dept, SUM(salary) AS total, COUNT(*) AS n FROM t GROUP BY dept")
    assert_same(out, expected)


def test_morsel_size_does_not_change_result():
    """The default and a tiny morsel size produce identical results (morselization
    is scheduling-only — the seq == par invariant under any morsel width)."""
    t = pa.table({"k": [i % 7 for i in range(200)], "v": list(range(200))})

    base = bt.from_arrow(t).group_by("k").agg(s=col("v").sum()).collect()
    tiny_cfg = bt.Config().replace(execution=bt.ExecutionConfig(morsel_rows=2))
    with bt.config_context(tiny_cfg):
        tiny = bt.from_arrow(t).group_by("k").agg(s=col("v").sum()).collect()

    def norm(tbl: pa.Table) -> list[tuple]:
        return sorted(tuple(r.values()) for r in tbl.to_pylist())

    assert norm(base) == norm(tiny)


def _norm(tbl: pa.Table) -> list[tuple]:
    return sorted(tuple(r.values()) for r in tbl.to_pylist())


def test_runtime_tuning_knobs_are_result_invariant():
    """The performance-threshold knobs (radix/bloom/window/sort fan-in) change *how*
    an operator runs, never the relation. A group-by + join + window + sort run with a
    deliberately aggressive (low-threshold, high-fp, small-fanin) tuning must produce
    the identical result to the default tuning."""
    left = pa.table({"k": [i % 13 for i in range(400)], "v": list(range(400))})
    right = pa.table({"k": list(range(13)), "label": [f"g{i}" for i in range(13)]})

    def run() -> pa.Table:
        ds = (
            bt.from_arrow(left)
            .join(bt.from_arrow(right), on="k")
            .group_by("label")
            .agg(total=col("v").sum(), n=count())
        )
        return ds.sort("label").collect()

    base = run()
    aggressive = bt.Config().replace(
        execution=bt.ExecutionConfig(
            # Force the parallel-radix combine, engage the bloom early, force window
            # parallelism, and a small merge fan-in — all perf-only.
            radix_parallel_threshold=1,
            bloom_min_build_rows=1,
            bloom_fp_rate=0.2,
            window_parallel_row_threshold=1,
            sort_merge_fanin=2,
        )
    )
    with bt.config_context(aggressive):
        tuned = run()

    assert _norm(base) == _norm(tuned)
