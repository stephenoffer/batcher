"""Result-invariance of the conductor's adaptive-tuning wiring, cross-checked vs DuckDB.

The conductor activates learned decisions (adaptive gate, join-strategy bandit, worker/partition
fan-out, spill codec, credit windows) and records measured outcomes back into the hub. Every one
of those is a PERFORMANCE/SCHEDULING lever — it must produce a byte-identical result whether it
fires or not. These tests force each lever on and prove the answer equals both DuckDB and the
un-tuned default.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _norm(d: dict) -> list:
    return sorted(zip(*[d[c] for c in sorted(d)], strict=True))


# --- 1 + 2. adaptive gate (learned on) is result-invariant -----------------------------------
def test_learned_adaptive_gate_matches_one_shot_and_duckdb(duck):
    from conftest import assert_same

    left = pa.table({"k": [1, 2, 3, 1, 2, 3], "v": [1, 2, 3, 4, 5, 6]})
    right = pa.table({"k": [1, 2, 3], "w": [10, 20, 30]})
    duck.register("l", left)
    duck.register("r", right)
    dl, dr = bt.from_arrow(left), bt.from_arrow(right)

    def q():
        return dl.group_by("k").agg(s=col("v").sum()).join(dr, on="k")

    auto = q().collect(adaptive="auto")  # structural gate + any learned flips
    one_shot = q().collect(adaptive=False)
    assert _norm(auto.to_pydict()) == _norm(one_shot.to_pydict())
    assert_same(
        auto,
        duck.sql(
            "SELECT s.k, s.s, r.w FROM (SELECT k, SUM(v) s FROM l GROUP BY k) s JOIN r ON s.k = r.k"
        ),
    )


# --- 5 + 6 + 7. forcing any learned join arm keeps the join equal to DuckDB -------------------
@pytest.mark.parametrize("arm", ["hash", "broadcast", "sort_merge"])
def test_learned_join_arm_is_result_invariant(duck, monkeypatch, arm):
    from batcher.kyber.rules import selection
    from conftest import assert_same

    monkeypatch.setattr(selection, "learned_join_strategy", lambda hub, sig, *a, **k: arm)
    emp = pa.table({"id": [1, 2, 3, 4, 5], "dept_id": [10, 20, 10, 99, 20]})
    dept = pa.table({"dept_id": [10, 20, 30], "dept": ["eng", "sales", "ops"]})
    duck.register("emp", emp)
    duck.register("dept", dept)
    joined = bt.from_arrow(emp).join(bt.from_arrow(dept), on="dept_id")
    got = joined.collect()
    default = bt.from_arrow(emp).join(bt.from_arrow(dept), on="dept_id").collect()
    assert _norm(got.to_pydict()) == _norm(default.to_pydict())
    assert_same(got, duck.sql("SELECT * FROM emp JOIN dept USING (dept_id)"))


# --- 4. spill compression codec is lossless (spilled result is byte-identical) ---------------
@pytest.mark.parametrize("codec", ["zstd", "lz4", None])
def test_spill_compression_is_result_invariant(duck, codec):
    from batcher import kyber
    from batcher.config import active_config, config_context
    from batcher.dist.spill import spill_collect
    from conftest import assert_same

    tbl = pa.table({"k": [i % 7 for i in range(400)], "v": list(range(400))})
    duck.register("t", tbl)
    ds = bt.from_arrow(tbl).group_by("k").agg(s=col("v").sum())
    opt_lp = kyber.optimize_logical(ds._plan, sources=ds._sources, hub=None)

    cfg = active_config()
    mem = dataclasses.replace(cfg.memory, spill_compression=codec)
    with config_context(dataclasses.replace(cfg, memory=mem)):
        spilled = spill_collect(opt_lp, ds._sources, 8)
    assert spilled is not None
    assert_same(spilled, duck.sql("SELECT k, SUM(v) s FROM t GROUP BY k"))


# --- 3 + 9 + 10 + 13. the full record→learn→consume loop stays correct end to end ------------
def test_second_run_uses_learned_feedback_and_stays_correct(duck):
    """A first run records feedback (partition rows, group reduction, join arm, credit window);
    the second run consumes it. Both must equal DuckDB — the loop only tunes performance."""
    from conftest import assert_same

    left = pa.table({"k": [i % 5 for i in range(50)], "v": list(range(50))})
    right = pa.table({"k": [0, 1, 2, 3, 4], "w": [100, 200, 300, 400, 500]})
    duck.register("l", left)
    duck.register("r", right)
    dl, dr = bt.from_arrow(left), bt.from_arrow(right)

    def q():
        return dl.group_by("k").agg(s=col("v").sum()).join(dr, on="k")

    expected = duck.sql(
        "SELECT s.k, s.s, r.w FROM (SELECT k, SUM(v) s FROM l GROUP BY k) s JOIN r ON s.k = r.k"
    )
    first = q().collect()  # cold: records the measured outcomes into the hub
    second = q().collect()  # warm: consumes the learned partition/agg/strategy signals
    assert_same(first, expected)
    assert_same(second, expected)
    assert _norm(first.to_pydict()) == _norm(second.to_pydict())
