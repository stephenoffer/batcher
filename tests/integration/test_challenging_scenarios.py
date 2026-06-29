"""Out-of-the-box survival on the hardest single-node shapes — the "challenging
situations" catalog: skew, row/size explosion, downloading/exploding UDFs, spill under
a tight budget, and degenerate edges. Each must produce the *correct* result with **no
tuning** (zero config), so a regular user never hits an OOM, a hang, or a wrong answer.

Single-node is the out-of-the-box path; the distributed path reuses the same mergeable
primitives and the same `core.execute_with_udfs`, so a bound proven here (e.g. the UDF
morsel bound) holds on the cluster too. Scale is kept modest so the suite is fast while
still exercising spill (a tight `max_memory_bytes`) and the unbounded-batch hazards.
"""

from __future__ import annotations

import time

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.config import Config, MemoryConfig, config_context

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

# A tight spill budget so the breaker scenarios must spill, not balloon in memory.
_TIGHT = Config().replace(memory=MemoryConfig(max_memory_bytes=32 << 20))
_N = 200_000


# --- Skew -----------------------------------------------------------------------


def test_skew_single_group():
    out = (
        bt.from_pydict({"k": [1] * _N, "v": [float(i) for i in range(_N)]})
        .group_by("k")
        .agg(s=col("v").sum())
        .collect()
    )
    assert out.num_rows == 1
    assert out.to_pydict()["s"][0] == sum(float(i) for i in range(_N))


def test_skew_hot_join_key_quadratic_output():
    # One key matching 400 x 400 → 160k output rows; must not wrongly dedup or drop.
    left = bt.from_pydict({"k": [1] * 400, "a": list(range(400))})
    right = bt.from_pydict({"k": [1] * 400, "b": list(range(400))})
    out = left.join(right, on="k").collect()
    assert out.num_rows == 160_000


def test_skew_powerlaw_groupby():
    keys = [0] * (_N * 9 // 10) + list(range(_N // 10))
    out = bt.from_pydict({"k": keys, "v": [1.0] * _N}).group_by("k").agg(s=col("v").sum()).collect()
    assert out.num_rows == _N // 10  # range(_N//10) already includes key 0


# --- Row / size explosion (must stay bounded & correct) -------------------------


def test_row_explosion_udf_one_to_many():
    def expand(b: pa.RecordBatch) -> pa.RecordBatch:
        vals = b.column(0).to_pylist()
        return pa.record_batch({"x": pa.array([v for v in vals for _ in range(20)], pa.int64())})

    out = bt.from_pydict({"x": list(range(10_000))}).map_batches(expand).collect()
    assert out.num_rows == 200_000


def test_wide_row_string_repeat():
    out = (
        bt.from_pydict({"s": ["ab"] * 20_000})
        .with_columns(big=col("s").str.repeat(2000))
        .select("big")
        .collect()
    )
    assert out.num_rows == 20_000
    assert len(out.to_pydict()["big"][0]) == 4000


# --- UDFs: downloads & the unbounded-batch bound (the regression) ---------------


def test_plain_udf_input_is_bounded_to_a_morsel():
    # The breaking point fixed: a plain-function `map_batches` with no `batch_size` must
    # NOT receive the whole partition as one batch (a downloading / exploding `fn` would
    # OOM). It is bounded to the engine morsel size out of the box.
    from batcher.config import active_config

    morsel = active_config().execution.morsel_rows
    seen: list[int] = []

    def probe(b: pa.RecordBatch) -> pa.RecordBatch:
        seen.append(b.num_rows)
        return b

    out = bt.from_pydict({"x": list(range(100_000))}).map_batches(probe).collect()
    assert out.num_rows == 100_000
    assert seen, "the UDF must be called"
    assert max(seen) <= morsel, f"a plain UDF must see at most one morsel, saw {max(seen)}"


def test_explicit_batch_size_still_honored():
    seen: list[int] = []

    def probe(b: pa.RecordBatch) -> pa.RecordBatch:
        seen.append(b.num_rows)
        return b

    bt.from_pydict({"x": list(range(50_000))}).map_batches(probe, batch_size=4096).collect()
    assert max(seen) <= 4096


def test_downloading_udf_stays_bounded_and_correct():
    # A per-batch "download" (simulated latency) over a large input runs in bounded
    # chunks and preserves every row.
    def fake_download(b: pa.RecordBatch) -> pa.RecordBatch:
        time.sleep(0.0005)  # stand-in for a network fetch per batch
        vals = b.column(0).to_pylist()
        return pa.record_batch({"x": pa.array([v + 1 for v in vals], pa.int64())})

    out = bt.from_pydict({"x": list(range(50_000))}).map_batches(fake_download).collect()
    assert out.num_rows == 50_000
    assert sorted(out.to_pydict()["x"]) == list(range(1, 50_001))


def test_udf_error_surfaces_clearly():
    def boom(_b: pa.RecordBatch) -> pa.RecordBatch:
        raise ValueError("intentional UDF failure")

    with pytest.raises((ValueError, RuntimeError)):
        bt.from_pydict({"x": list(range(100))}).map_batches(boom).collect()


# --- Spill under a tight memory budget (survive + correct) ----------------------


def test_spill_groupby_high_cardinality():
    with config_context(_TIGHT):
        out = (
            bt.from_pydict({"k": list(range(_N)), "v": [1.0] * _N})
            .group_by("k")
            .agg(s=col("v").sum())
            .collect()
        )
    assert out.num_rows == _N


def test_spill_external_sort():
    with config_context(_TIGHT):
        out = bt.from_pydict({"v": [float((i * 7919) % _N) for i in range(_N)]}).sort("v").collect()
    vals = out.to_pydict()["v"]
    assert out.num_rows == _N
    assert vals == sorted(vals)


def test_spill_distinct_all_unique():
    with config_context(_TIGHT):
        out = bt.from_pydict({"k": list(range(_N))}).distinct().collect()
    assert out.num_rows == _N


def test_spill_join_large_build_side():
    with config_context(_TIGHT):
        left = bt.from_pydict({"k": list(range(_N)), "a": [1] * _N})
        right = bt.from_pydict({"k": list(range(_N)), "b": [2] * _N})
        out = left.join(right, on="k").collect()
    assert out.num_rows == _N


# --- Degenerate edges -----------------------------------------------------------


def test_edge_empty_input():
    out = (
        bt.from_pydict({"k": pa.array([], pa.int64()), "v": pa.array([], pa.float64())})
        .group_by("k")
        .agg(s=col("v").sum())
        .collect()
    )
    assert out.num_rows == 0


def test_edge_single_row():
    out = bt.from_pydict({"k": [1], "v": [2.0]}).group_by("k").agg(s=col("v").sum()).collect()
    assert out.num_rows == 1


def test_edge_filter_removes_everything():
    out = bt.from_pydict({"v": list(range(_N))}).filter(col("v") > 10**9).collect()
    assert out.num_rows == 0


def test_edge_all_null_aggregation():
    out = (
        bt.from_pydict({"k": [1] * 1000, "v": pa.array([None] * 1000, pa.float64())})
        .group_by("k")
        .agg(s=col("v").sum())
        .collect()
    )
    assert out.num_rows == 1


def test_edge_null_heavy_join_keys():
    # Null keys never match in an equi-join (SQL semantics) — must not crash or over-join.
    left = bt.from_pydict(
        {"k": pa.array([None] * 500 + list(range(500)), pa.int64()), "a": list(range(1000))}
    )
    right = bt.from_pydict(
        {"k": pa.array([None] * 500 + list(range(500)), pa.int64()), "b": list(range(1000))}
    )
    out = left.join(right, on="k").collect()
    assert out.num_rows == 500  # only the 500 non-null keys match, one-to-one


# --- Types ----------------------------------------------------------------------


def test_type_very_long_strings():
    out = bt.from_pydict({"s": ["x" * 500_000] * 200}).with_columns(n=col("s").str.len()).collect()
    assert out.num_rows == 200
    assert out.to_pydict()["n"][0] == 500_000


def test_type_wide_table_many_columns():
    out = bt.from_pydict({f"c{i}": list(range(1000)) for i in range(300)}).collect()
    assert out.num_rows == 1000
    assert out.num_columns == 300
