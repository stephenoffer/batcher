"""Differential coverage for the bool_and / bool_or aggregates vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _t() -> pa.Table:
    return pa.table(
        {
            "g": ["a", "a", "b", "b", "c", "a", "b"],
            # group a: T,T,T → and=T; b: F,T,(null) → and=F,or=T; c: F → and/or=F
            "flag": [True, True, False, True, False, True, None],
        }
    )


def test_bool_and_or_grouped_matches_duckdb(duck):
    from conftest import assert_same

    t = _t()
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(ba=col("flag").bool_and(), bo=col("flag").bool_or())
        .collect()
    )
    assert_same(out, duck.sql("SELECT g, bool_and(flag) ba, bool_or(flag) bo FROM t GROUP BY g"))


def test_bool_and_or_global_matches_duckdb(duck):
    from conftest import assert_same

    t = _t()
    duck.register("t", t)
    out = bt.from_arrow(t).agg(ba=col("flag").bool_and(), bo=col("flag").bool_or()).collect()
    assert_same(out, duck.sql("SELECT bool_and(flag) ba, bool_or(flag) bo FROM t"))


def test_mode_grouped_matches_duckdb(duck):
    from conftest import assert_same

    # Unique mode per group (no frequency ties) so the tiebreak rule is irrelevant
    # and the result matches DuckDB's mode() exactly.
    t = pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "b", "b", "c"],
            "v": [5, 5, 7, 9, 9, 9, 1, 4],  # a→5, b→9, c→4
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(m=col("v").mode()).collect()
    assert_same(out, duck.sql("SELECT g, mode(v) m FROM t GROUP BY g"))


def test_mode_grouped_spilled_matches_duckdb(duck):
    # The bounded out-of-core mode (forced via a tight memory cap) must match the
    # in-memory result. A hot key with many repeats has a clear unique mode (no
    # frequency tie), so the result is unambiguous vs DuckDB even under spill.
    import numpy as np
    import pyarrow.compute as pc

    from batcher.config import Config, MemoryConfig, config_context
    from conftest import assert_same

    rng = np.random.default_rng(11)
    n = 20000
    k = np.concatenate([np.zeros(15000, "int64"), rng.integers(1, 50, n - 15000).astype("int64")])
    # Value 7 dominates the hot key 0 (60% of its rows); cold keys get random values.
    v = np.where((k == 0) & (rng.random(n) < 0.6), 7, rng.integers(100, 100000, n)).astype("int64")
    tbl = pa.table({"k": k, "v": v})
    duck.register("m", tbl)
    cap = Config().replace(memory=MemoryConfig(max_memory_bytes=1 << 14))
    with config_context(cap):
        out = bt.from_arrow(tbl).group_by("k").agg(mo=col("v").mode()).collect()
    # Compare only the hot key (its mode is unambiguous); cold keys may tie.
    out_hot = out.filter(pc.equal(out["k"], 0))
    assert_same(out_hot, duck.sql("SELECT k, mode(v) mo FROM m WHERE k = 0 GROUP BY k"))


def test_arg_min_max_grouped_matches_duckdb(duck):
    from conftest import assert_same

    # Unique keys per group → arg_min/arg_max are unambiguous and match DuckDB's
    # arg_min/arg_max(value, key).
    t = pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "c"],
            "val": [10, 20, 30, 40, 50, 60],
            "key": [1, 3, 2, 5, 4, 7],
        }
    )
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(hi=col("val").arg_max(by=col("key")), lo=col("val").arg_min(by=col("key")))
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT g, arg_max(val, key) hi, arg_min(val, key) lo FROM t GROUP BY g"),
    )
