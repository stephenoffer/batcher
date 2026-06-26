"""Coverage for the Arrow MAP type accessors (`map.keys`/`values`/`get`).

A Map column passes through the engine zero-copy (Arrow C Data Interface); these
test the `.map` accessor functions. `element_at` is cross-checked against DuckDB;
the list-valued `map_keys`/`map_values` are pinned to fixtures (the harness can't
sort list columns) — DuckDB agrees on the same inputs.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _data():
    m = pa.array(
        [[("a", 1), ("b", 2)], [("c", 3)], [], None],
        type=pa.map_(pa.string(), pa.int64()),
    )
    return pa.table({"id": [1, 2, 3, 4], "m": m})


def test_map_keys_values_fixtures():
    out = (
        bt.from_arrow(_data())
        .select(k=col("m").map.keys(), v=col("m").map.values())
        .collect()
        .to_pydict()
    )
    assert out["k"] == [["a", "b"], ["c"], [], None]
    assert out["v"] == [[1, 2], [3], [], None]


def test_element_at_present_and_absent():
    out = (
        bt.from_arrow(_data())
        .select(a=col("m").map.get("a"), c=col("m").map.get("c"))
        .collect()
        .to_pydict()
    )
    assert out["a"] == [1, None, None, None]
    assert out["c"] == [None, 3, None, None]


def test_map_passthrough_and_filter():
    # A map column survives scan, project, and filter unchanged (zero-copy).
    ds = bt.from_arrow(_data())
    out = ds.filter(col("id") <= 2).select("m").collect().column("m").to_pylist()
    assert out == [[("a", 1), ("b", 2)], [("c", 3)]]


def test_histogram_matches_duckdb(duck):
    data = {"g": ["a", "a", "a", "b", "b", "c"], "v": [1, 1, 2, 3, 3, None]}
    ds = bt.from_pydict(data)
    duck.register("t", ds.collect())
    out = ds.group_by("g").agg(h=col("v").histogram()).collect().to_pydict()
    exp = duck.sql("SELECT g, histogram(v) h FROM t GROUP BY g").to_arrow_table().to_pydict()
    # Map column → compare as {key: dict(entries)}; an all-null group is a NULL map.
    om = {g: (dict(h) if h is not None else None) for g, h in zip(out["g"], out["h"], strict=True)}
    em = {g: (dict(h) if h is not None else None) for g, h in zip(exp["g"], exp["h"], strict=True)}
    assert om == em


def test_histogram_spilled_matches_duckdb(duck):
    # The bounded out-of-core histogram (forced via a tight memory cap) must match
    # DuckDB. A hot key with many repeats + ~10% nulls + an all-null group exercises
    # the streaming map-build, null exclusion, and the NULL-map case under spill.
    import numpy as np

    from batcher.config import Config, MemoryConfig, config_context

    rng = np.random.default_rng(13)
    n = 20000
    g = np.concatenate([np.zeros(15000, "int64"), rng.integers(1, 50, n - 15000).astype("int64")])
    v = pa.array(rng.integers(0, 200, n).astype("int64"), mask=rng.random(n) < 0.1)
    g = np.concatenate([g, np.full(50, 99, "int64")])  # all-null group → NULL map
    v = pa.concat_arrays([v, pa.array(np.zeros(50, "int64"), mask=np.ones(50, bool))])
    tbl = pa.table({"g": g, "v": v})
    duck.register("h", tbl)
    cap = Config().replace(memory=MemoryConfig(max_memory_bytes=1 << 14))
    with config_context(cap):
        out = bt.from_arrow(tbl).group_by("g").agg(h=col("v").histogram()).collect().to_pydict()
    exp = duck.sql("SELECT g, histogram(v) h FROM h GROUP BY g").to_arrow_table().to_pydict()
    om = {g: (dict(h) if h is not None else None) for g, h in zip(out["g"], out["h"], strict=True)}
    em = {g: (dict(h) if h is not None else None) for g, h in zip(exp["g"], exp["h"], strict=True)}
    assert om == em


def test_histogram_single_node_equals_distributed():
    data = {"g": ["a", "a", "b", "b", "b", "a"], "v": [1, 2, 2, 2, 3, 1]}
    ds = bt.from_pydict(data).group_by("g").agg(h=col("v").histogram())
    single = {
        g: dict(h) for g, h in zip(*[ds.collect().to_pydict()[k] for k in ("g", "h")], strict=True)
    }
    dd = ds.collect(distributed=True, num_workers=2).to_pydict()
    multi = {g: dict(h) for g, h in zip(dd["g"], dd["h"], strict=True)}
    assert single == multi
