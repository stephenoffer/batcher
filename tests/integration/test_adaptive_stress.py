"""Adversarial stress matrix for the adaptive config (worst-case data shapes).

The promise of zero-config auto-tuning is that a user who configures nothing gets the
*same result* as any explicit configuration, on any data — including the shapes that
break naive defaults (empty input, a single giant row, a hot-key skew, all-null/
all-same/all-distinct columns, NaN/Inf, extreme integers, many tiny batches, high and
low cardinality, wide rows). This suite asserts that across a matrix of shapes x
operations:

* the **adaptive default** (`collect()`, all auto) == a **forced baseline** (unbounded
  memory, fixed morsel, `adaptive=False`, fusion off) — the result is invariant to the
  adaptive machinery; and
* a **forced tiny budget** (1 MiB → the Rust pool spills, morsels shrink, breakers go
  out-of-core) == the same baseline — spilling under the worst shapes stays correct.

Any mismatch or crash is a real adaptive-config bug (this is how the Phase-1 global-agg
and empty-input spill bugs were caught). The baseline is the ground-truth oracle, so no
external engine is needed; a few cases also cross-check DuckDB.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.config import Config, config_context

pytestmark = pytest.mark.integration

_RNG = np.random.default_rng(0)


# --- configs ----------------------------------------------------------------------


def _baseline_config() -> Config:
    """Defeat every adaptive lever — the ground-truth execution path."""
    base = Config()
    return base.replace(
        memory=dataclasses.replace(base.memory, unbounded_memory=True),
        execution=dataclasses.replace(
            base.execution, adaptive_morsel_sizing=False, fuse_linear=False
        ),
    )


def _tiny_budget_config() -> Config:
    """Force out-of-core: a 1 MiB cap makes the Rust pool spill and morsels shrink."""
    base = Config()
    return base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=1 << 20))


# --- adversarial datasets ----------------------------------------------------------
# Every standard dataset has columns: k (int64 group key), v (float64 value), s (string).
# Special shapes (empty / giant-row / wide) are separate factories.


def _table(k, v, s) -> pa.Table:
    return pa.table(
        {
            "k": pa.array(k, type=pa.int64()),
            "v": pa.array(v, type=pa.float64()),
            "s": pa.array(s, type=pa.large_string()),
        }
    )


def _standard_datasets() -> dict[str, pa.Table]:
    out: dict[str, pa.Table] = {}
    n = 20_000

    # empty
    out["empty"] = _table([], [], [])
    # single row
    out["single_row"] = _table([7], [3.5], ["a"])
    # all-null value + all-null key
    out["all_null_v"] = _table(list(range(n)), [None] * n, ["x"] * n)
    out["all_null_k"] = _table([None] * n, _RNG.random(n).tolist(), ["x"] * n)
    # cardinality-1 key (one giant group)
    out["card1_key"] = _table([5] * n, _RNG.random(n).tolist(), ["g"] * n)
    # all-distinct key (n groups)
    out["all_distinct_key"] = _table(
        list(range(n)), _RNG.random(n).tolist(), [f"s{i}" for i in range(n)]
    )
    # hot-key skew: one key holds 95%
    skew_k = [0] * int(n * 0.95) + _RNG.integers(1, 500, n - int(n * 0.95)).tolist()
    _RNG.shuffle(skew_k)
    out["hot_key_skew"] = _table(skew_k, _RNG.random(n).tolist(), ["h"] * n)
    # NaN / Inf floats
    vals = _RNG.random(n)
    vals[::101] = math.nan
    vals[::197] = math.inf
    vals[::293] = -math.inf
    out["nan_inf_v"] = _table(_RNG.integers(0, 50, n).tolist(), vals.tolist(), ["n"] * n)
    # extreme integers (int64 min/max in the key)
    ek = _RNG.integers(0, 10, n).tolist()
    ek[0], ek[1], ek[2] = (1 << 63) - 1, -(1 << 63), 0
    out["extreme_int_key"] = _table(ek, _RNG.random(n).tolist(), ["e"] * n)
    # nulls scattered through keys and values
    sk = _RNG.integers(0, 100, n).astype(object)
    sk[::7] = None
    sv = _RNG.random(n)
    sv2 = sv.astype(object)
    sv2[::11] = None
    out["scattered_nulls"] = _table(sk.tolist(), sv2.tolist(), ["m"] * n)
    # long + empty + unicode strings
    longs = [
        ("" if i % 3 == 0 else ("ü" * (i % 50)) if i % 3 == 1 else "z" * (i % 2000))
        for i in range(n)
    ]
    out["varied_strings"] = _table(_RNG.integers(0, 30, n).tolist(), _RNG.random(n).tolist(), longs)
    # high-cardinality (n/2 distinct keys)
    out["high_card_key"] = _table(
        _RNG.integers(0, n // 2, n).tolist(), _RNG.random(n).tolist(), ["c"] * n
    )
    # low-cardinality (3 keys) over many rows
    out["low_card_key"] = _table(
        _RNG.integers(0, 3, n).tolist(), _RNG.random(n).tolist(), ["l"] * n
    )
    return out


def _from_many_tiny_batches(table: pa.Table) -> bt.Dataset:
    """A dataset delivered as many 1-row batches (stresses coalescing/rebatching)."""
    batches = table.to_batches(max_chunksize=1)
    schema = table.schema
    return bt.from_batches((lambda: iter(batches)), schema)


# --- operations (Dataset -> result dict), reusable across shapes -------------------


def _ops() -> dict[str, callable]:
    return {
        "filter_half": lambda ds: ds.filter(col("v") >= 0.5),
        "filter_none": lambda ds: ds.filter(col("v") > 1e9),
        "filter_all": lambda ds: ds.filter(col("v") <= 1e9),
        "with_col": lambda ds: ds.with_columns(w=col("v") * 2 + 1),
        "groupby_sum": lambda ds: ds.group_by("k").agg(s=col("v").sum(), n=count()),
        "groupby_stats": lambda ds: ds.group_by("k").agg(
            mn=col("v").min(), mx=col("v").max(), me=col("v").mean()
        ),
        "global_sum": lambda ds: ds.group_by().agg(s=col("v").sum(), n=count()),
        "global_median": lambda ds: ds.group_by().agg(m=col("v").median()),
        "groupby_median": lambda ds: ds.group_by("k").agg(m=col("v").median()),
        "groupby_nunique": lambda ds: ds.group_by("k").agg(d=col("s").n_unique()),
        "distinct_k": lambda ds: ds.select("k").distinct(),
        "sort_v": lambda ds: ds.sort("v"),
        "topn": lambda ds: ds.sort("v", descending=True).limit(17),
        "self_join": lambda ds: (
            ds.group_by("k").agg(s=col("v").sum()).join(ds.group_by("k").agg(n=count()), on="k")
        ),
        "window_rank": lambda ds: ds.window(
            partition_by=["k"], order_by=[("v", False)], functions={"r": "rank"}
        ),
        "global_window_sum": lambda ds: ds.window(
            order_by=[("v", False)], functions={"rs": ("sum", "v")}
        ),
        "union_all": lambda ds: ds.union(ds),
        "union_distinct": lambda ds: ds.select("k").union(ds.select("k"), distinct=True),
        "map_batches_auto": lambda ds: ds.ml.map_batches(_AddOne),
    }


class _AddOne:
    """A load-once class fn → triggers the auto-batch-sizing path."""

    def __call__(self, b: pa.RecordBatch) -> pa.RecordBatch:
        return b.append_column("w", pa.array([(x.as_py() or 0) + 1 for x in b.column("v")]))


def _scrub(x: object) -> object:
    """Canonicalize a cell for multiset comparison: tag NaN, and round floats to 9
    significant figures so float **non-associativity** (a sum/mean reordered across a
    different batching) is not mistaken for a correctness difference — exactly the
    tolerance the DuckDB differential oracle applies. ±Inf is preserved as-is."""
    if isinstance(x, float):
        if math.isnan(x):
            return "__nan__"
        if math.isinf(x) or x == 0.0:
            return x
        return round(x, 8 - math.floor(math.log10(abs(x))))
    return x


def _norm(table: pa.Table) -> list:
    """Order-independent, NaN-safe, float-tolerant multiset of rows."""
    rows = [tuple(_scrub(v) for v in r.values()) for r in table.to_pylist()]
    return sorted(rows, key=repr)


def _collect(ds: bt.Dataset, cfg: Config) -> pa.Table:
    with config_context(cfg):
        return ds.collect()


# --- the matrix --------------------------------------------------------------------

_DATASETS = _standard_datasets()
_OPS = _ops()


@pytest.mark.parametrize("shape", sorted(_DATASETS))
@pytest.mark.parametrize("op", sorted(_OPS))
def test_adaptive_matches_baseline(shape, op):
    """Adaptive default == forced baseline (result-invariant to the adaptive machinery)."""
    table = _DATASETS[shape]
    query = _OPS[op]
    baseline = _norm(_collect(query(bt.from_arrow(table)), _baseline_config()))
    adaptive = _norm(_collect(query(bt.from_arrow(table)), Config()))  # all auto
    assert adaptive == baseline


@pytest.mark.parametrize("shape", sorted(_DATASETS))
@pytest.mark.parametrize("op", sorted(_OPS))
def test_tiny_budget_spill_matches_baseline(shape, op):
    """A 1 MiB budget (forced spill / shrunk morsels / out-of-core breakers) stays correct."""
    table = _DATASETS[shape]
    query = _OPS[op]
    baseline = _norm(_collect(query(bt.from_arrow(table)), _baseline_config()))
    spilled = _norm(_collect(query(bt.from_arrow(table)), _tiny_budget_config()))
    assert spilled == baseline


@pytest.mark.parametrize("op", sorted(_OPS))
def test_many_tiny_batches_matches_baseline(op):
    """A stream of 1-row batches (worst coalescing case) under the auto budget == baseline."""
    table = _DATASETS["scattered_nulls"]
    query = _OPS[op]
    baseline = _norm(_collect(query(bt.from_arrow(table)), _baseline_config()))
    auto = _norm(_collect(query(_from_many_tiny_batches(table)), Config()))
    assert auto == baseline


# --- special shapes that break the standard k/v/s schema ---------------------------


def _giant_row_table() -> pa.Table:
    # A single row whose string column is 4 MiB — far bigger than one morsel (1 MiB), so
    # byte-aware morsel splitting cannot shrink below it (a morsel is at least one row).
    return pa.table({"k": pa.array([1], pa.int64()), "s": pa.array(["x" * (4 << 20)])})


def _wide_table() -> pa.Table:
    # 600 columns — stresses per-row width estimates and the schema-handling paths.
    rng = np.random.default_rng(1)
    return pa.table({f"c{i}": rng.integers(0, 100, 200).tolist() for i in range(600)})


def _nested_table() -> pa.Table:
    # Deeply nested list<struct> + map-like columns the engine must thread through.
    rng = np.random.default_rng(2)
    n = 5000
    return pa.table(
        {
            "k": pa.array((rng.integers(0, 20, n)).tolist(), pa.int64()),
            "lst": pa.array([[i % 5, (i + 1) % 5, (i + 2) % 5] for i in range(n)]),
            "st": pa.array([{"a": i % 7, "b": f"v{i % 3}"} for i in range(n)]),
        }
    )


_SPECIAL = {
    "giant_row": (
        _giant_row_table(),
        {
            "filter": lambda ds: ds.filter(col("k") >= 0),
            "select": lambda ds: ds.select("k"),
            "distinct": lambda ds: ds.select("k").distinct(),
            "map_batches": lambda ds: ds.ml.map_batches(_PassThrough),
        },
    ),
    "wide_600col": (
        _wide_table(),
        {
            "filter": lambda ds: ds.filter(col("c0") >= 50),
            "groupby": lambda ds: ds.group_by("c0").agg(s=col("c1").sum()),
            "sort": lambda ds: ds.sort("c0"),
            "distinct": lambda ds: ds.select("c0").distinct(),
        },
    ),
    "nested_types": (
        _nested_table(),
        {
            "filter": lambda ds: ds.filter(col("k") >= 5),
            "groupby_count": lambda ds: ds.group_by("k").agg(n=count()),
            "distinct_k": lambda ds: ds.select("k").distinct(),
            "sort_k": lambda ds: ds.sort("k"),
        },
    ),
}


class _PassThrough:
    def __call__(self, b: pa.RecordBatch) -> pa.RecordBatch:
        return b


def test_auto_sensed_small_budget_still_correct(monkeypatch):
    """The *auto* path (no explicit cap) on a constrained container: when the sensor
    reports a small envelope, `resolve_auto_config` must derive a positive spill budget
    and the query must complete correctly out-of-core — the real zero-config scenario
    (vs the explicit-tiny-budget proxy above)."""
    from batcher.carbonite.memory import pressure

    monkeypatch.setattr(pressure.PressureMonitor, "envelope_bytes", lambda self: 3 << 20)
    table = _DATASETS["hot_key_skew"]
    query = _OPS["groupby_sum"]
    baseline = _norm(_collect(query(bt.from_arrow(table)), _baseline_config()))
    # Default Config → resolve_auto_config senses the (mocked-small) envelope and spills.
    auto = _norm(query(bt.from_arrow(table)).collect())
    assert auto == baseline


# --- DuckDB correctness cross-check (catches a bug present in BOTH Batcher configs) ----
# Invariance (adaptive == baseline) can't catch a result wrong in both; the gold oracle
# can. Restricted to shapes/ops with matching DuckDB↔Arrow null/number semantics (NaN/Inf
# aggregation semantics differ across engines, so those shapes are excluded here).

_DUCK_SHAPES = [
    "hot_key_skew",
    "high_card_key",
    "low_card_key",
    "all_distinct_key",
    "scattered_nulls",
    "extreme_int_key",
    "card1_key",
    "all_null_v",
]
_DUCK_SQL = {
    "groupby_sum": "SELECT k, sum(v) AS s, count(*) AS n FROM t GROUP BY k",
    "global_sum": "SELECT sum(v) AS s, count(*) AS n FROM t",
    "distinct_k": "SELECT DISTINCT k FROM t",
    "filter_half": "SELECT * FROM t WHERE v >= 0.5",
    "groupby_median": "SELECT k, median(v) AS m FROM t GROUP BY k",
}


@pytest.mark.parametrize("shape", _DUCK_SHAPES)
@pytest.mark.parametrize("op", sorted(_DUCK_SQL))
def test_adaptive_matches_duckdb_on_adversarial_shapes(shape, op):
    """The adaptive default produces the *correct* result (vs DuckDB) on worst-case data —
    not merely the same as the baseline (which could share a bug)."""
    duckdb = pytest.importorskip("duckdb")
    table = _DATASETS[shape]
    got = _norm(_OPS[op](bt.from_arrow(table)).collect())  # adaptive default
    con = duckdb.connect()
    con.register("t", table)
    want = _norm(con.execute(_DUCK_SQL[op]).to_arrow_table())
    assert got == want


def test_concurrent_queries_share_pool_correctly():
    """Several queries running at once under the auto budget share the process-wide
    pool (cross-query admission + cooperative spill) and each still returns the right
    result — the shared-envelope worst case."""
    from concurrent.futures import ThreadPoolExecutor

    shapes = ["hot_key_skew", "high_card_key", "low_card_key", "all_distinct_key"]
    query = _OPS["groupby_sum"]
    expected = {
        s: _norm(_collect(query(bt.from_arrow(_DATASETS[s])), _baseline_config())) for s in shapes
    }

    def run(s):
        return s, _norm(query(bt.from_arrow(_DATASETS[s])).collect())

    with ThreadPoolExecutor(max_workers=len(shapes)) as pool:
        results = dict(pool.map(run, shapes * 3))  # 12 concurrent queries
    for s in shapes:
        assert results[s] == expected[s]


@pytest.mark.parametrize("shape", sorted(_SPECIAL))
def test_special_shapes_adaptive_and_spill_match_baseline(shape):
    """Giant single row (> morsel), 600-column-wide, and nested-type tables: the adaptive
    default and a forced tiny budget both equal the baseline (no crash, no mismatch)."""
    table, ops = _SPECIAL[shape]
    for query in ops.values():
        baseline = _norm(_collect(query(bt.from_arrow(table)), _baseline_config()))
        adaptive = _norm(_collect(query(bt.from_arrow(table)), Config()))
        spilled = _norm(_collect(query(bt.from_arrow(table)), _tiny_budget_config()))
        assert adaptive == baseline
        assert spilled == baseline
