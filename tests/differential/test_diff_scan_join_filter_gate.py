"""The engine's scan-level runtime join filter, under its per-join cost gate, vs DuckDB.

The streaming executor digests a reducible join's build-side key set and masks the probe side's
*scan* with it (`crates/bc-interp/src/stream/runtime_filter.rs`). It used to engage only when the
query's largest input held 16M rows, so no test fixture ever reached it without forcing it on.
It is now gated per join: a probe side of at least four morsels that is at least four times its
build side. The fixtures here are sized to clear that gate on the default setting, so these
tests exercise the shipped path, not only the forced one.

Every join type runs against DuckDB over keys with nulls on both sides, duplicate build keys,
probe keys with no match, a composite key and a two-dimension star whose two filters land on
the same scan. `Left`/`Full`/`Right`/`Anti` must never be reduced on their preserved side, so
they are here as much as `Inner`/`Semi` are: a filter that leaked onto one of them would delete
answers. Each query runs with the filter off, on the default gate, and forced, and all three
must equal the oracle.

A positive control proves the default setting actually places a filter on these fixtures, and
two negative controls prove the gate and the kill switch decline — without them, every
comparison above could be passing on a path that never filters.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered
from batcher import col
from batcher.core import executor

pytestmark = pytest.mark.differential

#: Probe rows: well past four morsels and past the 65,536 rows below which nothing shards.
_N = 300_000
#: Distinct fact keys. The dimensions hold a seventh / an eleventh of them.
_KEYS = 60_000


def _fact() -> pa.Table:
    rng = np.random.default_rng(7)
    k = rng.integers(0, _KEYS, _N)
    null = rng.random(_N) < 0.02
    return pa.table(
        {
            "k": pa.array(k, pa.int64(), mask=null),
            "k2": pa.array(rng.integers(0, 50, _N), pa.int64()),
            "v": pa.array(rng.integers(0, 1_000, _N), pa.int64()),
            "s": pa.array([f"s{i % 13}" for i in range(_N)]),
        }
    )


def _dim() -> pa.Table:
    keys = list(range(0, _KEYS, 7))
    keys += keys[:500]  # duplicate build keys: the join must fan out, the filter must not
    keys.append(None)  # a null build key matches nothing and must not widen the key set
    keys.append(_KEYS + 10)  # a build key no probe row carries
    return pa.table(
        {
            "dk": pa.array(keys, pa.int64()),
            "dk2": pa.array([i % 50 for i in range(len(keys))], pa.int64()),
            "region": pa.array([f"r{i % 5}" for i in range(len(keys))]),
        }
    )


def _dim3() -> pa.Table:
    keys = list(range(0, 50, 11))
    return pa.table({"ek": pa.array(keys, pa.int64()), "tag": [f"t{i}" for i in keys]})


_QUERIES = {
    "inner": "SELECT region, count(*) AS n, sum(v) AS sv FROM fact JOIN dim ON k = dk "
    "WHERE v < 900 GROUP BY region",
    "semi": "SELECT count(*) AS n, sum(v) AS sv FROM fact WHERE k IN (SELECT dk FROM dim)",
    "anti": "SELECT count(*) AS n FROM fact WHERE NOT EXISTS (SELECT 1 FROM dim WHERE dk = k)",
    "left": "SELECT count(*) AS n, count(dk) AS m, sum(v) AS sv FROM fact LEFT JOIN dim ON k = dk",
    "full": "SELECT count(*) AS n, count(dk) AS m, count(k) AS f FROM fact FULL JOIN dim ON k = dk",
    "right": "SELECT count(*) AS n, count(k) AS f FROM fact RIGHT JOIN dim ON k = dk",
    "composite": "SELECT count(*) AS n, sum(v) AS sv FROM fact JOIN dim ON k = dk AND k2 = dk2",
    "star": "SELECT tag, region, count(*) AS n FROM fact JOIN dim ON k = dk "
    "JOIN dim3 ON k2 = ek GROUP BY tag, region",
    "rows": "SELECT k, v, region FROM fact JOIN dim ON k = dk WHERE v < 10",
}

_ORDERED = (
    "SELECT k, v, region FROM fact JOIN dim ON k = dk WHERE v < 500 "
    "ORDER BY v DESC, k, region LIMIT 25"
)


@pytest.fixture(scope="module")
def tables() -> dict[str, pa.Table]:
    return {"fact": _fact(), "dim": _dim(), "dim3": _dim3()}


def _session(tables: dict[str, pa.Table]) -> bt.Session:
    s = bt.Session()
    for name, t in tables.items():
        s.register(name, t)
    return s


@pytest.mark.parametrize("setting", ["0", "default", "force"])
@pytest.mark.parametrize("name", sorted(_QUERIES))
def test_every_join_type_matches_duckdb_with_the_filter_off_gated_and_forced(
    duck, tables, monkeypatch, name, setting
) -> None:
    if setting == "default":
        monkeypatch.delenv("BATCHER_RUNTIME_JOIN_FILTER", raising=False)
    else:
        monkeypatch.setenv("BATCHER_RUNTIME_JOIN_FILTER", setting)
    for t, v in tables.items():
        duck.register(t, v)
    sql = _QUERIES[name]
    assert_same(_session(tables).sql(sql).collect(), duck.sql(sql))


@pytest.mark.parametrize("setting", ["0", "default", "force"])
def test_an_ordered_limit_over_a_filtered_join_keeps_the_oracle_order(
    duck, tables, monkeypatch, setting
) -> None:
    if setting == "default":
        monkeypatch.delenv("BATCHER_RUNTIME_JOIN_FILTER", raising=False)
    else:
        monkeypatch.setenv("BATCHER_RUNTIME_JOIN_FILTER", setting)
    for t, v in tables.items():
        duck.register(t, v)
    assert_same_ordered(_session(tables).sql(_ORDERED).collect(), duck.sql(_ORDERED))


def _listed(monkeypatch, fact: bt.Dataset, dim: bt.Dataset) -> tuple[int, set[str]]:
    """Run a filtered star join and return its count and the operators the engine marked."""
    captured: list[list[dict]] = []
    real = executor._record_op_feedback

    def spy(sink, ops, batch_size, planned=()):
        captured.append([dict(op) for op in ops])
        return real(sink, ops, batch_size, planned)

    monkeypatch.setattr(executor, "_record_op_feedback", spy)
    got = fact.filter(col("v") < 900).join(dim, on="k").agg(n=col("v").count()).collect()
    listed = (
        {op["kind"] for op in captured[-1] if op.get("runtime_filtered")} if captured else set()
    )
    return got.to_pydict()["n"][0], listed


def _star(build_keys: list[int]) -> tuple[bt.Dataset, bt.Dataset, int]:
    fact = bt.from_pydict(
        {"k": [i % _KEYS for i in range(_N)], "v": [i % 1_000 for i in range(_N)]}
    )
    dim = bt.from_pydict({"k": build_keys, "w": [1] * len(build_keys)})
    keys = set(build_keys)
    want = sum(1 for i in range(_N) if i % 1_000 < 900 and i % _KEYS in keys)
    return fact, dim, want


def test_the_default_gate_places_a_filter_on_a_lopsided_join(monkeypatch) -> None:
    """Positive control: without it, every comparison above could be passing unfiltered."""
    monkeypatch.delenv("BATCHER_RUNTIME_JOIN_FILTER", raising=False)
    fact, dim, want = _star(list(range(0, _KEYS, 7)))
    n, listed = _listed(monkeypatch, fact, dim)
    assert n == want
    assert "filter" in listed, f"the default gate placed no filter on a 35:1 join: {listed}"


def test_the_gate_declines_a_build_side_as_large_as_a_quarter_of_the_probe(monkeypatch) -> None:
    monkeypatch.delenv("BATCHER_RUNTIME_JOIN_FILTER", raising=False)
    fact, dim, want = _star(list(range(0, _N // 3)))
    n, listed = _listed(monkeypatch, fact, dim)
    assert n == want
    assert not listed, f"a 3:1 join was filtered: {listed}"


def test_the_kill_switch_places_no_filter(monkeypatch) -> None:
    monkeypatch.setenv("BATCHER_RUNTIME_JOIN_FILTER", "0")
    fact, dim, want = _star(list(range(0, _KEYS, 7)))
    n, listed = _listed(monkeypatch, fact, dim)
    assert n == want
    assert not listed, f"BATCHER_RUNTIME_JOIN_FILTER=0 still filtered: {listed}"
