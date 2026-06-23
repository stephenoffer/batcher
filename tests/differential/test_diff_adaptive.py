"""Adaptive (intra-query) execution: stage-boundary re-optimization with exact stats."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, core, count
from batcher.api.adaptive import execute_adaptive


@pytest.fixture
def tables(duck):
    rng = np.random.default_rng(0)
    fact = pa.table(
        {
            "k": rng.integers(0, 100, 50000).astype("int64"),
            "region": rng.integers(0, 5, 50000).astype("int64"),
            "v": rng.integers(0, 100, 50000).astype("int64"),
        }
    )
    dim = pa.table({"k": np.arange(100, dtype="int64"), "name": [f"n{i}" for i in range(100)]})
    duck.register("f", fact)
    duck.register("d", dim)
    return fact, dim


def _multistage(fact, dim):
    agg = (
        bt.from_arrow(fact).filter(col("region") > 2).group_by("k").agg(s=col("v").sum(), n=count())
    )
    return agg.join(bt.from_arrow(dim), on="k").select("k", "name", "s", "n")


def _rows(t):
    return sorted(tuple(r.values()) for r in t.to_pylist())


def test_adaptive_matches_normal_and_duckdb(duck, tables):
    from conftest import assert_same

    fact, dim = tables
    q = _multistage(fact, dim)
    assert _rows(q.collect(adaptive=True)) == _rows(q.collect())
    assert_same(
        q.collect(adaptive=True),
        duck.sql(
            "SELECT a.k, d.name, a.s, a.n FROM "
            "(SELECT k, SUM(v) s, COUNT(*) n FROM f WHERE region > 2 GROUP BY k) a "
            "JOIN d ON a.k = d.k"
        ),
    )


def test_adaptive_uses_exact_cardinalities(tables):
    fact, dim = tables
    q = _multistage(fact, dim)
    res = execute_adaptive(q._plan, q._sources, core.default_hub())
    # filter+groupby stage, the join stage, then the final projection → multiple stages.
    assert res.stages >= 2
    # The join's build-side decision is made from materialized (exact) sizes.
    assert any(d.provenance == "exact" for d in res.decisions)


def test_adaptive_join_over_two_aggregates(duck):
    from conftest import assert_same

    rng = np.random.default_rng(1)
    a = pa.table(
        {
            "k": rng.integers(0, 50, 20000).astype("int64"),
            "x": rng.integers(0, 10, 20000).astype("int64"),
        }
    )
    b = pa.table(
        {
            "k": rng.integers(0, 50, 30000).astype("int64"),
            "y": rng.integers(0, 10, 30000).astype("int64"),
        }
    )
    duck.register("a", a)
    duck.register("b", b)
    qa = bt.from_arrow(a).group_by("k").agg(sx=col("x").sum())
    qb = bt.from_arrow(b).group_by("k").agg(sy=col("y").sum())
    q = qa.join(qb, on="k").select("k", "sx", "sy")
    assert _rows(q.collect(adaptive=True)) == _rows(q.collect())
    assert_same(
        q.collect(adaptive=True),
        duck.sql(
            "SELECT a.k, a.sx, b.sy FROM "
            "(SELECT k, SUM(x) sx FROM a GROUP BY k) a JOIN "
            "(SELECT k, SUM(y) sy FROM b GROUP BY k) b ON a.k = b.k"
        ),
    )
