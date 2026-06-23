"""`Dataset.sample` — fraction, reproducibility, and partition-independence.

Sampling is random, so it has no DuckDB differential oracle; instead we assert its
contract: the kept fraction is honored, the same seed reproduces the same rows, the
result is a subset of the input, and — the load-bearing invariant — the sampled set
is identical single-node and distributed (deterministic content-hash sampling).
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration


def _ds(n=2000):
    return bt.from_pydict({"id": list(range(n)), "g": [i % 7 for i in range(n)]})


def test_fraction_is_approximately_honored():
    n = 2000
    kept = _ds(n).sample(0.25, seed=1).count()
    assert 0.20 * n <= kept <= 0.30 * n  # ~25% within a generous band


def test_reproducible_same_seed():
    a = _ds().sample(0.3, seed=42).to_pydict()["id"]
    b = _ds().sample(0.3, seed=42).to_pydict()["id"]
    assert a == b


def test_boundaries():
    assert _ds(100).sample(1.0).count() == 100
    assert _ds(100).sample(0.0).count() == 0


def test_result_is_subset():
    n = 500
    kept = set(_ds(n).sample(0.4, seed=3).to_pydict()["id"])
    assert kept <= set(range(n))


def test_invalid_fraction_raises():
    with pytest.raises(PlanError, match="fraction"):
        _ds(10).sample(1.5)


def test_sample_then_filter_composes():
    out = _ds(500).sample(0.5, seed=9).filter(bt.col("g") == 0).to_pydict()
    assert all(g == 0 for g in out["g"])


def test_distributed_equals_single_node():
    pytest.importorskip("ray")
    ds = _ds(3000)
    single = sorted(ds.sample(0.3, seed=11).to_pydict()["id"])
    dist = sorted(
        ds.sample(0.3, seed=11).collect(distributed=True, num_workers=3).column("id").to_pylist()
    )
    assert single == dist  # deterministic, partition-independent sampling


def test_count_sample_keeps_exactly_n():
    assert _ds(2000).sample(n=50, seed=1).count() == 50
    assert _ds(30).sample(n=100).count() == 30  # n > input keeps all


def test_count_sample_reproducible_and_subset():
    a = _ds(1000).sample(n=40, seed=9).to_pydict()["id"]
    b = _ds(1000).sample(n=40, seed=9).to_pydict()["id"]
    assert a == b
    assert set(a) <= set(range(1000)) and len(a) == 40


def test_count_sample_distributed_equals_single_node():
    pytest.importorskip("ray")
    t = _ds(4000)
    single = set(t.sample(n=60, seed=11).to_pydict()["id"])
    distrib = set(
        t.sample(n=60, seed=11).collect(distributed=True, num_workers=4).column("id").to_pylist()
    )
    assert single == distrib and len(single) == 60


def test_sample_requires_exactly_one_of_fraction_n():
    with pytest.raises(PlanError, match="exactly one"):
        _ds(10).sample(0.5, n=5)
    with pytest.raises(PlanError, match="exactly one"):
        _ds(10).sample()


def test_sql_tablesample():
    import pyarrow as pa

    t = pa.table({"v": list(range(1000))})
    # RESERVOIR(n ROWS) → fixed count; BERNOULLI(p PERCENT) → fraction.
    assert bt.sql("SELECT * FROM t TABLESAMPLE RESERVOIR(50 ROWS)", t=t).count() == 50
    pct = bt.sql("SELECT * FROM t TABLESAMPLE BERNOULLI(20 PERCENT)", t=t).count()
    assert 100 <= pct <= 300  # ~20% within a generous band
