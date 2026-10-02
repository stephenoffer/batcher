"""A learned row count replaces a node's size, not its column statistics.

The estimator's learned-first branch used to return a bare `RelStats(rows)` for a node whose
size a past run measured, dropping every column statistic. Each join above such a node then
saw no distinct count on its keys and divided by the other side's alone; on JOB q29a a
1.15M-row subtree met a single-movie side (ndv 1) and was estimated at 1.04e12 rows, which
sent the query out of core. The companion fix is in the plan signature: an equality's literal
now names its own predicate, so `country_code = '[ru]'`'s measurement cannot answer for
`country_code = '[us]'`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.signature import plan_signature
from batcher.kyber.stats import StatsEstimator
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

pytestmark = pytest.mark.unit


def _joined():
    fact = bt.from_arrow(
        pa.table({"k": [i % 100 for i in range(10_000)], "v": list(range(10_000))})
    )
    dim = bt.from_arrow(pa.table({"k": list(range(100)), "w": list(range(100))}))
    return fact.join(dim, on="k")


_STATS = [
    SourceStatistics(
        row_count=10_000, columns={"k": ColumnStat(ndv=100.0, provenance=Provenance.EXACT)}
    ),
    SourceStatistics(
        row_count=100, columns={"k": ColumnStat(ndv=100.0, provenance=Provenance.EXACT)}
    ),
]


def test_a_learned_size_keeps_the_nodes_column_statistics():
    ds = _joined()
    cold = StatsEstimator(ds._sources, source_stats=_STATS).estimate(ds._plan)
    assert cold.column("k").ndv is not None, "control: the structural estimate knows k's ndv"
    learned = {plan_signature(ds._plan): {"rows": 40.0, "n_obs": 1}}
    warm = StatsEstimator(ds._sources, learned=learned, source_stats=_STATS).estimate(ds._plan)
    assert warm.rows == 40.0 and warm.provenance is Provenance.LEARNED
    ndv = warm.column("k").ndv
    assert ndv is not None, "the learned size dropped the key's distinct count"
    assert ndv <= 40.0, "a distinct count cannot exceed the measured row count"


def test_equalities_on_different_values_are_different_signatures():
    ds = bt.from_arrow(pa.table({"c": ["us", "ru", "us"]}))
    us = ds.filter(bt.col("c") == "us")._plan
    ru = ds.filter(bt.col("c") == "ru")._plan
    assert plan_signature(us) != plan_signature(ru)
    # Range bounds still normalize, so a moved bound keeps its learned selectivity.
    n = bt.from_arrow(pa.table({"x": [1, 2, 3]}))
    assert plan_signature(n.filter(bt.col("x") > 1)._plan) == plan_signature(
        n.filter(bt.col("x") > 2)._plan
    )
