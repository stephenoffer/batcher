"""Plan-shape tests for `pre_aggregate_beneath_dimension`.

The rule moves an aggregate's string group keys off the fact rows: `(F JOIN A) JOIN C`
grouped by `A`'s strings and `C`'s keys becomes `A JOIN Agg_partial(F JOIN C)`, and
`(F JOIN C) JOIN A` gets the partial beneath the top join as it stands. These pin when it
fires and when it must not; `tests/differential/test_diff_pre_aggregate_beneath_dimension.py`
proves the rewritten plan returns what DuckDB returns.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.config import active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.agg_pushdown import pre_aggregate_beneath_dimension
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Aggregate, Join
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit

_N = 400


def _facts():
    return bt.from_pydict(
        {
            "f_cust": [i % 10 for i in range(_N)],
            "f_date": [i % 20 for i in range(_N)],
            "f_v": [float(i) for i in range(_N)],
        }
    )


def _cust():
    return bt.from_pydict(
        {"c_sk": list(range(10)), "c_id": [f"C{i}" for i in range(10)], "c_n": [1] * 10}
    )


def _dates():
    return bt.from_pydict({"d_sk": list(range(20)), "d_year": [2000 + i // 10 for i in range(20)]})


def _ctx(ds, ndv):
    est = StatsEstimator(ds._sources, learned={"__column_ndv__": ndv})
    return OptimizerContext(config=active_config(), sources=ds._sources, hub=None, estimator=est)


_NDV = {"f_cust": 10.0, "d_year": 2.0, "f_date": 20.0, "c_sk": 10.0, "d_sk": 20.0}


def _string_side_first():
    """`(facts JOIN cust) JOIN dates`: TPC-DS q11's `year_total` as join order builds it."""
    return (
        _facts()
        .join(_cust(), left_on="f_cust", right_on="c_sk")
        .join(_dates(), left_on="f_date", right_on="d_sk")
        .group_by("c_id", "d_year")
        .agg(tot=col("f_v").sum(), n=bt.count(), lo=col("f_v").min())
    )


def _plain_side_first():
    """`(facts JOIN dates) JOIN cust`: the plain dimension already beside the facts."""
    return (
        _facts()
        .join(_dates(), left_on="f_date", right_on="d_sk")
        .join(_cust(), left_on="f_cust", right_on="c_sk")
        .group_by("c_id", "d_year")
        .agg(tot=col("f_v").sum(), n=bt.count())
    )


def _partial(plan):
    found = [
        n
        for n in walk(plan)
        if isinstance(n, Aggregate) and any(a.alias.startswith("__rp") for a in n.aggregates)
    ]
    return found[0] if found else None


def _scans(plan):
    from batcher.plan.logical import Scan

    return sorted(n.source_id for n in walk(plan) if isinstance(n, Scan))


def test_rule_registered():
    assert "pre_aggregate_beneath_dimension" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_reassociates_the_facts_with_the_plain_dimension():
    ds = _string_side_first()
    out = pre_aggregate_beneath_dimension(ds._plan, _ctx(ds, _NDV))
    assert isinstance(out, Aggregate) and isinstance(out.input, Join)
    partial = _partial(out)
    assert partial is not None, "the facts must be pre-aggregated"
    facts, dates = 0, 2
    assert _scans(partial) == [facts, dates], "the partial holds the facts and the plain dim"
    assert out.available_columns() == ds._plan.available_columns()


def test_pushes_beneath_the_top_join_when_the_plain_dimension_is_already_there():
    ds = _plain_side_first()
    out = pre_aggregate_beneath_dimension(ds._plan, _ctx(ds, _NDV))
    assert out is not None
    partial = _partial(out)
    assert partial is not None
    assert _scans(partial) == [0, 1]
    assert out.available_columns() == ds._plan.available_columns()


def test_idempotent():
    ds = _string_side_first()
    ctx = _ctx(ds, _NDV)
    once = pre_aggregate_beneath_dimension(ds._plan, ctx)
    assert pre_aggregate_beneath_dimension(once, ctx) is None


def test_no_fire_without_distinct_counts():
    """Cold, nothing proves a reduction, so nothing is pushed."""
    ds = _string_side_first()
    est = StatsEstimator(ds._sources, learned={})
    ctx = OptimizerContext(config=active_config(), sources=ds._sources, hub=None, estimator=est)
    assert pre_aggregate_beneath_dimension(ds._plan, ctx) is None


def test_no_fire_when_the_keys_promise_too_little():
    """Keys whose distinct counts multiply to most of the input cannot pay for a partial."""
    ds = _string_side_first()
    assert pre_aggregate_beneath_dimension(ds._plan, _ctx(ds, {**_NDV, "f_cust": 150.0})) is None


def test_no_fire_when_a_measure_reads_the_plain_dimension():
    """TPC-H q10's shape: re-associating would carry the measures across the upper join."""
    ds = (
        _facts()
        .join(_cust(), left_on="f_cust", right_on="c_sk")
        .join(_dates(), left_on="f_date", right_on="d_sk")
        .group_by("c_id")
        .agg(tot=col("d_year").sum())
    )
    assert pre_aggregate_beneath_dimension(ds._plan, _ctx(ds, _NDV)) is None


def test_no_fire_without_a_string_key():
    """The licence is the string work moved off the facts; integer keys alone earn none."""
    ds = (
        _facts()
        .join(_cust(), left_on="f_cust", right_on="c_sk")
        .join(_dates(), left_on="f_date", right_on="d_sk")
        .group_by("c_n", "d_year")
        .agg(tot=col("f_v").sum())
    )
    assert pre_aggregate_beneath_dimension(ds._plan, _ctx(ds, _NDV)) is None
