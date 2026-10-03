"""Which pre-aggregations the partitioned-build fallback may cross, on hand-built IR.

`chunked_sideways._partition_join` may restrict a build below an eager pre-aggregation, whose
passes then emit a group split across passes as several partial rows. That is exact only when
the aggregate above folds each partial value with the function that combines it and nothing
else reads one. Each refusal here is a shape that would return a wrong answer, so each is
paired with the accepted shape it differs from by one field.
"""

from __future__ import annotations

import pytest

from batcher.api.orchestration.chunked_sideways import _agg_reads, _partition_join

pytestmark = pytest.mark.unit

_SIZES = {0: 1_000, 1: 500, 2: 10}


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


def _plan(top_func: str, *, filter_partial: bool = False) -> dict:
    """`top_func(s)` over dim join [filter s > 0] over pre-agg `sum(v) AS s` over big join."""
    inner_join = {
        "op": "hash_join",
        "join_type": "inner",
        "left_keys": ["f"],
        "right_keys": ["b"],
        "output": [{"side": "left", "name": n, "alias": n} for n in ("k", "v")],
        "left": {"op": "scan", "source_id": 0},
        "right": {"op": "scan", "source_id": 1},
    }
    pre = {
        "op": "aggregate",
        "input": inner_join,
        "group_keys": [{"expr": _col("k"), "alias": "k"}],
        "aggregates": [{"func": "sum", "alias": "s", "input": _col("v")}],
    }
    if filter_partial:
        gt = {"e": "binary", "op": "gt", "left": _col("s"), "right": {"e": "lit", "value": 0}}
        pre = {"op": "filter", "input": pre, "predicate": gt}
    top = {
        "op": "aggregate",
        "input": {
            "op": "hash_join",
            "join_type": "inner",
            "left_keys": ["k"],
            "right_keys": ["d"],
            "output": [{"side": "left", "name": "s", "alias": "s"}],
            "left": pre,
            "right": {"op": "scan", "source_id": 2},
        },
        "group_keys": [],
        "aggregates": [{"func": top_func, "alias": "out", "input": _col("s")}],
    }
    return top


def _target(top: dict):
    return _partition_join(top["input"], _SIZES.__getitem__, 0, _agg_reads(top))


def test_a_sum_of_partial_sums_crosses_the_pre_aggregation():
    path, keys, size = _target(_plan("sum"))
    assert (path, keys, size) == (("left", "input", "right"), ["b"], 500)


@pytest.mark.parametrize("func", ["max", "min", "count", "count_star"])
def test_a_fold_that_does_not_combine_a_partial_sum_stays_above_it(func):
    path, _, size = _target(_plan(func))
    assert path == ("right",) and size == 10  # only the dimension, above the pre-aggregation


def test_a_filter_on_a_partial_value_stays_above_it():
    path, _, size = _target(_plan("sum", filter_partial=True))
    assert path == ("right",) and size == 10


def test_the_first_pass_count_does_not_charge_the_reported_overrun_to_the_other_builds():
    """The engine's reported bytes are what it had counted when it stopped, not the builds'
    total. TPC-H q9 under a 0.65 GB cap reported 470 MB beside a 192 MB side; charging the
    difference to the other builds asked for 120 passes (declined past 32) where 2 fit."""
    from batcher.api.orchestration.chunked_sideways import _MAX_PASSES, _pass_count

    assert _pass_count(469_546_331, 348_966_092, 192_000_000) == 2
    assert _pass_count(10**12, 10**6, 10**12) == _MAX_PASSES
    assert _pass_count(1_000, 1_000_000, 0) is None
