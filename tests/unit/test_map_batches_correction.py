"""`map_batches` is a learnable fan-out, keyed by the UDF's identity.

A UDF may filter, explode, or pass rows through 1:1 — which one is a property of the
*code*, invisible to the structural estimator. So `MapBatches` is `_CORRECTABLE`: the
measured fan-out from a past run corrects the 1:1 guess, exactly as `Unnest` does.

That is only sound because the plan signature now distinguishes UDFs (`_udf_identity`). If
every `map_batches` over the same input shared one learned entry, a filtering dedup and an
exploding chunker — the two ends of an AI pipeline — would poison each other, the same
defect that once let one table's row count answer for another's.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.learning import CARDINALITY_CORRECTION_KEY
from batcher.kyber.signature import plan_signature
from batcher.kyber.stats import StatsEstimator
from batcher.plan.logical import MapBatches

pytestmark = pytest.mark.unit


def _chunk(batch):  # a named, exploding-shaped UDF
    return batch


def _dedup(batch):  # a different named UDF
    return batch


class _Embed:  # a factory-class UDF (build-once-per-worker pattern)
    def __call__(self, batch):
        return batch


class _Source:
    def __init__(self, rows: int) -> None:
        self._rows = rows

    def row_count(self) -> int:
        return self._rows


def _base():
    return bt.from_pydict({"a": [1, 2, 3]})._plan


def test_distinct_udfs_get_distinct_signatures():
    sigs = {
        plan_signature(MapBatches(input=_base(), fn=_chunk)),
        plan_signature(MapBatches(input=_base(), fn=_dedup)),
        plan_signature(MapBatches(input=_base(), fn=_Embed())),
    }
    assert len(sigs) == 3


def test_same_udf_is_stable_across_rebuilds():
    a = plan_signature(MapBatches(input=_base(), fn=_chunk))
    b = plan_signature(MapBatches(input=_base(), fn=_chunk))
    assert a == b


def test_cold_map_batches_is_one_to_one():
    est = StatsEstimator([_Source(100)])
    assert est.estimate(MapBatches(input=_base(), fn=_chunk)).rows == pytest.approx(100.0)


def test_measured_fan_out_corrects_the_estimate():
    node = MapBatches(input=_base(), fn=_chunk)
    learned = {CARDINALITY_CORRECTION_KEY: {plan_signature(node): 20.0}}
    est = StatsEstimator([_Source(100)], learned=learned)
    assert est.estimate(node).rows == pytest.approx(2000.0)


def test_one_udf_correction_does_not_poison_another():
    chunk = MapBatches(input=_base(), fn=_chunk)
    dedup = MapBatches(input=_base(), fn=_dedup)
    learned = {CARDINALITY_CORRECTION_KEY: {plan_signature(chunk): 20.0}}
    est = StatsEstimator([_Source(100)], learned=learned)
    assert est.estimate(chunk).rows == pytest.approx(2000.0)
    assert est.estimate(dedup).rows == pytest.approx(100.0)  # untouched
