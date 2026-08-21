"""The salt fan-out is sized from the *measured* share, on the path that needs no opt-in.

`salt_factor` is `ceil(share x partitions)`, so what it is handed decides everything. The
column-statistics path — the cheapest of the three, the one that fires on a shape's first
ever run and needs no user opt-in — returned the hot *values* and threw the frequencies
away, so the caller had to substitute `skew_join_fraction`, the threshold at which a value
starts counting as hot. At the 0.10 default that is `ceil(0.10 x 8) = 1`, which floors to a
fan-out of 2 however skewed the key really is: exactly the mis-sizing `dist.skew` records
having fixed on the detection path (~12.5 s against ~1.9 s), left standing on the other one.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher import kyber
from batcher.dist.skew import salt_factor
from batcher.io.source.inmemory import InMemorySource
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hub import MetadataHub
from batcher.plan.logical import Join, JoinOutputCol, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import source_stats_key

pytestmark = pytest.mark.unit

_HOT = "7"
_SHARE = 0.40
_FRACTION = 0.10
_PARTITIONS = 8


def _source(rows: int) -> InMemorySource:
    return InMemorySource(
        pa.table({"k": pa.array(list(range(rows)), type=pa.int64())}).to_batches()
    )


def _join(left: InMemorySource, right: InMemorySource) -> Join:
    return Join(
        left=Scan(source_id=0, schema=SchemaRef.from_arrow(left.schema())),
        right=Scan(source_id=1, schema=SchemaRef.from_arrow(right.schema())),
        left_keys=("k",),
        right_keys=("k",),
        join_type="inner",
        output=(
            JoinOutputCol(side="left", name="k", alias="k"),
            JoinOutputCol(side="right", name="k", alias="k_r"),
        ),
    )


def _hub_with_measured_skew(fact: InMemorySource) -> MetadataHub:
    """A hub holding exactly what the metadata loop records for a skewed column."""
    hub = MetadataHub(InProcessBackend())
    key = source_stats_key(fact)
    assert key is not None, "the fixture source must be keyable, or nothing is measured"
    kyber.record_column_stats(hub, {"k": 1000.0}, {}, mcv={"k": {_HOT: _SHARE}}, source_key=key)
    return hub


def test_the_measured_share_travels_with_the_hot_values() -> None:
    fact, dim = _source(4_000), _source(1_000)
    hub = _hub_with_measured_skew(fact)
    hot, share = kyber.hot_join_value_shares(
        _join(fact, dim), [fact, dim], hub, _FRACTION, _PARTITIONS
    )
    assert hot == [_HOT]
    assert share == pytest.approx(_SHARE), (
        "the frequency is read off the same table the hot set came from, so returning the "
        "values without it forces the caller to substitute the detection threshold"
    )


def test_the_share_sizes_the_fan_out_and_the_threshold_does_not() -> None:
    # What the measurement buys, stated as the two numbers that differ.
    assert salt_factor(_SHARE, _PARTITIONS) == 4  # ceil(0.40 x 8)
    assert salt_factor(_FRACTION, _PARTITIONS) == 2  # ceil(0.10 x 8) -> 1, floored to 2


def test_an_unmeasured_column_reports_no_share_rather_than_a_wrong_one() -> None:
    fact, dim = _source(4_000), _source(1_000)
    cold = MetadataHub(InProcessBackend())
    hot, share = kyber.hot_join_value_shares(
        _join(fact, dim), [fact, dim], cold, _FRACTION, _PARTITIONS
    )
    assert hot == []
    assert share == 0.0
