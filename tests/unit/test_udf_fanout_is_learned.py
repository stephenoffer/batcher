"""A UDF's fan-out is unknowable structurally, perfectly learnable, and was discarded.

The estimator lists `MapBatches` in its `_CORRECTABLE` set and keys the correction on the
UDF's identity, with the reasoning written out: a UDF may filter, explode, or pass rows
through 1:1, and which one is a property of the *code*. Absent a measurement it assumes 1:1.

Nothing supplied the measurement. Both UDF routes bypass `run_relational`, the one place
`record_cardinality_outcome` runs, so every pipeline's measured output count was thrown
away. These pin that the loop now closes, and that it stays keyed per UDF.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.learning import load_learned_stats
from batcher.kyber.signature import plan_signature
from batcher.metadata.hub import MetadataHub

pytestmark = pytest.mark.unit


@pytest.fixture
def hub() -> MetadataHub:
    """A fresh process hub, so one test's learned fan-out cannot answer for another's."""
    from batcher.core.runtime import default_hub, reset_default_hub

    reset_default_hub()
    yield default_hub()
    reset_default_hub()


def explode_four(batch: pa.RecordBatch) -> pa.RecordBatch:
    """One row in, four out — the shape an object-detection stage has."""
    table = pa.Table.from_batches([batch])
    return pa.concat_tables([table] * 4).combine_chunks().to_batches()[0]


def keep_none(batch: pa.RecordBatch) -> pa.RecordBatch:
    """A classifier that keeps nothing — the other extreme of the same unknowability."""
    return batch.slice(0, 0)


def _learned_rows(hub: MetadataHub, plan) -> float | None:
    entry = load_learned_stats(hub).get(plan_signature(plan))
    return None if entry is None else entry.get("rows")


class TestTheFanOutIsRecorded:
    def test_an_exploding_udf_records_its_measured_output(self, hub: MetadataHub) -> None:
        ds = bt.from_pydict({"x": [1, 2, 3]}).map_batches(explode_four)
        plan = ds._plan

        assert _learned_rows(hub, plan) is None  # nothing known before it runs
        out = ds.collect()

        assert out.num_rows == 12
        # 1:1 is what the structural estimator assumes; the measurement says otherwise.
        assert _learned_rows(hub, plan) == pytest.approx(12.0)

    def test_a_filtering_udf_records_its_measured_output(self, hub: MetadataHub) -> None:
        ds = bt.from_pydict({"x": [1, 2, 3]}).map_batches(keep_none)
        plan = ds._plan

        assert ds.collect().num_rows == 0
        assert _learned_rows(hub, plan) == pytest.approx(0.0)

    def test_two_udfs_over_the_same_input_do_not_share_a_fan_out(self, hub: MetadataHub) -> None:
        # The whole point of keying on UDF identity: an exploding stage's measurement must
        # not answer for a filtering one, however alike their plans look.
        exploding = bt.from_pydict({"x": [1, 2, 3]}).map_batches(explode_four)
        filtering = bt.from_pydict({"x": [1, 2, 3]}).map_batches(keep_none)

        exploding.collect()
        filtering.collect()

        assert _learned_rows(hub, exploding._plan) == pytest.approx(12.0)
        assert _learned_rows(hub, filtering._plan) == pytest.approx(0.0)

    def test_the_estimate_moves_toward_the_measurement(self, hub: MetadataHub) -> None:
        from batcher.kyber.cardinality import CardinalityEstimator

        ds = bt.from_pydict({"x": list(range(10))}).map_batches(explode_four)
        plan = ds._plan

        srcs = ds._sources
        cold = CardinalityEstimator(srcs, {}).estimate(plan).rows
        ds.collect()
        warm = CardinalityEstimator(srcs, load_learned_stats(hub)).estimate(plan).rows

        # Cold, the structural estimator can only assume the UDF is 1:1.
        assert cold == pytest.approx(10.0)
        # Warm, it knows the stage quadruples its input.
        assert warm > cold

    def test_a_run_without_a_hub_still_works(self) -> None:
        assert bt.from_pydict({"x": [1, 2]}).map_batches(explode_four).collect().num_rows == 8


class TestItDoesNotDisturbTheRelationalLoop:
    def test_a_plan_with_no_udf_is_unaffected(self, hub: MetadataHub) -> None:
        ds = bt.from_pydict({"x": [1, 2, 3, 4]}).filter(bt.col("x") > 2)
        assert ds.collect().num_rows == 2
        # The relational route records through `run_relational` as it always did.
        assert _learned_rows(hub, ds._plan) == pytest.approx(2.0)
