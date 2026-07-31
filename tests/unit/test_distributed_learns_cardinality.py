"""A distributed run learned its scheduling knobs and forgot its cardinality.

`record_distributed` closes the scheduling loops — worker fan-out, credit window. The
plan-level learned state is a different loop, and it ran on the single-node path only: the
distributed route never reached `record_cardinality_outcome`. So the identical query learned
from running on one node and learned nothing from running on twenty, which is backwards —
the distributed workload is the long-running, repeatedly-issued one.

These exercise the recorder directly rather than through Ray, so they pin the contract
without needing a cluster: what counts as a countable result, what declines, and that a
failure cannot break a run that already produced its answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.orchestration.stages import _record_distributed_cardinality
from batcher.kyber.learning import load_learned_stats
from batcher.kyber.signature import plan_signature
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hub import MetadataHub

pytestmark = pytest.mark.unit


class _PartitionedResult:
    """A result left on the fleet: it reports its rows from handles, fetching nothing."""

    def __init__(self, rows: int) -> None:
        self._rows = rows
        self.fetched = False

    def row_count(self) -> int | None:
        return self._rows


class _OpaqueResult:
    """A result that cannot say how many rows it has without being materialized."""


def _learned_rows(hub: MetadataHub, plan) -> float | None:
    entry = load_learned_stats(hub).get(plan_signature(plan))
    return None if entry is None else entry.get("rows")


@pytest.fixture
def plan_and_sources():
    ds = bt.from_pydict({"x": [1, 2, 3, 4]}).filter(bt.col("x") > 1)
    return ds._plan, ds._sources


class TestTheCardinalityLoopCloses:
    def test_a_materialized_table_is_recorded(self, plan_and_sources) -> None:
        plan, sources = plan_and_sources
        hub = MetadataHub(InProcessBackend())

        _record_distributed_cardinality(hub, plan, sources, pa.table({"x": [2, 3, 4]}))

        assert _learned_rows(hub, plan) == pytest.approx(3.0)

    def test_a_partitioned_result_is_recorded_without_fetching(self, plan_and_sources) -> None:
        plan, sources = plan_and_sources
        hub = MetadataHub(InProcessBackend())
        result = _PartitionedResult(rows=7)

        _record_distributed_cardinality(hub, plan, sources, result)

        assert _learned_rows(hub, plan) == pytest.approx(7.0)
        # Learning must not change what the run costs.
        assert result.fetched is False

    def test_an_uncountable_result_declines_rather_than_materializing(
        self, plan_and_sources
    ) -> None:
        plan, sources = plan_and_sources
        hub = MetadataHub(InProcessBackend())

        _record_distributed_cardinality(hub, plan, sources, _OpaqueResult())

        assert _learned_rows(hub, plan) is None

    def test_an_empty_result_is_a_real_measurement(self, plan_and_sources) -> None:
        # Zero rows is a fact about the data, not a missing reading — a predicate that
        # matched nothing is exactly the cardinality a future plan wants to know.
        plan, sources = plan_and_sources
        hub = MetadataHub(InProcessBackend())

        empty = pa.table({"x": pa.array([], pa.int64())})
        _record_distributed_cardinality(hub, plan, sources, empty)

        assert _learned_rows(hub, plan) == pytest.approx(0.0)

    def test_a_none_hub_is_a_no_op(self, plan_and_sources) -> None:
        plan, sources = plan_and_sources
        _record_distributed_cardinality(None, plan, sources, pa.table({"x": [1]}))

    def test_a_failing_recorder_does_not_break_a_finished_run(self, plan_and_sources) -> None:
        plan, sources = plan_and_sources

        class _Exploding:
            def row_count(self) -> int:
                raise RuntimeError("handles unavailable")

        # The answer is already produced. Counting a partitioned result reaches the fleet's
        # handles, so a worker lost in the gap would otherwise turn a completed query into a
        # failed one — caught the same as a hub write failing.
        hub = MetadataHub(InProcessBackend())
        _record_distributed_cardinality(hub, plan, sources, _Exploding())
        assert _learned_rows(hub, plan) is None


class TestItAgreesWithTheSingleNodePath:
    def test_the_same_query_learns_the_same_rows_either_way(self, plan_and_sources) -> None:
        plan, sources = plan_and_sources
        single = MetadataHub(InProcessBackend())
        distributed = MetadataHub(InProcessBackend())

        from batcher.api.orchestration.run import record_cardinality_outcome

        record_cardinality_outcome(single, plan, sources, 3)
        _record_distributed_cardinality(distributed, plan, sources, pa.table({"x": [2, 3, 4]}))

        assert _learned_rows(single, plan) == _learned_rows(distributed, plan)
