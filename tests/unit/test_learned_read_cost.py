"""A scanned byte is not the same price everywhere, and the cost model now knows it.

Core has always measured each source's read throughput. Nothing outside `explain` consumed
it, so a plan joining a cold object-store table against a page-cache-warm one was ranked as
though reading either were equally cheap — and which side to build, which to broadcast, and
which order to join in all turn on exactly that comparison.

The multipliers are relative to the plan's *median* measured source, which is what makes the
change a sharpening rather than a re-tuning: only ratios between this plan's sources can move
a ranking, so the `io` axis is never re-scaled against `cpu`, and a plan with nothing measured
is ranked byte-for-byte as before.
"""

from __future__ import annotations

import pytest

from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hub import MetadataHub
from batcher.metadata.io_stats import (
    _DEAD_BAND,
    _MIN_MEASURED_BYTES,
    _READ_COST_CLAMP,
    load_source_throughput_mbps,
    record_source_io,
    relative_read_cost,
)

pytestmark = pytest.mark.unit

_MB = 1024 * 1024


def _hub_with(throughputs: dict[str, float]) -> MetadataHub:
    """A hub whose sources have each been read once at the given MB/s.

    Each read is scaled to ten seconds so every one of them clears the volume floor below
    which a read is fixed overhead rather than a throughput measurement.
    """
    hub = MetadataHub(InProcessBackend())
    for identity, mbps in throughputs.items():
        record_source_io(hub, identity, int(mbps * _MB * 10), 10_000.0)
    return hub


class TestRelativeReadCost:
    def test_a_cold_store_prices_every_source_alike(self) -> None:
        hub = MetadataHub(InProcessBackend())
        assert relative_read_cost(hub, ["a", "b"]) == [1.0, 1.0]

    def test_no_hub_prices_every_source_alike(self) -> None:
        assert relative_read_cost(None, ["a", "b"]) == [1.0, 1.0]

    def test_a_single_source_has_no_ratio_to_read(self) -> None:
        # One source cannot be compared to anything, and re-scaling the io axis against cpu
        # off an absolute throughput would be a re-tuning of the model, not a sharpening.
        hub = _hub_with({"a": 50.0})
        assert relative_read_cost(hub, ["a"]) == [1.0]

    def test_one_measured_source_among_several_is_not_enough(self) -> None:
        hub = _hub_with({"a": 50.0})
        assert relative_read_cost(hub, ["a", "b", "c"]) == [1.0, 1.0, 1.0]

    def test_the_slow_source_costs_more_per_byte(self) -> None:
        hub = _hub_with({"slow": 50.0, "fast": 200.0})
        slow, fast = relative_read_cost(hub, ["slow", "fast"])
        assert slow > 1.0 > fast
        # Relative to the median of the two, which is their mean at 125 MB/s.
        assert slow == pytest.approx(125.0 / 50.0)
        assert fast == pytest.approx(125.0 / 200.0)

    def test_sources_on_the_same_storage_are_priced_the_same(self) -> None:
        # Compression ratio, column widths, cache warmth and scheduling luck move a measured
        # MB/s by tens of percent between two relations on one disk. None of that is a reason
        # to re-rank a plan, so inside the dead band the factor is exactly 1.0.
        hub = _hub_with({"a": 100.0, "b": 130.0, "c": 85.0})
        assert relative_read_cost(hub, ["a", "b", "c"]) == [1.0, 1.0, 1.0]

    def test_a_genuinely_different_storage_class_still_gets_through(self) -> None:
        hub = _hub_with({"nvme": 2000.0, "local": 1800.0, "cold_object_store": 40.0})
        nvme, local, cold = relative_read_cost(hub, ["nvme", "local", "cold_object_store"])
        assert (nvme, local) == (1.0, 1.0)
        assert cold > _DEAD_BAND

    def test_an_unmeasured_source_gets_the_reference_price(self) -> None:
        # Not a guess in either direction: charging it as slow punishes a source for being
        # new, and charging it as fast rewards the same thing.
        hub = _hub_with({"slow": 50.0, "fast": 200.0})
        factors = relative_read_cost(hub, ["slow", "fast", "unseen"])
        assert factors[2] == 1.0

    def test_an_extreme_spread_is_clamped(self) -> None:
        # One learned number, smoothed from a handful of possibly-unlucky reads, must not be
        # able to dominate the model however far it sits from the median.
        hub = _hub_with({"glacial": 1.0, "typical": 100.0, "quick": 100_000.0})
        glacial, typical, quick = relative_read_cost(hub, ["glacial", "typical", "quick"])
        assert glacial == pytest.approx(_READ_COST_CLAMP)
        assert quick == pytest.approx(1.0 / _READ_COST_CLAMP)
        assert typical == pytest.approx(1.0)

    def test_a_uniformly_faster_fleet_changes_nothing(self) -> None:
        # The baseline is this plan's own median, so nothing about the *comparison* moved.
        slow_box = relative_read_cost(_hub_with({"a": 50.0, "b": 200.0}), ["a", "b"])
        fast_box = relative_read_cost(_hub_with({"a": 500.0, "b": 2000.0}), ["a", "b"])
        assert slow_box == pytest.approx(fast_box)

    def test_an_empty_source_list_is_handled(self) -> None:
        assert relative_read_cost(_hub_with({"a": 1.0}), []) == []

    def test_an_unidentifiable_source_gets_the_reference_price(self) -> None:
        hub = _hub_with({"slow": 50.0, "fast": 200.0})
        assert relative_read_cost(hub, ["slow", "fast", ""])[2] == 1.0


class TestSmallReadsAreNotThroughput:
    """A read too small to be dominated by byte movement measures overhead, not bandwidth."""

    def test_a_tiny_read_is_not_recorded(self) -> None:
        hub = MetadataHub(InProcessBackend())
        # A 25-row dimension table. Its read time is opening the source, not moving bytes —
        # and because that cost barely varies with size, the *smaller* the relation the slower
        # it appears. On TPC-H sf1 this made `nation` and `region` read as the most expensive
        # sources in the query and re-ranked every plan that touched them.
        record_source_io(hub, "nation", 2_000, 0.4)
        assert load_source_throughput_mbps(hub, "nation") is None

    def test_a_sub_millisecond_read_is_not_recorded(self) -> None:
        hub = MetadataHub(InProcessBackend())
        record_source_io(hub, "s", _MIN_MEASURED_BYTES * 4, 0.2)
        assert load_source_throughput_mbps(hub, "s") is None

    def test_a_read_past_the_floors_is_recorded(self) -> None:
        hub = MetadataHub(InProcessBackend())
        record_source_io(hub, "s", _MIN_MEASURED_BYTES * 10, 1000.0)
        assert load_source_throughput_mbps(hub, "s") == pytest.approx(
            _MIN_MEASURED_BYTES * 10 / _MB
        )

    def test_tiny_relations_no_longer_re_rank_a_plan(self) -> None:
        hub = MetadataHub(InProcessBackend())
        record_source_io(hub, "region", 500, 0.3)  # 5 rows
        record_source_io(hub, "nation", 2_000, 0.4)  # 25 rows
        record_source_io(hub, "part", 40 * _MB, 100.0)  # 200k rows
        assert relative_read_cost(hub, ["region", "nation", "part"]) == [1.0, 1.0, 1.0]


class TestThroughputIsMachineScoped:
    def test_a_figure_learned_on_other_hardware_is_not_adopted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from batcher.metadata import hardware_scope

        hub = MetadataHub(InProcessBackend())
        record_source_io(hub, "s3://bucket/t.parquet", 100 * _MB, 1000.0)
        assert load_source_throughput_mbps(hub, "s3://bucket/t.parquet") == pytest.approx(100.0)

        # The same store, read from a different machine class. MB/s is the NIC, the disk, the
        # page cache, and the decompressing cores — none of which transfers, so blending a
        # driver's local read with a worker's cold S3 read gives a figure wrong for both.
        monkeypatch.setattr(hardware_scope, "fingerprint", lambda: "different-box")
        assert load_source_throughput_mbps(hub, "s3://bucket/t.parquet") is None


class TestCostModelUsesIt:
    def _model(self, factors: list[float] | None):
        import pyarrow as pa

        from batcher.kyber.cardinality import CardinalityEstimator
        from batcher.kyber.cost import CostModel
        from batcher.plan.logical import Scan
        from batcher.plan.schema import SchemaRef

        schema = SchemaRef(pa.schema([pa.field("a", pa.int64())]))
        estimator = CardinalityEstimator([None, None], {})
        model = CostModel(estimator, source_io_factors=factors)
        return model, Scan(source_id=0, schema=schema), Scan(source_id=1, schema=schema)

    def test_no_factors_prices_scans_identically(self) -> None:
        model, left, right = self._model(None)
        assert model.op_cost(left).io == model.op_cost(right).io

    def test_the_expensive_source_is_priced_higher(self) -> None:
        model, slow, fast = self._model([2.5, 0.4])
        assert model.op_cost(slow).io == pytest.approx(model.op_cost(fast).io * (2.5 / 0.4))

    def test_only_the_io_axis_moves(self) -> None:
        # The factor describes what a *byte* costs. Charging cpu for it too would double-count
        # the same effect on an axis that is not about storage at all.
        model, slow, fast = self._model([4.0, 1.0])
        assert model.op_cost(slow).cpu == model.op_cost(fast).cpu

    def test_a_source_id_outside_the_vector_is_the_reference(self) -> None:
        model, left, _right = self._model([3.0])
        plain, _l, _r = self._model(None)
        assert model.op_cost(left).io == pytest.approx(plain.op_cost(left).io * 3.0)


class TestPlanCacheSeesIt:
    def test_the_key_moves_when_a_source_gets_relatively_slower(self) -> None:
        from batcher.kyber.plan_cache import _read_cost_key

        class _Src:
            def __init__(self, name: str) -> None:
                self._name = name

            def identity(self) -> str:
                return self._name

        sources = [_Src("a"), _Src("b")]
        even = _read_cost_key(_hub_with({"a": 100.0, "b": 100.0}), sources)
        skewed = _read_cost_key(_hub_with({"a": 25.0, "b": 400.0}), sources)
        assert even != skewed

    def test_the_key_is_stable_under_a_small_drift(self) -> None:
        from batcher.kyber.plan_cache import _read_cost_key

        class _Src:
            def __init__(self, name: str) -> None:
                self._name = name

            def identity(self) -> str:
                return self._name

        sources = [_Src("a"), _Src("b")]
        # Keyed on the raw factor the memo would miss on every query and cease to exist.
        base = _read_cost_key(_hub_with({"a": 100.0, "b": 120.0}), sources)
        drifted = _read_cost_key(_hub_with({"a": 101.0, "b": 119.0}), sources)
        assert base == drifted

    def test_a_cold_store_keys_as_before(self) -> None:
        from batcher.kyber.plan_cache import _read_cost_key

        assert _read_cost_key(None, None) == "-"
        assert _read_cost_key(MetadataHub(InProcessBackend()), None) == "-"
