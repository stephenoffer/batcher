"""The learned per-column statistics tables stay bounded, and stay useful once bounded.

Each table is a single store entry holding `{source ⟂ column: value}` for every column of
every source ever measured. It is read whole on every plan and written whole on every
measurement, so its size is a per-query cost — and its key space grows by one entry per new
source identity, which a workload over dated partitions or per-tenant files produces on
every run. These pin the cap, the eviction order, and the no-op fast path.
"""

from __future__ import annotations

import pytest

from batcher.kyber.column_tables import (
    _TABLE_MAX,
    AVG_BYTES_KEY,
    MCV_KEY,
    NDV_KEY,
    QUANTILES_KEY,
    STATS_NAMESPACE,
    columns_for,
    merge_column_table,
    qualify,
)
from batcher.kyber.learning import record_column_stats
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hub import MetadataHub

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _table(hub: MetadataHub, key: str) -> dict:
    return hub.get_keyed_param(STATS_NAMESPACE, key) or {}


class TestTheTableIsBounded:
    def test_a_stream_of_new_sources_does_not_grow_the_table_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hub = _hub()
        cap = 40
        monkeypatch.setitem(_TABLE_MAX, NDV_KEY, cap)
        # One fresh source identity per run is the ordinary shape of a partitioned or
        # per-tenant workload, not an exotic one.
        for i in range(cap + 25):
            merge_column_table(hub, NDV_KEY, {qualify(f"s{i}", "id"): float(i)})

        table = _table(hub, NDV_KEY)
        assert len(table) == cap
        assert qualify("s0", "id") not in table  # the oldest went
        assert qualify(f"s{cap + 24}", "id") in table  # the newest stayed

    def test_each_table_has_its_own_cap(self) -> None:
        # A quantile grid and an MCV map cost far more per entry than a bare float, so the
        # entry counts are deliberately not the same number.
        assert _TABLE_MAX[QUANTILES_KEY] < _TABLE_MAX[NDV_KEY]
        assert _TABLE_MAX[MCV_KEY] < _TABLE_MAX[AVG_BYTES_KEY]

    def test_eviction_prefers_the_least_recently_written(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hub = _hub()
        cap = 20
        monkeypatch.setitem(_TABLE_MAX, QUANTILES_KEY, cap)
        grid = {"probs": [0.0, 1.0], "values": [0.0, 1.0]}
        for i in range(cap):
            merge_column_table(hub, QUANTILES_KEY, {qualify(f"s{i}", "c"): grid})

        hot = qualify("s0", "c")
        # A *changed* measurement of the oldest entry moves it to the end, so the next
        # overflow takes the entry behind it rather than the one just written.
        merge_column_table(hub, QUANTILES_KEY, {hot: {"probs": [0.5], "values": [2.0]}})
        merge_column_table(hub, QUANTILES_KEY, {qualify("new", "c"): grid})

        table = _table(hub, QUANTILES_KEY)
        assert len(table) == cap
        assert hot in table
        assert qualify("s1", "c") not in table  # the oldest untouched entry went instead


class TestTheSteadyStateIsCheap:
    def test_re_measuring_the_same_values_writes_nothing(self) -> None:
        hub = _hub()
        merge_column_table(hub, NDV_KEY, {qualify("src", "id"): 100.0})
        writes = 0
        inner = hub.put_keyed_param

        def counting_put(namespace: str, key: str, value: object) -> None:
            nonlocal writes
            writes += 1
            inner(namespace, key, value)

        hub.put_keyed_param = counting_put  # type: ignore[method-assign]

        for _ in range(10):
            merge_column_table(hub, NDV_KEY, {qualify("src", "id"): 100.0})

        assert writes == 0

    def test_a_changed_value_still_writes(self) -> None:
        hub = _hub()
        merge_column_table(hub, NDV_KEY, {qualify("src", "id"): 100.0})
        merge_column_table(hub, NDV_KEY, {qualify("src", "id"): 250.0})
        assert _table(hub, NDV_KEY)[qualify("src", "id")] == 250.0

    def test_a_new_column_beside_known_ones_still_writes(self) -> None:
        hub = _hub()
        merge_column_table(hub, NDV_KEY, {qualify("src", "a"): 1.0})
        merge_column_table(hub, NDV_KEY, {qualify("src", "a"): 1.0, qualify("src", "b"): 2.0})
        assert _table(hub, NDV_KEY) == {qualify("src", "a"): 1.0, qualify("src", "b"): 2.0}


class TestRecordColumnStatsStillWorks:
    """The public writer keeps its behavior on top of the bounded merge."""

    def test_it_records_all_four_tables_qualified_by_source(self) -> None:
        hub = _hub()
        record_column_stats(
            hub,
            ndv={"id": 10.0},
            quantiles={"id": {"probs": [0.0, 1.0], "values": [1.0, 9.0]}},
            avg_bytes={"id": 8.0},
            mcv={"id": {"3": 0.5}},
            source_key="src",
        )
        learned = hub.load_keyed_params(STATS_NAMESPACE)

        assert columns_for(learned, NDV_KEY, "src") == {"id": 10.0}
        assert columns_for(learned, AVG_BYTES_KEY, "src") == {"id": 8.0}
        assert columns_for(learned, MCV_KEY, "src") == {"id": {"3": 0.5}}
        assert columns_for(learned, QUANTILES_KEY, "src")["id"]["values"] == [1.0, 9.0]

    def test_another_source_with_the_same_column_name_stays_separate(self) -> None:
        hub = _hub()
        record_column_stats(hub, ndv={"id": 10.0}, quantiles={}, source_key="a")
        record_column_stats(hub, ndv={"id": 99.0}, quantiles={}, source_key="b")
        learned = hub.load_keyed_params(STATS_NAMESPACE)

        assert columns_for(learned, NDV_KEY, "a") == {"id": 10.0}
        assert columns_for(learned, NDV_KEY, "b") == {"id": 99.0}

    def test_a_first_measurement_bumps_the_plan_generation(self) -> None:
        from batcher.kyber.learning import generation

        hub = _hub()
        before = generation()
        record_column_stats(hub, ndv={"id": 10.0}, quantiles={}, source_key="src")
        assert generation() > before

        # ...and re-measuring the same column does not, or no plan would ever be reused.
        settled = generation()
        record_column_stats(hub, ndv={"id": 10.0}, quantiles={}, source_key="src")
        assert generation() == settled

    def test_a_legacy_unqualified_entry_is_still_honored(self) -> None:
        hub = _hub()
        hub.put_keyed_param(STATS_NAMESPACE, NDV_KEY, {"id": 7.0})
        learned = hub.load_keyed_params(STATS_NAMESPACE)
        assert columns_for(learned, NDV_KEY, "any-source") == {"id": 7.0}

        # A measurement of this source beats the legacy global one.
        record_column_stats(hub, ndv={"id": 42.0}, quantiles={}, source_key="any-source")
        learned = hub.load_keyed_params(STATS_NAMESPACE)
        assert columns_for(learned, NDV_KEY, "any-source") == {"id": 42.0}
