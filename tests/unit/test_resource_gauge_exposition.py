"""Carbonite's resource gauges have to be identifiable in a scrape, not merely present.

`observe.counters.resources` flattens every `stats()` reading into a gauge generically, which
is what let the buffer pool, the spill store, the admission limiter and the shuffle session
start being exported without a change there. The cost of that genericity was that it had no
curated HELP text to print -- so it printed none, and these were the only series in the whole
exposition without one.

That is not a cosmetic gap. `# HELP` is what Grafana's metric browser shows as a series'
description and what `promtool check metrics` looks for, so the readings Carbonite works
hardest to produce were exactly the ones an operator could not identify without reading
Batcher's source. A description derived from the group and the field path says everything a
curated line would, and unlike a hand-written table it cannot fall behind the fields.

The per-group cardinality cap has the matching problem: it dropped series silently, which is
indistinguishable from the resource never having reported them.
"""

from __future__ import annotations

import re

import pytest

from batcher.observe.counters.resources import _MAX_SERIES_PER_GROUP, ResourceGauges

pytestmark = pytest.mark.unit


def _families(lines: list[str]) -> dict[str, set[str]]:
    """Metric name -> the metadata line kinds (`HELP`, `TYPE`) the exposition gave it."""
    seen: dict[str, set[str]] = {}
    for line in lines:
        match = re.match(r"^# (HELP|TYPE) (\S+)", line)
        if match:
            seen.setdefault(match.group(2), set()).add(match.group(1))
        elif line and not line.startswith("#"):
            seen.setdefault(re.split(r"[ {]", line, maxsplit=1)[0], set())
    return seen


def test_every_gauge_carries_help_as_well_as_type():
    """The property that was missing: a scraped series says what it is."""
    gauges = ResourceGauges()
    gauges.record("memory", {"pool": {"used_bytes": 4, "limit_bytes": 8}, "pressure": "NOMINAL"})
    families = _families(gauges.render())
    assert families, "nothing rendered"
    for name, kinds in families.items():
        if name == "batcher_resource_series_dropped":
            continue
        assert "HELP" in kinds, f"{name} has no HELP line"
        assert "TYPE" in kinds, f"{name} has no TYPE line"


def test_the_help_text_names_the_field_and_the_group():
    """Derived, not curated -- so it has to actually read as a description."""
    gauges = ResourceGauges()
    gauges.record("memory", {"pool": {"used_bytes": 4}})
    line = next(x for x in gauges.render() if x.startswith("# HELP"))
    assert "pool used bytes" in line
    assert "buffer-pool" in line
    # A gauge is a level; saying so is what stops a consumer differencing it into noise.
    assert "level" in line


def test_help_text_is_a_single_line_with_nothing_to_escape():
    """A HELP line carrying a newline or a backslash breaks the whole exposition."""
    gauges = ResourceGauges()
    gauges.record("weird\ngroup", {"a\\b": 1})
    for line in gauges.render():
        assert "\n" not in line
        assert "\\" not in line


def test_a_capped_group_says_how_many_series_it_dropped():
    """The cap is right; its silence was not. An operator can see the readings are partial."""
    gauges = ResourceGauges()
    over = _MAX_SERIES_PER_GROUP + 7
    gauges.record("shuffle", {f"f{i:04d}": i for i in range(over)})
    lines = gauges.render()
    assert "batcher_resource_series_dropped 7" in lines


def test_an_uncapped_group_reports_zero_dropped_rather_than_omitting_the_series():
    """A gauge that appears only on failure has no baseline to alert against."""
    gauges = ResourceGauges()
    gauges.record("spill", {"tier_bytes": 1, "free_bytes": 2})
    assert "batcher_resource_series_dropped 0" in gauges.render()


def test_nothing_is_rendered_before_a_reading_arrives():
    """No reading is not the same as a reading of zero, and must not export as one."""
    assert ResourceGauges().render() == []


def test_a_string_leaf_is_still_a_state_set_with_its_own_help():
    """The enumerated-level form keeps its label and gains the description it lacked."""
    gauges = ResourceGauges()
    gauges.record("memory", {"pressure": "HIGH"})
    lines = gauges.render()
    assert 'batcher_memory_pressure{state="HIGH"} 1' in lines
    assert any(x.startswith("# HELP batcher_memory_pressure ") for x in lines)


class TestBaseUnits:
    """Prometheus asks for base units, and every duration Batcher exported was milliseconds.

    Not a cosmetic deviation. `histogram_quantile` returns a figure in the bucket's own unit,
    so the shipped Grafana panel plotting query latency declared `unit: ms` and would have
    read `0.09` for a 90 ms query the moment anyone corrected the series. Three other panels
    were dividing by 1000 in their own expressions -- the consumer compensating for the
    producer, which is the tell that the exporter had it wrong.
    """

    @staticmethod
    def _exposition() -> str:
        import batcher as bt
        from batcher.observe import prometheus_text, reset_metrics, start_metrics, stop_metrics

        start_metrics()
        try:
            reset_metrics()
            bt.from_pydict({"v": [1.0, 2.0, 3.0, 4.0] * 50}).filter(bt.col("v") > 1).collect()
            return prometheus_text()
        finally:
            reset_metrics()
            stop_metrics()

    def test_no_series_is_named_for_milliseconds(self):
        """The rule, stated as the thing a scrape can see."""
        offenders = sorted(set(re.findall(r"batcher_[a-z_]*_ms[a-z_]*", self._exposition())))
        assert not offenders, f"millisecond-named series: {offenders}"

    def test_the_duration_histogram_is_in_seconds(self):
        """Its buckets are the same boundaries, expressed in the unit the name claims."""
        text = self._exposition()
        assert "batcher_query_duration_seconds_bucket" in text
        edges = re.findall(r'batcher_query_duration_seconds_bucket\{le="([^"]+)"\}', text)
        assert "0.001" in edges, edges
        assert "+Inf" in edges

    def test_the_histogram_sum_is_scaled_with_its_buckets(self):
        """A sum left in milliseconds against second-valued buckets is a silent, wrong mean.

        The positive control for the rename: a sub-second query has to report a sub-1 sum,
        which is exactly what an unconverted `_sum` would not.
        """
        text = self._exposition()
        total = float(re.search(r"batcher_query_duration_seconds_sum (\S+)", text).group(1))
        count = float(re.search(r"batcher_query_duration_seconds_count (\S+)", text).group(1))
        assert count == 1
        assert 0 < total < 60, f"a small query reported {total} seconds"

    def test_cpu_and_execution_totals_are_seconds(self):
        """Both were divided by 1000 in the dashboard's own expressions before this."""
        text = self._exposition()
        for metric in ("batcher_cpu_seconds_total", "batcher_execution_seconds_total"):
            value = float(re.search(rf"^{metric} (\S+)$", text, re.M).group(1))
            assert 0 <= value < 60, f"{metric} = {value} does not read as seconds"

    def test_byte_series_are_untouched(self):
        """The control: only durations were wrong, and the fix must not rescale bytes."""
        text = self._exposition()
        scanned = float(re.search(r"^batcher_bytes_scanned_total (\S+)$", text, re.M).group(1))
        assert scanned >= 1000, f"bytes look rescaled: {scanned}"
