"""A data contract that is checked and never charted is one nobody notices degrading.

`ds.dq.validate()` answers "is today's data good". It cannot answer "has the null rate been
climbing for a week", and that is the question that catches an upstream change while it is
still a warning. So each constraint's result goes onto the same event bus every other
subsystem reports to, and `observe` folds it into the counters a scrape already reads.

These tests pin the fold rather than the publisher: the counters, the per-constraint
breakdown, the bound on its cardinality, and the label escaping — because a constraint name
carries a regex, and one unescaped quote makes the whole Prometheus exposition unparseable
rather than just that line.
"""

from __future__ import annotations

import pytest

from batcher._internal import events
from batcher.observe.metrics import (
    metrics_snapshot,
    prometheus_text,
    reset_metrics,
    start_metrics,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_counters():
    start_metrics()
    reset_metrics()
    yield
    reset_metrics()


def _publish(name: str, *, violations: int, ok: bool, rows: int = 100, **extra) -> None:
    events.publish(
        events.DQ,
        name=name,
        constraint=name,
        check=extra.pop("check", "row"),
        severity=extra.pop("severity", "error"),
        violations=violations,
        rows=rows,
        ok=ok,
        **extra,
    )


def test_a_passing_check_is_still_counted():
    """A series that only appears when something breaks has no baseline to compare against."""
    _publish("not_null(id)", violations=0, ok=True)
    dq = metrics_snapshot()["data_quality"]
    assert dq["checks_total"] == 1
    assert dq["failed_total"] == 0
    assert dq["violations_total"] == 0
    assert dq["by_constraint"]["not_null(id)"] == {"checks": 1, "failed": 0, "violations": 0}


def test_violations_and_failures_are_separate_counters():
    """One failed check can carry a million violating rows; the two answer different questions."""
    _publish("positive(amount)", violations=1_000_000, ok=False)
    _publish("not_null(id)", violations=3, ok=True)  # inside its `mostly` tolerance
    dq = metrics_snapshot()["data_quality"]
    assert dq["checks_total"] == 2
    assert dq["failed_total"] == 1
    assert dq["violations_total"] == 1_000_003


def test_counters_accumulate_across_runs():
    for _ in range(3):
        _publish("not_null(id)", violations=2, ok=False)
    dq = metrics_snapshot()["data_quality"]
    assert dq["by_constraint"]["not_null(id)"] == {"checks": 3, "failed": 3, "violations": 6}


def test_constraint_cardinality_is_bounded():
    """A `check(name=...)` built from a row value must not grow the map without bound."""
    for i in range(300):
        _publish(f"check(row {i})", violations=1, ok=False)
    dq = metrics_snapshot()["data_quality"]
    assert len(dq["by_constraint"]) <= 257  # 256 distinct names plus the "other" fold
    assert "other" in dq["by_constraint"]
    # Nothing is lost from the totals, only from the breakdown.
    assert dq["violations_total"] == 300


def test_prometheus_exports_the_totals():
    _publish("positive(amount)", violations=4, ok=False)
    text = prometheus_text()
    assert "batcher_dq_checks_total 1" in text
    assert "batcher_dq_failed_total 1" in text
    assert "batcher_dq_violations_total 4" in text
    assert 'batcher_dq_constraint_violations_total{constraint="positive(amount)"} 4' in text


def test_prometheus_escapes_a_regex_constraint_name():
    """A pattern carries quotes and backslashes, which the exposition format reserves."""
    _publish(r'matches(sku, "^[A-Z]\d$")', violations=1, ok=False)
    line = next(
        line
        for line in prometheus_text().splitlines()
        if line.startswith("batcher_dq_constraint_violations_total{")
    )
    assert line.count('"') % 2 == 0, line
    assert r"\"" in line and r"\\" in line
    assert line.endswith("} 1")


def test_relation_level_results_carry_their_measured_value():
    _publish("mean_between(x, 1, 2)", violations=1, ok=False, check="aggregate", rows=0, value=9.5)
    dq = metrics_snapshot()["data_quality"]
    assert dq["by_constraint"]["mean_between(x, 1, 2)"]["violations"] == 1


def test_the_section_is_present_even_when_nothing_ran():
    dq = metrics_snapshot()["data_quality"]
    assert dq == {"checks_total": 0, "failed_total": 0, "violations_total": 0, "by_constraint": {}}
