"""The per-signature priors describe data, so they must not be split per machine class.

`metadata.hardware_scope` draws the line at what a stored value *describes*: scope a
machine-unit measurement, and never scope a statement about data, because scoping those
"would fragment the statistics that took the most work to collect, turning a well-calibrated
fleet into N poorly-calibrated ones for no gain".

All three priors here are on the data side of that line — a join's two input row counts, a
breaker's shuffled row count, an aggregate's `groups / input_rows` ratio — and every one is
recorded on the *driver* from the whole query's figures, never per shard. So the same query
planned single-node and then distributed was writing two entries for identical data, and
neither run could inform the other.
"""

from __future__ import annotations

import pytest

from batcher.kyber.learned_tuning.priors import (
    learned_build_sides,
    learned_partial_agg,
    learned_partition_count,
    record_group_reduction,
    record_join_sides,
    record_partition_rows,
)
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hardware_scope import planning_for

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def test_measured_join_sides_are_visible_from_another_machine_class() -> None:
    hub = _hub()
    record_join_sides(hub, "sig", 5_000_000.0, 400.0)
    assert learned_build_sides(hub, "sig") == (5_000_000.0, 400.0)
    with planning_for("worker-class"):
        assert learned_build_sides(hub, "sig") == (5_000_000.0, 400.0), (
            "a relation has the same number of rows whichever machine counts them"
        )


def test_measured_shuffle_rows_are_visible_from_another_machine_class() -> None:
    hub = _hub()
    for _ in range(8):  # clear the reader's confidence gate
        record_partition_rows(hub, "sig", 8_000_000.0)
    local = learned_partition_count(hub, "sig", target_rows=1_000_000)
    assert local is not None, "the writer's own reader must see it"
    with planning_for("worker-class"):
        assert learned_partition_count(hub, "sig", target_rows=1_000_000) == local


def test_a_measured_group_reduction_is_visible_from_another_machine_class() -> None:
    hub = _hub()
    for _ in range(8):  # clear whatever confidence gate the reader applies
        record_group_reduction(hub, "sig", groups=10.0, input_rows=1_000_000.0)
    local = learned_partial_agg(hub, "sig")
    with planning_for("worker-class"):
        assert learned_partial_agg(hub, "sig") == local
