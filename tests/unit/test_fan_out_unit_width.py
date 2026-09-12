"""The row fan-out is capped by the fleet at ONE WORKING UNIT's width, whatever that unit is.

`_widest_useful_fan_out` answers "how many units is it worth cutting the rows into", which is
`cluster cores / one unit's width`. It assumed `_TARGET_TASK_CPUS` (4), the width of a stateless
map task -- but the map/aggregate route now runs on `_agg_actor_width`-wide actors (16 on a
1,024-core fleet of 16-core nodes), so it was cutting four times more partitions than there were
units and paying per-partition dispatch, transfer and fold on each. Measured on the `udf` board,
eight sweeps against seven: median 574 -> 552 ms, best 552 -> 519, peak cluster busy 56-61% ->
80-87%.

The memory half is what makes this safe, and it is the half the earlier rejected arm got wrong.
That arm forced the count to the worker width directly and so bypassed `_byte_partition_count`,
quadrupling per-partition memory on a wide scan -- "an OOM rather than a slow query". This
changes only the *parallelism* term, which is a preference; the byte term is applied as a `max`
after it and can still raise the count as far as memory requires.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors import map as M

pytestmark = pytest.mark.unit


@pytest.fixture
def fleet(monkeypatch):
    """A 1,024-core fleet, so the arithmetic below is the cluster's rather than this box's."""
    monkeypatch.setattr(M, "_cluster_cores", lambda: 1024.0)


def test_the_default_width_is_a_stateless_task(fleet):
    """Unchanged for every caller that does not name a width: 1024 / 4."""
    assert M._widest_useful_fan_out() == 1024 // M._TARGET_TASK_CPUS == 256


def test_a_wider_unit_asks_for_proportionally_fewer(fleet):
    """A 16-thread actor is four task-widths, so it wants a quarter of the partitions."""
    assert M._widest_useful_fan_out(16) == 64
    assert M._widest_useful_fan_out(8) == 128


def test_a_width_below_one_core_cannot_explode_the_fan_out(fleet):
    """`task_cpus` is clamped at 1.0 from below: a fractional unit is still one unit."""
    assert M._widest_useful_fan_out(0.125) == 1024
    assert M._widest_useful_fan_out(0) == 1024 // M._TARGET_TASK_CPUS


def test_the_byte_bound_still_overrides_the_widened_term(fleet, monkeypatch):
    """The safety property. The parallelism term is a preference; memory is a bound.

    Without this the change would be the rejected arm in a new place -- that one forced the
    count to the worker width and so could not be raised back for memory.
    """
    monkeypatch.setattr(M, "_source_total_rows", lambda source: 10_000_000)
    monkeypatch.setattr(M, "_byte_partition_count", lambda *a, **k: 4096)

    class _Source:
        def splits(self):
            return [None] * 100_000

    n = M._adaptive_partition_count(_Source(), None, 64, task_cpus=16)

    assert n == 4096, "a memory bound of 4,096 must survive a rows term of 64"
