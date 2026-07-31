"""The straggler term: a partitioned window on a skewed column is not a balanced shuffle.

A window's frame spans a whole partition, so a hot partition cannot be salted the way a join's
hot key can, nor pre-reduced the way an aggregate's can. It lands whole on one worker and the
stage waits for it. Charged by volume alone, that plan was priced identically to a balanced one.

The guarantee under test is the same one every other addition here carries: an unmeasured
column, a single worker, and the operators that *can* rebalance must all be priced exactly as
they were.
"""

from __future__ import annotations

import pytest

from batcher.kyber.cost.imbalance import MAX_IMBALANCE, partition_imbalance

pytestmark = pytest.mark.unit


def test_a_balanced_key_costs_nothing_extra():
    """A key spread over many values is what the volume model already assumes."""
    assert partition_imbalance([{"a": 0.01, "b": 0.01}], 32) == 1.0


def test_an_unmeasured_key_is_not_assumed_skewed():
    """No statistics is not evidence of skew; a cold store must rank plans as before."""
    assert partition_imbalance([{}], 32) == 1.0
    assert partition_imbalance([], 32) == 1.0


def test_a_hot_value_costs_its_share_of_the_fleet():
    """One value holding 47% of the rows makes one reducer do 47% of the work."""
    assert partition_imbalance([{"hot": 0.47}], 32) == pytest.approx(0.47 * 32)


def test_the_multiplier_is_bounded():
    """One approximate statistic must not be allowed to dominate a whole plan total."""
    assert partition_imbalance([{"hot": 0.99}], 1024) == MAX_IMBALANCE


def test_a_value_at_its_fair_share_is_not_skew():
    """`f == 1/W` is exactly balanced, and the multiplier is continuous through it."""
    assert partition_imbalance([{"hot": 1 / 8}], 8) == 1.0
    assert partition_imbalance([{"hot": 0.2}], 8) == pytest.approx(1.6)


def test_a_single_worker_has_no_partitions_to_imbalance():
    assert partition_imbalance([{"hot": 0.9}], 1) == 1.0


def test_a_compound_key_is_rescued_by_its_balanced_component():
    """`PARTITION BY hot_country, user_id` splits finely even though `country` does not.

    Taking the minimum across keys is what stops a compound partitioning from being charged
    for skew that its other column already removes.
    """
    assert partition_imbalance([{"US": 0.9}, {"u1": 0.001}], 32) == 1.0


def test_a_mildly_uneven_key_is_below_the_hot_floor():
    """Without a floor every measured column would carry some multiplier, making it a
    constant rather than a signal."""
    assert partition_imbalance([{"top": 0.09}], 4) == 1.0


def _window_plan():
    """`sum(v) OVER (PARTITION BY k)` — the one shuffling shape that cannot rebalance."""
    import batcher as bt
    from batcher import col

    frame = bt.from_pydict({"k": [f"c{i % 50}" for i in range(1000)], "v": list(range(1000))})
    return frame, frame.with_columns(r=col("v").sum().over("k"))._plan


def _stats_with(mcv: dict):
    """A `stats_of` stand-in reporting `mcv` for every column.

    A stub rather than real data on purpose: an in-memory source records no most-common
    values, so a frame built from a Python dict cannot exercise this path at all and a test
    written against one would pass whatever the term did.
    """

    class _Column:
        def __init__(self) -> None:
            self.mcv = mcv

    class _Stats:
        def column(self, _name: str) -> _Column:
            return _Column()

    return lambda _node: _Stats()


def test_the_term_reaches_net_cost_for_a_window():
    """A measured hot partition multiplies the window's `net` cost; an unmeasured one does not."""
    from batcher.kyber.cost.shuffle import net_cost

    _frame, plan = _window_plan()
    rows_of, width_of = (lambda _n: 1000.0), (lambda _n: 64.0)
    balanced = net_cost(plan, rows_of, width_of, 32, 1.0, _stats_with({}))
    skewed = net_cost(plan, rows_of, width_of, 32, 1.0, _stats_with({"hot": 0.5}))
    assert balanced > 0.0
    assert skewed == pytest.approx(balanced * 16.0)


def test_an_aggregate_is_never_charged_for_skew():
    """It pre-reduces the hot key to one partial row per worker before shuffling anything.

    Charging it would push the optimizer away from the one operator shape that already solves
    the problem.
    """
    import batcher as bt
    from batcher import col
    from batcher.kyber.cost.shuffle import net_cost

    frame = bt.from_pydict({"k": [f"c{i % 50}" for i in range(1000)], "v": list(range(1000))})
    plan = frame.group_by("k").agg(total=col("v").sum())._plan
    rows_of, width_of = (lambda _n: 1000.0), (lambda _n: 64.0)
    assert net_cost(plan, rows_of, width_of, 32, 1.0, _stats_with({"hot": 0.9})) == pytest.approx(
        net_cost(plan, rows_of, width_of, 32, 1.0, _stats_with({}))
    )


def test_no_stats_callable_reproduces_the_previous_cost():
    """Every caller that passes none — which is every caller before this — is unchanged."""
    from batcher.kyber.cost.shuffle import net_cost

    _frame, plan = _window_plan()
    rows_of, width_of = (lambda _n: 1000.0), (lambda _n: 64.0)
    assert net_cost(plan, rows_of, width_of, 32) == pytest.approx(
        net_cost(plan, rows_of, width_of, 32, 1.0, _stats_with({"hot": 0.9})) / 16.0
    )
