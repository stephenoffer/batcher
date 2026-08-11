"""A breaker is fanned out by what it *holds*, not only by what flows into it.

`_desired_parallelism` sized a stage from its input: rows, and rows times width. Both describe
the data flowing **in**, and for a hash operator the quantity that OOMs is the state it holds.

A `GROUP BY user_id` over a narrow two-column input is the case: small by rows, smaller by
input bytes, and resident in one entry per distinct user. The input is exactly as wide and as
numerous as a two-group aggregate's, so neither existing term can tell them apart, and the
hundred-million-group version got one task.

The constraint on the fix is the one the byte term already holds to: every term may only ask
for *more* parallelism, so nothing with no measurable state can be re-shaped by this.
"""

from __future__ import annotations

import pytest

from batcher.config import active_config
from batcher.kyber.annotate import _desired_parallelism, _task_memory_budget
from batcher.plan.resource import HardwareProfile

pytestmark = pytest.mark.unit

_GIB = 1 << 30
_TARGET_ROWS = 4_000_000
_TARGET_BYTES = 256 << 20


def _fanout(**kwargs) -> int:
    """`_desired_parallelism` over a deliberately unremarkable input volume."""
    args = {
        "in_rows": 1000.0,
        "width": 16.0,
        "target_rows": _TARGET_ROWS,
        "target_bytes": _TARGET_BYTES,
    }
    args.update(kwargs)
    return _desired_parallelism(
        args["in_rows"],
        args["width"],
        args["target_rows"],
        args["target_bytes"],
        state_bytes=args.get("state_bytes", 0.0),
        task_memory_bytes=args.get("task_memory_bytes", 0.0),
    )


def test_no_state_figure_is_the_fan_out_it_always_was():
    """Every caller before this passed neither argument, and must be unmoved."""
    assert _fanout() == 1
    assert _fanout(in_rows=40_000_000.0) == 10


def test_no_hardware_profile_contributes_no_demand():
    """A state size with nothing to compare it against says nothing."""
    assert _fanout(state_bytes=100 * _GIB) == 1


def test_a_state_larger_than_one_task_asks_for_more_tasks():
    """The defect, stated as the case it produces.

    A hundred gibibytes of hash table on a node that admits eight is not one task's work,
    however few rows went in.
    """
    assert _fanout(state_bytes=100.0 * _GIB, task_memory_bytes=8.0 * _GIB) == 13


def test_a_state_that_fits_asks_for_nothing():
    """The term is a floor, not a rescaling: a resident state adds no demand."""
    assert _fanout(state_bytes=1.0 * _GIB, task_memory_bytes=8.0 * _GIB) == 1


def test_the_state_term_can_only_raise_the_fan_out():
    """`max`, never `min` — a small state must not shrink a fan-out the rows demanded."""
    by_rows = _fanout(in_rows=40_000_000.0)
    assert by_rows > 1
    assert _fanout(in_rows=40_000_000.0, state_bytes=1.0, task_memory_bytes=64.0 * _GIB) == by_rows


def test_two_aggregates_with_identical_inputs_are_told_apart():
    """The pair the input-derived terms cannot distinguish, which is the whole point."""
    narrow_input = {"in_rows": 100_000.0, "width": 16.0, "task_memory_bytes": 4.0 * _GIB}
    two_groups = _fanout(state_bytes=64.0, **narrow_input)
    hundred_million_groups = _fanout(state_bytes=80.0 * _GIB, **narrow_input)
    assert two_groups == 1
    assert hundred_million_groups > two_groups


def test_the_budget_is_the_one_carbonite_admits_against():
    """Kyber's fan-out and Carbonite's admission must be sized against one number."""
    cfg = active_config()
    hardware = HardwareProfile(memory_bytes=64 * _GIB)
    assert _task_memory_budget(hardware, cfg) == pytest.approx(64 * _GIB * cfg.memory.hard_limit)


def test_an_unknown_worker_memory_reports_no_opinion():
    """`0` is "no profile", and must not read as a zero-byte envelope (which would divide)."""
    assert _task_memory_budget(HardwareProfile(), active_config()) == 0.0
    assert _task_memory_budget(None, active_config()) == 0.0


def test_the_annotation_carries_the_raised_fan_out():
    """End to end, through `annotate_ops`, since that is where the term is consumed."""
    import batcher as bt
    from batcher import col
    from batcher.kyber.annotate import annotate_ops
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel

    frame = bt.from_pydict({"k": list(range(2_000)), "v": list(range(2_000))})
    plan = frame.group_by("k").agg(t=col("v").sum())._plan
    estimator = CardinalityEstimator(frame._sources)
    cfg = active_config()
    model = CostModel(estimator)

    def fanout_for(memory_bytes: int) -> int:
        ops = annotate_ops(
            plan, estimator, cfg, model, None, HardwareProfile(memory_bytes=memory_bytes)
        )
        return max(op.bounds.n_max_parallelism for op in ops)

    # A node that can hold the whole aggregate against one that can hold a sliver of it.
    assert fanout_for(64 * _GIB) <= fanout_for(4096)
