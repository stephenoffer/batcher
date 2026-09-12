"""The spill gate must count what the materializing path holds, not only breaker state.

`SpillAdvisor.resident_total_exceeds_budget` summed the resident input and
`peak_bytes` — the plan's dominant *breaker* state. That misses the term that actually
kills a large query. The in-memory path runs the materializing executor, which holds every
operator's full output, and Kyber sizes the row-wise operators at `m_max_bytes = 0`. That
sizing is correct — a `Filter` and a `Project` retain no state — and it is not what the
envelope needs to know, because their output is materialized before the operator above
reads it. So the plan's largest resident object is routinely one nothing in the arithmetic
mentions.

TPC-H q4 at sf100 is the shape, and the numbers are the reason this test exists. Its
`Filter` and `Project` over `lineitem` carry 292,422,301 rows and are both sized `0.00G`;
the `Join` above them is sized 2.18 GiB and is the whole of `peak_bytes`. Against a
15.65 GiB input and a 19.02 GiB budget the gate read 17.83 GiB, said "fits", and the query
was OOM-killed at 21.26 GiB. Forced out of core the same query returns its 5 rows at
**1.07 GiB**.

The asymmetry is the point: reading "fits" wrongly costs the process, and reading "spill"
wrongly costs latency on a query that was already at the edge of the envelope.
"""

from __future__ import annotations

import pytest

from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds

pytestmark = pytest.mark.unit

_GIB = 1 << 30


def _op(kind: str, est_rows: float, m_max: int = 0, op_id: int = 0) -> PhysicalOp:
    """One annotated operator: a row count, and the state Kyber thinks it holds."""
    return PhysicalOp(
        op_id=OpId(op_id),
        kind=kind,
        backend="interp",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=m_max, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
        properties=PlanProperties(est_rows=est_rows, row_size=8.0),
    )


def _q4_shaped_plan() -> PhysicalPlan:
    """TPC-H q4's annotated shape: a wide row-wise chain Kyber sizes at zero.

    The row counts are the ones Kyber actually produced at sf100, so the arithmetic under
    test is the arithmetic that failed rather than a re-scaled imitation of it.
    """
    return PhysicalPlan(
        ir={"op": "scan", "source_id": 0},
        output_schema=None,
        ops=(
            _op("Sort", 2_849_970, int(0.12 * _GIB), 0),
            _op("Aggregate", 2_849_970, int(1.17 * _GIB), 1),
            _op("Join", 28_499_699, int(2.18 * _GIB), 2),
            _op("Project", 28_499_699, 0, 3),
            _op("Filter", 28_499_699, 0, 4),
            _op("Scan", 150_000_000, 0, 5),
            # The two that matter: 292M rows apiece, sized at zero because they hold no
            # state, and materialized in full by the executor that runs them.
            _op("Project", 292_422_301, 0, 6),
            _op("Filter", 292_422_301, 0, 7),
            _op("Scan", 600_037_902, 0, 8),
        ),
    )


class _Advisor:
    """A `SpillAdvisor` with its two collaborators stubbed to fixed figures.

    Constructing a real one needs a `ResourceContext`, a `MemoryEstimator`, a learned model
    and a `PressureMonitor`; every one of them is irrelevant to the arithmetic under test,
    and the two figures that are relevant are exactly the two stubbed here.
    """

    def __init__(self, peak: int, budget: int):
        from batcher.carbonite.policies.spill_advice import SpillAdvisor

        self._advisor = SpillAdvisor.__new__(SpillAdvisor)
        self._advisor.peak_bytes = lambda _plan: peak
        self._advisor.hard_budget = lambda: budget

    def exceeds(self, input_bytes: int, plan: PhysicalPlan) -> bool:
        from batcher.carbonite.policies.spill_advice import SpillAdvisor

        return SpillAdvisor.resident_total_exceeds_budget(self._advisor, input_bytes, plan)

    def widest(self, input_bytes: int, plan: PhysicalPlan) -> int:
        from batcher.carbonite.policies.spill_advice import SpillAdvisor

        return SpillAdvisor._widest_intermediate(self._advisor, input_bytes, plan)


def test_the_q4_shape_is_routed_out_of_core():
    """The exact numbers that OOM-killed q4 must now read as "does not fit"."""
    plan = _q4_shaped_plan()
    advisor = _Advisor(peak=int(2.18 * _GIB), budget=int(19.02 * _GIB))

    # Input and peak alone are what the gate used to see, and they fit.
    assert int(15.65 * _GIB) + int(2.18 * _GIB) < int(19.02 * _GIB)
    # Counting the materialized intermediate, they do not.
    assert advisor.exceeds(int(15.65 * _GIB), plan)


def test_the_widest_intermediate_is_sized_from_the_measured_input():
    """The width comes from `input_bytes`, which already reflects pushed projections.

    `row_size` is the operator's *unprojected* row width — 292 bytes for that `Filter`
    against the ~22 bytes actually read — so sizing from it would over-read by an order of
    magnitude and push every large scan out of core. This pins the derivation, not just the
    verdict, because the two differ by 10x and only one of them is usable.
    """
    plan = _q4_shaped_plan()
    advisor = _Advisor(peak=0, budget=1)
    input_bytes = int(15.65 * _GIB)

    widest = advisor.widest(input_bytes, plan)

    scan_rows = 150_000_000 + 600_037_902
    expected = int(292_422_301 * (input_bytes / scan_rows))
    assert widest == expected
    # ~6.5 GiB: the same order as the input it came from, and nowhere near the ~85 GiB that
    # `est_rows * row_size` would have produced for the same operator.
    assert 5 * _GIB < widest < 8 * _GIB


def test_a_small_query_is_left_in_memory():
    """The guard only ever *adds* a spill, so a plan with headroom must be untouched."""
    plan = PhysicalPlan(
        ir={"op": "scan", "source_id": 0},
        output_schema=None,
        ops=(_op("Aggregate", 1_000, 1 << 20, 0), _op("Scan", 1_000_000, 0, 1)),
    )
    advisor = _Advisor(peak=1 << 20, budget=8 * _GIB)

    assert not advisor.exceeds(64 << 20, plan)


def test_a_plan_with_no_row_estimates_contributes_nothing():
    """An absent estimate must not read as a small one, nor manufacture a spill.

    Kyber emits no row count for an un-sized plan. Returning `0` here leaves the decision to
    the terms that *are* known, which is what the surrounding code already does for a `0`
    input; inventing a width from a missing row count would spill on no evidence.
    """
    plan = PhysicalPlan(ir={"op": "scan", "source_id": 0}, output_schema=None, ops=())
    advisor = _Advisor(peak=0, budget=8 * _GIB)

    assert advisor.widest(64 << 20, plan) == 0
    assert not advisor.exceeds(64 << 20, plan)
