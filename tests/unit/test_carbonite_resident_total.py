"""The input and the plan's peak state are concurrent, so what matters is their sum.

`input_exceeds_budget` and `should_spill` are two halves of one total, and each was compared
against the whole envelope on its own. Nothing summed them -- yet on the in-memory path they
coexist: the sources are resolved to Arrow batches before the engine starts and stay resident
for the whole execution, while the breaker builds its state on top of them.

So a query whose input is 70% of the envelope and whose breaker is 70% of it passes both checks
and needs 140%. Measured on a 24 M-row group-by under a 537 MB envelope: input 384 MB, live
partial state 384 MB, neither over the budget alone, both over it together -- and the query
stayed in memory and peaked at 2.4 GB. With the sum checked it routes out of core and peaks at
1.15 GB.
"""

from __future__ import annotations

import pytest

from batcher.carbonite import ResourceManager
from batcher.config import Config, MemoryConfig, config_context
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds

pytestmark = pytest.mark.unit

ENVELOPE = 1_000_000


def _plan(peak_bytes: int) -> PhysicalPlan:
    """A one-operator plan whose breaker is annotated at `peak_bytes`."""
    op = PhysicalOp(
        op_id=OpId(0),
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=peak_bytes, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
        properties=PlanProperties(est_rows=float(peak_bytes)),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


def _manager() -> ResourceManager:
    return ResourceManager()


@pytest.fixture
def envelope():
    """A pinned envelope, so the sensed machine size cannot decide these assertions."""
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=ENVELOPE))
    with config_context(cfg):
        yield ENVELOPE


def test_two_terms_that_each_fit_but_do_not_fit_together(envelope) -> None:
    """The case both existing checks miss: 70% + 70%."""
    rm = _manager()
    seventy = int(envelope * 0.7)
    plan = _plan(seventy)
    assert rm.input_exceeds_budget(seventy) is False, "the input alone fits, by construction"
    assert rm.should_spill(plan) is False, "the plan alone fits, by construction"
    assert rm.resident_total_exceeds_budget(seventy, plan) is True, (
        "the input and the breaker's state coexist on the in-memory path, so a query needing "
        "140% of the envelope must route out of core"
    )


def test_a_query_that_genuinely_fits_is_left_alone(envelope) -> None:
    """Over-routing costs latency on every small query, so the sum must not be trigger-happy."""
    rm = _manager()
    small = int(envelope * 0.2)
    assert rm.resident_total_exceeds_budget(small, _plan(small)) is False


def test_an_unsizable_input_is_no_evidence_either_way(envelope) -> None:
    """`0` means "I cannot tell", exactly as it does for `input_exceeds_budget` -- it must not
    be read as "small". The other signals (`should_spill`, live pressure) still apply."""
    rm = _manager()
    assert rm.resident_total_exceeds_budget(0, _plan(int(envelope * 0.9))) is False


def test_it_subsumes_the_input_only_check(envelope) -> None:
    """An input over the envelope on its own must still route, whatever the plan says -- so
    replacing the input-only call at the routing site loses nothing."""
    rm = _manager()
    huge = envelope * 4
    assert rm.input_exceeds_budget(huge) is True
    assert rm.resident_total_exceeds_budget(huge, _plan(0)) is True
