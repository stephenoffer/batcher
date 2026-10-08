"""The staging gate asks its "is anything left to correct?" question of the plan that ran.

`api.adaptive.gating.note_one_shot_ops` records, under the plan the router asked about, which
stageable join operands the executed plan had to guess and the row count it planned each on --
read off its own operator annotations, under the signatures Core reports measurements by.
`_ran_and_held_up` answers from that record against the rows each operand measured. Pinned
here: nothing is recorded for a query the gate never reached; only a breaker-produced join
operand carrying a default-guess estimate is recorded, matched on the node-class kind names
`kyber.annotate` writes; a run that guessed nothing stops the gate; a guess stops it only once
the planned rows sit within `reoptimize_error` of the measured ones; and `resolve_adaptive`
clears the in-flight key so one query's run is never filed under another's plan.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.api.adaptive import gating
from batcher.kyber import learning
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds
from batcher.plan.stats import Provenance


@pytest.fixture(autouse=True)
def _fresh():
    gating._RAN_GUESSED.clear()
    token = gating._ROUTED.set(None)
    yield
    gating._RAN_GUESSED.clear()
    gating._ROUTED.reset(token)


def _op(i, kind, inputs=(), *, materializes=False, provenance=Provenance.EXACT, sig="", rows=0.0):
    return PhysicalOp(
        op_id=OpId(i),
        kind=kind,
        backend="native",
        algorithm="",
        bounds=ResourceBounds(0, 0, 0, materializes=materializes),
        inputs=tuple(OpId(x) for x in inputs),
        properties=PlanProperties(est_rows=rows, provenance=provenance, signature=sig),
    )


def _plan(guess: Provenance, rows: float = 1000.0) -> PhysicalPlan:
    ops = (
        _op(0, "Join", (1, 2), materializes=True),
        _op(1, "Aggregate", (3,), materializes=True, provenance=guess, sig="agg-sig", rows=rows),
        _op(2, "Scan", provenance=Provenance.DEFAULT, sig="scan-sig"),  # streamed: never staged
        _op(3, "Scan", sig="scan2"),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=ops)


def test_nothing_is_recorded_for_a_query_the_gate_never_reached():
    gating.note_one_shot_ops(_plan(Provenance.DEFAULT))
    assert not gating._RAN_GUESSED


def test_only_a_guessed_breaker_join_operand_is_recorded():
    gating._ROUTED.set("k")
    gating.note_one_shot_ops(_plan(Provenance.DEFAULT))
    assert gating._RAN_GUESSED == {"k": (("agg-sig", 1000.0),)}


def test_an_operand_with_a_breaker_beneath_it_is_recorded():
    """A projection over an aggregate feeds the join: the operand is breaker-produced."""
    ops = (
        _op(0, "Join", (1, 3), materializes=True),
        _op(1, "Project", (2,), provenance=Provenance.DEFAULT, sig="proj-sig", rows=5.0),
        _op(2, "Aggregate", (3,), materializes=True, sig="agg-sig"),
        _op(3, "Scan", sig="scan"),
    )
    gating._ROUTED.set("k")
    gating.note_one_shot_ops(PhysicalPlan(ir={}, output_schema=None, ops=ops))
    assert gating._RAN_GUESSED == {"k": (("proj-sig", 5.0),)}


def test_a_run_that_guessed_nothing_stops_the_gate():
    gating._ROUTED.set("k")
    gating.note_one_shot_ops(_plan(Provenance.EXACT))
    assert gating._RAN_GUESSED == {"k": ()}
    assert gating._ran_and_held_up("k", hub=object())


@pytest.mark.parametrize(
    ("measured", "stops"),
    [({}, False), ({"agg-sig": 5000.0}, False), ({"agg-sig": 1500.0}, True)],
    ids=["unmeasured", "off-by-5x", "within-tolerance"],
)
def test_a_guess_stops_the_gate_only_once_its_planned_rows_held_up(monkeypatch, measured, stops):
    gating._ROUTED.set("k")
    gating.note_one_shot_ops(_plan(Provenance.DEFAULT))
    monkeypatch.setattr(learning, "measured_rows", lambda hub: measured)
    assert gating._ran_and_held_up("k", hub=object()) is stops
    assert not gating._ran_and_held_up("other", hub=object()), "an unrecorded plan is not stopped"


def test_resolve_clears_the_in_flight_key():
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    gating._ROUTED.set("stale")
    gating.resolve_adaptive(False, Scan(0, SchemaRef.from_arrow(pa.schema([]))), [], None)
    assert gating._ROUTED.get() is None
