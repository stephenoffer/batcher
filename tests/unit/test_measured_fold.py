"""Two measured quantities read one hub, and neither consumes the other's evidence.

`measured_selectivity` and `measured_width` fold the same feedback history through the same
incremental machinery in `kyber.measured_fold`: each keeps a cursor into
`MetadataHub.signed_appends` and absorbs only the rows appended since it last looked. That is
what keeps the fold off the critical path of every `optimize`, and it is also the one thing a
single shared cursor would silently break — the first reader would advance it, the second
would see nothing fresh, and would report an empty result forever while every row it needed
sat in the hub. Nothing else about it would look wrong.

So the fold state is keyed by quantity as well as by hub, and this is the test that says so.
"""

from __future__ import annotations

import pytest

from batcher.config import active_config
from batcher.kyber.measured_selectivity import measured_selectivities
from batcher.kyber.measured_width import measured_widths
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.plan.feedback import OperatorFeedback, OpId

pytestmark = pytest.mark.unit


def _filter_rows(sig: str, *, rows_out: int, result_bytes: int, selectivity: float) -> MetadataHub:
    """A hub holding enough filter observations of one shape to clear the evidence gate."""
    hub = MetadataHub(InProcessBackend())
    for i in range(max(3, active_config().optimizer.cardinality_correction_min_samples)):
        hub.record(
            OperatorFeedback(
                op_id=OpId(i),
                kind="filter",
                n_actual=rows_out,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=selectivity,
                batch_size=1,
                n_input=rows_out,
                signature=sig,
                n_estimated=float(rows_out),
                result_bytes=result_bytes,
            )
        )
    return hub


def test_each_quantity_folds_the_shared_history_for_itself():
    """Both readers see the same rows, whichever of them reads first."""
    hub = _filter_rows("sig", rows_out=100, result_bytes=100 * 512, selectivity=0.25)
    assert measured_widths(hub)["sig"] == pytest.approx(512.0)
    assert measured_selectivities(hub)["sig"] == pytest.approx(0.25)
    assert measured_widths(hub)["sig"] == pytest.approx(512.0), "re-reading must be stable"


def test_a_reader_ignores_the_rows_that_are_not_its_own():
    """An aggregate measures a width and no selectivity, and the width must survive that."""
    hub = _filter_rows("sig", rows_out=100, result_bytes=100 * 512, selectivity=0.25)
    for i in range(max(3, active_config().optimizer.cardinality_correction_min_samples)):
        hub.record(
            OperatorFeedback(
                op_id=OpId(100 + i),
                kind="aggregate",
                n_actual=10,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=1.0,
                batch_size=1,
                signature="agg",
                result_bytes=10 * 512,
            )
        )
    assert measured_selectivities(hub).keys() == {"sig"}
    assert measured_widths(hub).keys() == {"sig", "agg"}


def _record(hub, sig, *, kind="filter", rows_out=100, result_bytes=None, selectivity=0.25, op=0):
    hub.record(
        OperatorFeedback(
            op_id=OpId(op),
            kind=kind,
            n_actual=rows_out,
            t_op_ms=1.0,
            m_peak_bytes=0,
            selectivity=selectivity,
            batch_size=1,
            n_input=rows_out,
            signature=sig,
            n_estimated=float(rows_out),
            result_bytes=rows_out * 512 if result_bytes is None else result_bytes,
        )
    )


def test_only_the_signatures_that_moved_are_re_summarized():
    """Incremental by shape, not just by row — and the answer is the full rebuild's.

    The fold absorbed rows incrementally but rebuilt its *summary* over every signature it
    had ever seen, computing a median apiece, on every read. That put a cost proportional to
    the session's cumulative history back on the critical path of each `optimize` — TPC-DS
    at scale 1 measured a probe query at 9.4 ms cold and 21.5 ms after 100 other queries in
    the same session, most of it here and in the correction fold next door.

    What must not change is the answer, so this drives the incremental path (many shapes,
    then one more observation of one of them) and holds it against a fold that has seen
    exactly the same history in one go.
    """
    gate = max(3, active_config().optimizer.cardinality_correction_min_samples)
    incremental = MetadataHub(InProcessBackend())
    for shape in range(20):
        for i in range(gate):
            _record(incremental, f"s{shape}", selectivity=0.25, op=shape * 100 + i)
    measured_selectivities(incremental)  # summarize, then move one shape only
    for i in range(gate):
        _record(incremental, "s7", selectivity=0.75, op=9000 + i)

    whole = MetadataHub(InProcessBackend())
    for shape in range(20):
        for i in range(gate):
            _record(whole, f"s{shape}", selectivity=0.25, op=shape * 100 + i)
    for i in range(gate):
        _record(whole, "s7", selectivity=0.75, op=9000 + i)

    assert measured_selectivities(incremental) == measured_selectivities(whole)


def test_a_shape_whose_samples_spread_out_stops_reporting():
    """An entry is *removed* when its window stops being concentrated, not left stale.

    The rebuild dropped a disqualified signature implicitly by not re-emitting it; an
    incremental update has to delete it on purpose, and a mean nobody should trust is
    exactly the kind of value that would otherwise persist forever.
    """
    gate = max(3, active_config().optimizer.cardinality_correction_min_samples)
    window = active_config().optimizer.cardinality_correction_window
    hub = MetadataHub(InProcessBackend())
    for i in range(gate):
        _record(hub, "sig", selectivity=0.25, op=i)
    assert "sig" in measured_selectivities(hub)
    # Fill the whole window with wildly spread values, so the concentration gate refuses it.
    for i in range(window):
        _record(hub, "sig", selectivity=0.01 if i % 2 else 0.99, op=1000 + i)
    assert "sig" not in measured_selectivities(hub)
