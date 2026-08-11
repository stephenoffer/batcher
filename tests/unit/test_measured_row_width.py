"""An intermediate's byte width, measured rather than inferred.

`StatsEstimator.row_width` answers a **scan** well: learned per-column averages, Arrow type
priors, a connector's measured `byte_size`, and the cheap per-column reading `_learn_row_bytes`
takes. It answers an **intermediate** badly, because the output width of a join or an aggregate is
re-derived by summing per-column priors through every operator that reshapes the row, and the
error compounds with depth.

`cost/model.py` declines to charge for width at all because of it, and says what would change its
mind: "the width *estimate* is not yet good enough to rank on; it belongs here once intermediate
widths are measured rather than inferred". The engine reports `result_bytes` next to the rows each
operator emitted, on every execution, under the same stable signature the cardinality loop is
keyed by. Their ratio is that measurement.

**This module shipped once and was reverted.** It was correct in everything but its key: a
signature is structural, and `plan_signature` rendered every scan as the bare token `["scan"]`, so
a 4 KiB table's width was applied to a 16-byte one. The scan token now carries the source's
data-stable identity, which is what makes the measurement safe to consume — and the first test
below is exactly the case that forced the revert.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
import batcher.core as core
from batcher.config import active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.learning import load_learned_stats
from batcher.kyber.measured_width import measured_widths
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.plan.feedback import OperatorFeedback, OpId

pytestmark = pytest.mark.unit

_ROWS = 4_000


@pytest.fixture
def two_files(tmp_path):
    """A wide-payload table and a narrow one, same query shape over each."""
    wide, narrow = tmp_path / "wide.parquet", tmp_path / "narrow.parquet"
    pq.write_table(pa.table({"k": list(range(_ROWS)), "doc": ["x" * 4096] * _ROWS}), wide)
    pq.write_table(
        pa.table({"k": list(range(_ROWS)), "m": [float(i) for i in range(_ROWS)]}), narrow
    )
    return str(wide), str(narrow)


def _record(hub, sig, *, rows_out, result_bytes, count=None, kind="aggregate"):
    """`count` observations of a shape emitting `rows_out` rows in `result_bytes` bytes."""
    if count is None:
        count = max(3, active_config().optimizer.cardinality_correction_min_samples)
    for i in range(count):
        hub.record(
            OperatorFeedback(
                op_id=OpId(i),
                kind=kind,
                n_actual=rows_out,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=1.0,
                batch_size=1,
                n_input=rows_out,
                signature=sig,
                n_estimated=float(rows_out),
                result_bytes=result_bytes,
            )
        )
    return hub


def test_one_tables_width_does_not_reach_another(two_files):
    """The case that forced this module's revert, now covered by the scan-token identity."""
    wide, narrow = two_files
    hub = core.default_hub()
    other = bt.read_parquet(narrow).filter(bt.col("k") < _ROWS - 1)
    cold = CardinalityEstimator(other._sources).row_width(other._plan, 64.0)

    measured = bt.read_parquet(wide).filter(bt.col("k") < _ROWS - 1)
    for _ in range(4):
        measured.collect()

    warm = CardinalityEstimator(other._sources, load_learned_stats(hub))
    assert warm.row_width(other._plan, 64.0) == pytest.approx(cold)
    assert cold == pytest.approx(16.0), "two int64 columns"


def test_the_measured_width_reaches_the_shape_that_earned_it(two_files):
    """And the measurement is worth having where the key is right."""
    wide, _ = two_files
    hub = core.default_hub()
    query = bt.read_parquet(wide).filter(bt.col("k") < _ROWS - 1)
    for _ in range(4):
        query.collect()
    warm = CardinalityEstimator(query._sources, load_learned_stats(hub))
    # 4 KiB payload plus an int64 key and the string's offset.
    assert warm.row_width(query._plan, 64.0) == pytest.approx(4104.0, rel=0.01)


def test_a_measured_width_is_the_ratio_of_bytes_to_rows():
    hub = _record(MetadataHub(InProcessBackend()), "sig", rows_out=100, result_bytes=100 * 512)
    assert measured_widths(hub)["sig"] == pytest.approx(512.0)


def test_one_observation_is_not_enough():
    """The same evidence gate the cardinality-correction loop applies, for the same reason."""
    hub = _record(MetadataHub(InProcessBackend()), "sig", rows_out=10, result_bytes=100, count=1)
    assert measured_widths(hub) == {}


def test_inconsistent_samples_are_refused():
    """The residual: an in-memory relation contributes no identity, so shapes can still share.

    Two populations under one key average to a width that describes neither, and a wide spread
    is what that looks like. The same gate `measured_selectivity` and `cpu_shares` apply.
    """
    hub = MetadataHub(InProcessBackend())
    _record(hub, "sig", rows_out=100, result_bytes=100 * 16)
    _record(hub, "sig", rows_out=100, result_bytes=100 * 4096)
    assert "sig" not in measured_widths(hub)


def test_an_unreported_result_size_is_absent_evidence_not_zero():
    """`result_bytes == 0` is an engine that did not report the field.

    Treating it as a measured zero would drive a shape's width to zero and make every byte axis
    over it collapse — a broadcast of an arbitrarily large relation would look free.
    """
    hub = _record(MetadataHub(InProcessBackend()), "sig", rows_out=100, result_bytes=0)
    assert measured_widths(hub) == {}


def test_an_operator_that_emitted_nothing_contributes_nothing():
    """No rows means no width to measure, and a division to avoid."""
    hub = _record(MetadataHub(InProcessBackend()), "sig", rows_out=0, result_bytes=4096)
    assert measured_widths(hub) == {}


def test_an_incredible_width_is_refused():
    """A mis-attributed row would otherwise dominate the mean for its signature."""
    hub = _record(MetadataHub(InProcessBackend()), "sig", rows_out=1, result_bytes=1 << 40)
    assert measured_widths(hub) == {}


def test_a_measurement_can_only_raise_the_width():
    """`max`, never substitution, and the asymmetry is the point.

    `result_bytes` is the *result array's* bytes, so a dictionary-encoded or sliced output can
    measure below what the same rows occupy downstream. Substituting it there would under-size a
    memory envelope and a task fan-out — the failure that OOMs at cluster scale — while
    over-stating a width merely forfeits a broadcast.
    """
    frame = bt.from_pydict({"k": list(range(100)), "v": list(range(100))})
    plan = frame.group_by("k").agg(t=bt.col("v").sum())._plan
    from batcher.kyber.signature import plan_signature

    hub = _record(
        MetadataHub(InProcessBackend()), plan_signature(plan), rows_out=100, result_bytes=200
    )
    cold = CardinalityEstimator(frame._sources)
    warm = CardinalityEstimator(frame._sources, load_learned_stats(hub))
    assert warm.row_width(plan, 64.0) == cold.row_width(plan, 64.0) == pytest.approx(16.0)


def test_a_width_is_not_scoped_to_the_machine_that_measured_it():
    """A row's byte width is a property of the data, not of the hardware.

    The counterpart rule to `metadata.hardware_scope`: times and capacities are machine-scoped,
    statements about data are not. A width learned on one worker is true on every other.
    """
    hub = MetadataHub(InProcessBackend())
    for i in range(max(3, active_config().optimizer.cardinality_correction_min_samples)):
        hub.record(
            OperatorFeedback(
                op_id=OpId(i),
                kind="aggregate",
                n_actual=10,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=1.0,
                batch_size=1,
                signature="sig",
                result_bytes=10 * 256,
                hw_fingerprint="ffffffffffff",  # measured on some other machine class
            )
        )
    assert measured_widths(hub)["sig"] == pytest.approx(256.0)
