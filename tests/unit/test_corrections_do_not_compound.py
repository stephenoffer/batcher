"""A learned correction scales the *uncorrected* estimate, so chained corrections do not compound.

Each factor is measured against the estimate with no correction anywhere beneath it
(`StatsEstimator.reportable_estimate`). Applying it to an estimate built from inputs that are
already corrected counts every correction below twice: TPC-DS q68's four chained joins each
learned ~14x, compounded to ~38,000x, and a 25,000-row join was admitted as 72 GB and spilled.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.column_tables import CARDINALITY_CORRECTION_KEY
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Join

pytestmark = pytest.mark.unit


def _plan():
    a = bt.from_arrow(pa.table({"k": list(range(1000)), "x": list(range(1000))}))
    b = bt.from_arrow(pa.table({"k": list(range(1000)), "y": list(range(1000))}))
    c = bt.from_arrow(pa.table({"k": list(range(1000)), "z": list(range(1000))}))
    return a.join(b, on="k").join(c, on="k")


def test_a_parent_correction_lands_on_its_own_measured_size():
    ds = _plan()
    plan, sources = ds._plan, ds._sources
    top = plan
    while not isinstance(top, Join):
        top = top.input
    child = top.left if isinstance(top.left, Join) else top.right
    assert isinstance(child, Join)

    base = StatsEstimator(sources)
    raw_top = base.estimate(top).rows
    sig_top, sig_child = base.signature_of(top), base.signature_of(child)

    corrected = StatsEstimator(
        sources, {CARDINALITY_CORRECTION_KEY: {sig_top: 10.0, sig_child: 10.0}}
    )
    # Positive control: the child's own correction is applied.
    assert corrected.estimate(child).rows == pytest.approx(base.estimate(child).rows * 10.0)
    # The parent lands on raw x its own factor, not raw x 10 x 10.
    assert corrected.estimate(top).rows == pytest.approx(raw_top * 10.0)
    # And the recorded baseline stays the uncorrected estimate.
    assert corrected.reportable_estimate(top) == pytest.approx(raw_top)
