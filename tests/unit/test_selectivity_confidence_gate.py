"""A measured selectivity is refused when its samples are not one population.

`kyber.measured_selectivity` learns a filter's kept fraction under its structural plan
signature, and the estimator prefers it over the structural guess. That is a large win when
the signature names one thing. It does not always name one thing: `plan_signature` renders
every scan as the bare token `["scan"]`, so two filters of the same *shape* over different
relations share a single entry.

`x < 40` keeping 40 of 20,000 rows in one table and half the rows in another then averages to a
figure that predicts neither. Measured before this gate: after four runs of the selective table,
the permissive table's filter was estimated at **40 rows against an actual 20,000** — a 500x
under-estimate, and worse than the structural estimate (6,667) that the measurement replaced.
A cardinality error of that size flips a join order and a build side.

The gate does not fix the key; it bounds the damage. A wide spread is what two populations
sharing one key look like, whatever caused it, and the honest response to it is to fall back to
the structural estimate rather than to average. `cpu_shares` already applied exactly this test to
its utilization medians for the same reason, so the predicate now lives in `_internal.mathx` and
both read it.
"""

from __future__ import annotations

import pytest

import batcher as bt
import batcher.core as core
from batcher._internal.mathx import is_concentrated
from batcher.kyber.measured_fold import _MAX_REL_SPREAD
from batcher.kyber.measured_selectivity import measured_selectivities
from batcher.kyber.signature import plan_signature

pytestmark = pytest.mark.unit


def test_one_sample_is_trivially_concentrated():
    assert is_concentrated([0.5], _MAX_REL_SPREAD)


def test_constant_samples_pass():
    assert is_concentrated([0.25, 0.25, 0.25], _MAX_REL_SPREAD)


def test_a_tight_cluster_passes():
    assert is_concentrated([0.50, 0.55, 0.52, 0.48], _MAX_REL_SPREAD)


def test_two_populations_are_refused():
    """The shape of an accidental key collision."""
    assert not is_concentrated([0.002, 0.002, 1.0, 1.0], _MAX_REL_SPREAD)


def test_nothing_measured_does_not_block():
    """A non-positive median is "no evidence", and must not read as "inconsistent"."""
    assert is_concentrated([0.0, 0.0], _MAX_REL_SPREAD)


def test_the_gate_reaches_the_learning_loop():
    """End to end on the real engine: one population learns, two do not.

    The two queries genuinely share a signature — asserted, not assumed — and neither filter
    is provably true, so both actually record. An earlier version of this test used a filter
    the optimizer could prove always-true and drop, which recorded nothing and quietly tested
    the wrong thing.
    """
    hub = core.default_hub()
    rows = 20_000
    selective = bt.from_pydict({"k": list(range(rows))})  # k < 40 keeps 40 -> 0.002
    permissive = bt.from_pydict({"k": [i % 80 for i in range(rows)]})  # keeps half -> 0.5
    one = selective.filter(bt.col("k") < 40)
    two = permissive.filter(bt.col("k") < 40)
    signature = plan_signature(one._plan)
    assert signature == plan_signature(two._plan), "the collision this gate exists for"

    for _ in range(3):
        one.collect()
    learned = measured_selectivities(hub)
    assert learned.get(signature) == pytest.approx(0.002, rel=0.2), "one population is learnable"

    for _ in range(3):
        two.collect()
    assert signature not in measured_selectivities(hub), "two populations must be refused"


def test_a_refused_signature_falls_back_to_the_structural_estimate():
    """Refusing must restore the prior behavior, not produce a zero or an error."""
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.learning import load_learned_stats

    hub = core.default_hub()
    rows = 20_000
    selective = bt.from_pydict({"k": list(range(rows))})
    permissive = bt.from_pydict({"k": [i % 80 for i in range(rows)]})
    one = selective.filter(bt.col("k") < 40)
    two = permissive.filter(bt.col("k") < 40)
    for _ in range(3):
        one.collect()
    for _ in range(3):
        two.collect()

    cold = CardinalityEstimator(permissive._sources).estimate(two._plan).rows
    warm = CardinalityEstimator(permissive._sources, load_learned_stats(hub)).estimate(two._plan)
    assert warm.rows > 0
    # Whatever else the learned column statistics contribute, the refused selectivity must not
    # drag the estimate to the selective table's 0.002 of the input.
    assert warm.rows > 0.05 * rows, f"still poisoned by the other population ({warm.rows} rows)"
    assert cold > 0
