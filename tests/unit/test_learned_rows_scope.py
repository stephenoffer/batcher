"""A learned row count belongs to the operator it was measured on — not to every scan.

`plan_signature` structures every `Scan` as the bare token ``["scan"]``: it carries no
source identity, so *all* scans in a process hash to one key. That is fine as long as
nothing reads a learned **row count** for a scan — a base relation's cardinality is
already exact (a Parquet footer, an in-memory row count), so there is nothing to learn.

It was not fine when the estimator's learned-first branch applied to every node kind. Then
running any query whose root was a scan wrote that table's size under the shared key, and
the *next* query's scan — of a completely different relation — read it back. A 1,000-row
change set inherited a 5M-row table's cardinality, its join was sized at 2.4 TB, and
Carbonite spilled a 100,000-row build side to disk: a 15x slowdown on a pruned MERGE,
caused entirely by one relation's measurement masquerading as another's.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import kyber
from batcher.kyber.signature import plan_signature
from batcher.kyber.stats import StatsEstimator
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.expr_ir import count
from batcher.plan.stats import Provenance


def _estimate(ds, learned):
    return StatsEstimator(ds._sources, learned=learned).estimate(ds._plan)


def test_one_tables_measured_rows_never_becomes_another_tables_estimate() -> None:
    """The regression: a big table's measurement must not size a different, small scan."""
    big = bt.from_arrow(pa.table({"id": pa.array(range(5_000_000), pa.int64())}))
    small = bt.from_arrow(pa.table({"id": pa.array(range(10), pa.int64())}))

    # Both are bare scans, so they share a plan signature — that is the trap.
    assert plan_signature(big._plan) == plan_signature(small._plan)

    # The big table gets read and measured; the loop records its 5M rows under that key.
    learned = {plan_signature(big._plan): {"rows": 5_000_000.0, "n_obs": 1}}

    estimate = _estimate(small, learned)
    assert estimate.rows == 10, (
        f"the 10-row scan inherited the 5M-row table's learned cardinality ({estimate.rows})"
    )
    # And it is EXACT, not a weaker learned guess: the source counted it.
    assert estimate.provenance is Provenance.EXACT


def test_a_learned_row_count_still_corrects_an_aggregate() -> None:
    """The fix must not disable learning where it belongs — a breaker's output size."""
    ds = bt.from_arrow(pa.table({"k": [i % 10 for i in range(1000)]})).group_by("k").agg(n=count())

    base = _estimate(ds, {})
    assert base.provenance is Provenance.DEFAULT  # the "groups ≈ 10% of input" guess

    learned = {plan_signature(ds._plan): {"rows": 10.0}}  # measured: 10 groups
    after = _estimate(ds, learned)
    assert after.rows == 10
    assert after.provenance is Provenance.LEARNED


def test_record_execution_on_a_scan_does_not_poison_a_later_scan() -> None:
    """End to end through the real learning API, not a hand-built `learned` dict."""
    hub = MetadataHub(InProcessBackend())
    big = bt.from_arrow(pa.table({"id": pa.array(range(5_000_000), pa.int64())}))
    kyber.record_execution(hub, big._plan, 5_000_000)

    small = bt.from_arrow(pa.table({"id": pa.array(range(10), pa.int64())}))
    estimate = _estimate(small, hub.load_keyed_params("kyber.stats"))
    assert estimate.rows == 10


def test_a_settled_estimate_stops_invalidating_memoized_plans():
    """The generation follows the *stored* estimate, not the raw observation.

    A signature is structural, so two queries of the same shape share one learned row count
    and feed it different numbers. Judged against the raw observation, that entry is
    "materially changed" on every single execution — and the learned generation is
    **process-global**, so every such execution invalidated every memoized plan in the
    session. Measured on TPC-DS at scale 1 with sixty of the suite's queries already run:
    q77's signature held 100 while q77 measured 44 every time, and q77 re-planned for 131 ms
    of its 268 ms on every run, forever.

    Judged against the smoothed value the estimator actually reads, the entry settles: the
    step shrinks as evidence accrues, and once it is inside the materiality band the
    invalidation stops. The estimate keeps moving; the *re-planning* does not.
    """
    from batcher.kyber import learning
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends.in_process import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    plan = bt.from_pydict({"a": list(range(50))}).filter(bt.col("a") > 0)._plan

    learning.record_execution(hub, plan, 100)
    # Alternate two observations of the same shape, as two colliding queries would.
    for i in range(12):
        learning.record_execution(hub, plan, 44 if i % 2 else 100)

    before = learning.generation()
    for i in range(10):
        learning.record_execution(hub, plan, 44 if i % 2 else 100)
    assert learning.generation() == before, "a settled estimate is still re-planning the session"


def test_a_real_cardinality_shift_still_invalidates():
    """The other half: an estimate that genuinely moves must still drop the memo."""
    from batcher.kyber import learning
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends.in_process import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    plan = bt.from_pydict({"a": list(range(50))}).filter(bt.col("a") > 0)._plan
    for _ in range(8):
        learning.record_execution(hub, plan, 100)

    before = learning.generation()
    for _ in range(8):  # the relation grew by two orders of magnitude
        learning.record_execution(hub, plan, 10_000)
    assert learning.generation() != before, "a real shift served a stale plan"
