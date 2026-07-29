"""A query that goes out-of-core must still teach the optimizer.

`Core measures, Kyber decides` only holds if Core measures on *every* route. It did not:
the three spill routes each returned early, before `_close_learning_loops`, so a query big
enough to go out-of-core recorded its output row count and nothing else — and the explicit
``collect(spill=True)`` route, which bypasses `run_relational` entirely, recorded not even
that.

That is the worst possible place to drop the signal, and it is self-reinforcing: filter
selectivity, shuffled volume, group-reduction ratio and the memory-pressure flap rate are
precisely the inputs whose absence keeps the next run's estimate poor enough to spill again.

`learn_column_stats` is deliberately still absent from these routes and is asserted so
below: it measures ndv/quantiles from the scanned batches, and an out-of-core run never
holds them. Everything else needs only counts, so it is recorded.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def dataset():
    return bt.from_arrow(pa.table({"k": [i % 50 for i in range(20_000)], "v": list(range(20_000))}))


@pytest.fixture
def query(dataset):
    return dataset.filter(bt.col("v") > 10).group_by("k").agg(s=bt.col("v").sum())


def _spy(monkeypatch) -> list[str]:
    """Record which learning loops fire, without changing what they do."""
    import batcher.api.orchestration.run as run_mod
    import batcher.kyber as kyber
    from batcher.api import tuning
    from batcher.api.terminal import _metadata

    seen: list[str] = []
    for module, name, label in (
        (kyber, "record_execution", "execution"),
        (kyber, "record_selectivity", "selectivity"),
        (tuning, "record_run_feedback", "run_feedback"),
        (run_mod, "_record_flap_rate", "flap_rate"),
        (_metadata, "learn_column_stats", "column_stats"),
    ):
        original = getattr(module, name)

        def wrapper(*a, _o=original, _l=label, **k):
            seen.append(_l)
            return _o(*a, **k)

        monkeypatch.setattr(module, name, wrapper)
    return seen


@pytest.mark.integration
def test_in_memory_run_closes_every_loop(query, monkeypatch):
    """The baseline the spill routes are measured against."""
    seen = _spy(monkeypatch)
    query.collect()
    assert {"execution", "selectivity", "run_feedback", "flap_rate", "column_stats"} <= set(seen)


@pytest.mark.integration
def test_explicit_spill_records_cardinality_and_selectivity(query, monkeypatch):
    """`collect(spill=True)` bypasses `run_relational`, so it closes its own loops.

    It recorded nothing at all before — the one way to run a query that taught the
    optimizer literally nothing.
    """
    seen = _spy(monkeypatch)
    out = query.collect(spill=True)
    assert out.num_rows == 50
    assert "execution" in seen
    assert "selectivity" in seen
    # No resident batches on this route, so the whole-input measure cannot run.
    assert "column_stats" not in seen


@pytest.mark.integration
def test_in_run_spill_route_closes_the_resident_free_loops(query, monkeypatch):
    """The `rm.should_spill` route has the resource manager and the join decisions to hand,
    so it closes everything except the one measure that needs resident batches."""
    import batcher.carbonite as carbonite

    monkeypatch.setattr(carbonite.ResourceManager, "should_spill", lambda self, opt: True)
    seen = _spy(monkeypatch)
    out = query.collect()
    assert out.num_rows == 50
    assert {"execution", "selectivity", "run_feedback", "flap_rate"} <= set(seen)
    assert "column_stats" not in seen


@pytest.mark.integration
def test_spilled_result_is_unchanged(query):
    """Recording must never alter the answer.

    Order-independent on purpose: a `group_by` with no `sort` above it is an unordered
    relation, and the two routes legitimately emit their groups in different orders (bucket
    order out-of-core, hash order in memory). There is no sort in this plan, so an
    order-independent comparison cannot be hiding a sort bug.
    """
    spilled = query.collect(spill=True).to_pydict()
    memory = query.collect().to_pydict()
    assert sorted(zip(spilled["k"], spilled["s"], strict=True)) == sorted(
        zip(memory["k"], memory["s"], strict=True)
    )


@pytest.mark.integration
def test_spilled_run_actually_corrects_the_next_estimate(dataset, monkeypatch):
    """The point of closing the loop: the measurement has to reach the *estimator*.

    Recording into the hub proves only that a write happened. This asserts the read side —
    that a filter's selectivity measured on a spilled run sharpens the next estimate toward
    the truth, which is the whole reason the loop exists.
    """
    import batcher.carbonite as carbonite
    from batcher import core
    from batcher.kyber.learning import load_learned_stats
    from batcher.kyber.stats import StatsEstimator

    monkeypatch.setattr(carbonite.ResourceManager, "should_spill", lambda self, opt: True)
    filtered = dataset.filter(bt.col("v") > 18_999)  # keeps exactly 1,000 of 20,000 rows
    hub = core.default_hub()

    def estimate() -> float:
        return (
            StatsEstimator(filtered._sources, load_learned_stats(hub)).estimate(filtered._plan).rows
        )

    before = estimate()
    assert filtered.collect().num_rows == 1_000
    after = estimate()

    # The structural guess uses a generic selectivity constant; the measured one is exact.
    assert abs(after - 1_000) < abs(before - 1_000)
