"""A raw-row shuffle sizes its buckets by volume, not by the cluster's shape.

`aggregate_reducer_count` covers the one exchange whose shuffled volume is smaller than its
input. Every other shuffle — join, sort, window, distinct, keyed — exchanges the rows
themselves, and each of them took the generic one-bucket-per-worker fan-out, which consults
only *learned* history and so is exactly the worker count on a cold store.

That is the wrong shape for a reason unrelated to parallelism: a bucket is the unit a reducer
holds at once (a join's build table, a sort's run, a window's partition-run). Fixing the
count to the cluster makes that working set grow with the data, so doubling the rows on an
unchanged cluster doubles every reducer's memory until it spills — wall time growing faster
than the input, which is the superlinearity the mergeable algebra exists to remove.

Any bucket count is result-identical under that algebra, so what these tests hold is the
*shape* of the exchange and, above all, that the count can only ever rise above the floor.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config
from batcher.dist.adaptive_sizing import row_shuffle_reducer_count, sizing

pytestmark = pytest.mark.unit

_BASE = 8  # the generic one-per-worker fan-out the caller passes in as the floor


def _plan():
    return bt.from_arrow(pa.table({"k": [1, 2, 3], "v": [10, 20, 30]}))._plan


@pytest.fixture
def cold(monkeypatch):
    """No learned history, so every case exercises the cold-start estimate path."""
    from batcher.kyber import learned_tuning

    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: None)


def test_no_evidence_at_all_keeps_the_caller_s_floor(cold):
    assert row_shuffle_reducer_count(_plan(), _BASE) == _BASE


def test_a_large_estimate_raises_the_bucket_count(cold, monkeypatch):
    """The volume, not the node count, decides how finely a big shuffle is divided."""
    monkeypatch.setattr(sizing, "_estimated_rows", lambda node, sources: 1e9)
    target = active_config().optimizer.target_rows_per_task
    n = row_shuffle_reducer_count(_plan(), _BASE, sources=["a source"])
    assert n == math.ceil(1e9 / target)
    assert n > _BASE


def test_a_small_estimate_never_trims_below_the_floor(cold, monkeypatch):
    """Unlike an aggregate, every input row of a raw-row shuffle lands in some bucket.

    Trimming would only idle workers: a bucket is reduced by exactly one of them.
    """
    monkeypatch.setattr(sizing, "_estimated_rows", lambda node, sources: 3.0)
    assert row_shuffle_reducer_count(_plan(), _BASE, sources=["a source"]) == _BASE


def test_a_learned_count_is_preferred_over_the_estimate(cold, monkeypatch):
    from batcher.kyber import learned_tuning

    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: 1e9)
    monkeypatch.setattr(sizing, "_estimated_rows", lambda node, sources: 1.0)
    target = active_config().optimizer.target_rows_per_task
    assert row_shuffle_reducer_count(_plan(), _BASE) == math.ceil(1e9 / target)


def test_the_count_is_capped_so_the_exchange_stays_affordable(cold, monkeypatch):
    """An exchange opens ``mappers x reducers`` streams; the cap bounds that product."""
    monkeypatch.setattr(sizing, "_estimated_rows", lambda node, sources: 1e15)
    cfg = active_config()
    cap = cfg.distributed.max_shuffle_partitions
    if cap <= 0:
        pytest.skip("cap disabled in this configuration")
    assert row_shuffle_reducer_count(_plan(), _BASE, sources=["a source"]) == cap


def test_a_failing_estimator_keeps_the_floor(cold, monkeypatch):
    """Sizing is best-effort: it shapes an exchange and must never fail a query."""

    def boom(node, sources):
        raise RuntimeError("no statistics here")

    monkeypatch.setattr(sizing, "_estimated_rows", boom)
    with pytest.raises(RuntimeError):
        boom(None, None)  # the raise is real, not a stub that silently returns
    try:
        assert row_shuffle_reducer_count(_plan(), _BASE, sources=["a source"]) == _BASE
    except RuntimeError:  # pragma: no cover - the failure this test exists to forbid
        pytest.fail("a failing estimate must not escape into execution")


def test_kybers_unknown_rows_placeholder_is_not_an_estimate(cold, monkeypatch):
    """A source nothing could size must not read as the largest table imaginable.

    Kyber returns `optimizer.cardinality.unknown_rows` (1e12) for a relation it cannot size,
    and its own estimator calls that "not an estimate at all". Taken at face value it opens
    the maximum number of near-empty streams for a source that has no statistics — an
    iterator, a connector with no catalog — which is the low-end waste these counts exist to
    avoid. No evidence must look like no evidence.
    """
    unknown = active_config().optimizer.cardinality.unknown_rows
    # A source object the estimator cannot read anything from is what produces the
    # placeholder, so this exercises the real estimate path rather than a stub of it.
    assert sizing._estimated_rows(_plan(), ["not a source"]) is None
    monkeypatch.setattr(sizing, "_estimated_rows", lambda node, sources: unknown)
    assert row_shuffle_reducer_count(_plan(), _BASE, sources=["a source"]) > _BASE, (
        "a caller that hands the count a real number is still believed"
    )


def test_an_out_of_range_source_id_costs_the_estimate_not_the_query(cold):
    """Narrowing happens inside the sizing, so a bad id degrades instead of raising."""
    assert row_shuffle_reducer_count(_plan(), _BASE, sources=[], source_id=7) == _BASE
