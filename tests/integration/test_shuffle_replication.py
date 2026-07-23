"""Shuffle-output replication: a lost worker is re-fetched, not recomputed.

`shuffle_replication > 1` places a second copy of every mapper's published buckets on an
off-node survivor. When a worker then dies, the reducer fetches the byte-identical copy
instead of driving a recompute round (re-reading the source from object storage and
re-running the map — usually the longest phase of the query).

Two properties are pinned here, and they are not the same property:

1. **Correctness** — the result under worker loss equals the single-node answer. This must
   hold whether the loss was served by a replica or by a recompute, so it is the assertion
   that would catch a replica silently serving the *wrong* bytes. It is the load-bearing
   one: a stale replica reads back as an EMPTY bucket rather than an error, so a bug here
   drops rows silently rather than failing.
2. **That replication actually did something** — the recompute path is not entered. Without
   this, `shuffle_replication` could be wired to nothing at all and property 1 would still
   pass via the recompute fallback, which is exactly the state this feature was in before.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher import col, count
from batcher.config import Config, DistributedConfig, FlowControlConfig, config_context

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _data():
    rng = np.random.default_rng(19)
    n = 120_000
    return pa.table(
        {"k": rng.integers(0, 40, n).astype("int64"), "v": rng.integers(0, 100, n).astype("int64")}
    )


def _norm(t: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in r.values()) for r in t.to_pylist()
    }


def _replicated(factor: int = 2):
    return config_context(
        Config().replace(distributed=DistributedConfig(shuffle_replication=factor))
    )


def _replicated_tree(factor: int = 2, fan_in: int = 2):
    # `workers > fan_in` forces the hierarchical combiner tree instead of the flat reduce.
    # The tree carries replicas POSITIONALLY alongside its source frontier (not by worker
    # id), which is a different code path from the flat reduce and needs its own coverage.
    return config_context(
        Config().replace(
            distributed=DistributedConfig(shuffle_replication=factor),
            flow_control=FlowControlConfig(shuffle_fan_in=fan_in),
        )
    )


def _agg():
    return bt.from_arrow(_data()).group_by("k").agg(s=col("v").sum(), n=count())


@pytest.mark.parametrize("killed", [{1}, {0, 2}])
def test_replicated_aggregate_survives_worker_loss(killed):
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    expected = _agg().collect()
    ds = _agg()
    with _replicated():
        recovered = execute_aggregate_flight(
            [], ds._plan, ds._sources, workers=4, _fault_inject=killed
        )
    assert _norm(recovered) == _norm(expected)


def test_replication_serves_the_loss_without_a_recompute(monkeypatch):
    # Spy on the lineage reincarnation that every recompute round must perform. With a
    # replica in place the reducer falls over to it and this is never reached; without
    # replication the same kill drives at least one reincarnate (asserted below), which
    # is what makes this a real test of the feature rather than of the fallback.
    import batcher.carbonite.resilience as res
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    calls: list[int] = []
    real = res.ShuffleLineage

    class _SpyLineage(real):  # type: ignore[misc,valid-type]
        def reincarnate(self):
            calls.append(self.src_partition)
            return real.reincarnate(self)

    # `flight_aggregate` imports ShuffleLineage inside the reduce function, so patching
    # the source module is what the call site actually resolves at runtime.
    monkeypatch.setattr(res, "ShuffleLineage", _SpyLineage)

    expected = _agg().collect()
    ds = _agg()
    with _replicated():
        recovered = execute_aggregate_flight(
            [], ds._plan, ds._sources, workers=4, _fault_inject={1}
        )

    assert _norm(recovered) == _norm(expected)
    assert calls == [], f"expected the replica to serve the loss, but recomputed sources {calls}"


def test_unreplicated_same_kill_does_recompute(monkeypatch):
    # The control for the test above: with replication off, the identical kill must drive
    # a recompute. If this ever goes quiet, the spy above stops proving anything.
    import batcher.carbonite.resilience as res
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    calls: list[int] = []
    real = res.ShuffleLineage

    class _SpyLineage(real):  # type: ignore[misc,valid-type]
        def reincarnate(self):
            calls.append(self.src_partition)
            return real.reincarnate(self)

    monkeypatch.setattr(res, "ShuffleLineage", _SpyLineage)

    ds = _agg()
    with _replicated(factor=1):
        execute_aggregate_flight([], ds._plan, ds._sources, workers=4, _fault_inject={1})

    assert calls, "unreplicated worker loss should have recomputed the lost source"


@pytest.mark.parametrize("killed", [{1}, {0, 2}])
def test_replicated_tree_reduce_survives_worker_loss(killed):
    # The wide-shuffle path: 4 workers over fan_in 2 builds a 2-level combiner tree.
    # Replicas are only available at the LEAF level (an interior combiner's output lives
    # on one node and is never copied), so this asserts the mixed case is still correct —
    # some sources served from a replica, any interior loss still recomputed.
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    expected = _agg().collect()
    ds = _agg()
    with _replicated_tree():
        recovered = execute_aggregate_flight(
            [], ds._plan, ds._sources, workers=4, _fault_inject=killed
        )
    assert _norm(recovered) == _norm(expected)


def test_tree_reduce_without_replication_still_correct():
    # Control: the tree path must stay correct with replication off, so the test above
    # is measuring replication rather than the tree's own recompute recovery.
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    expected = _agg().collect()
    ds = _agg()
    with _replicated_tree(factor=1):
        recovered = execute_aggregate_flight(
            [], ds._plan, ds._sources, workers=4, _fault_inject={1}
        )
    assert _norm(recovered) == _norm(expected)


def test_factor_one_places_no_replicas():
    from batcher.dist.shuffle_replication import replicate_shuffle_output

    with _replicated(factor=1):
        assert replicate_shuffle_output(object(), ["a", "b"], 2, 2, set()) is None


def test_single_worker_places_no_replicas():
    # A replica is only useful if it can die independently; with one worker there is
    # nowhere to put it, so replication degrades to the recompute path rather than failing.
    from batcher.dist.shuffle_replication import replicate_shuffle_output

    with _replicated(factor=2):
        assert replicate_shuffle_output(object(), ["a"], 2, 1, set()) is None
