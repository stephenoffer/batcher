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
    # The wide-shuffle path: 4 workers over fan_in 2 builds a 2-level combiner tree, whose
    # leaves and interior levels both carry replicas now (see the test below).
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


def test_the_combiner_tree_replicates_its_interior_levels(monkeypatch):
    """A tree's interior partials get an off-node copy, like its leaves.

    They did not, at any replication factor. An interior combiner's merged partial lived on
    exactly one node, so losing that node threw away every level built so far and restarted
    the tree from the leaves — the recompute leaf replication exists to avoid, reintroduced
    one level up. `shuffle_replication=2` bought a wide shuffle strictly less than it bought
    a narrow one, which is backwards: the wide shuffle is the one running on more nodes.

    Scope, stated plainly: this pins that the copies are **placed and acked**, which is the
    defect that existed (the interior was wired to nothing). It does not kill a worker
    mid-tree — the fault hooks fire before and after the map barrier, not between combiner
    levels — so "the interior replica *served* a loss" is asserted by construction from
    `_combine_sources`' positional fallback, not end to end here.
    """
    import batcher.dist.flight_aggregate as agg_mod
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    levels: list[tuple[int, int]] = []
    real = agg_mod.replicate_interior_outputs

    def _spy(actors, outputs, workers, dead, probe=None):
        out = real(actors, outputs, workers, dead, probe)
        levels.append((len(outputs), 0 if out is None else sum(len(f) for f in out)))
        return out

    monkeypatch.setattr(agg_mod, "replicate_interior_outputs", _spy)

    expected = _agg().collect()
    ds = _agg()
    with _replicated_tree():
        got = execute_aggregate_flight([], ds._plan, ds._sources, workers=4)

    assert _norm(got) == _norm(expected)
    assert levels, "fan_in 2 over 4 workers must build at least one interior level"
    for published, copies in levels:
        assert copies >= published, f"{published} interior partials but only {copies} copies"


def test_the_interior_is_not_replicated_when_replication_is_off(monkeypatch):
    # The control. Without it the test above would pass on a `replicate_interior_outputs`
    # that ignored the config and copied unconditionally, which is a cost nobody asked for.
    import batcher.dist.flight_aggregate as agg_mod
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    levels: list[int] = []
    real = agg_mod.replicate_interior_outputs

    def _spy(actors, outputs, workers, dead, probe=None):
        out = real(actors, outputs, workers, dead, probe)
        levels.append(0 if out is None else sum(len(f) for f in out))
        return out

    monkeypatch.setattr(agg_mod, "replicate_interior_outputs", _spy)

    ds = _agg()
    with _replicated_tree(factor=1):
        execute_aggregate_flight([], ds._plan, ds._sources, workers=4)

    assert levels and all(c == 0 for c in levels)


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


# ---------------------------------------------------------------------------
# The other three shuffles. Replication used to serve the aggregate alone, so a
# spot cluster got re-fetch recovery for `GROUP BY` and a full map-stage recompute
# for every join, sort and window. These pin the same two properties per operator:
# the answer survives the loss, and the replica — not the recompute — served it.
# ---------------------------------------------------------------------------


def _spy_republish(monkeypatch):
    """Record every source the join/sort/window reduce had to regenerate.

    These three do **not** recompute through `ShuffleLineage` the way the aggregate does
    — they hand `run_bucket_reduce` a `republish` closure that re-runs the map onto a
    survivor directly. Spying on the aggregate's lineage therefore observes nothing here,
    and an assertion built on it passes whether or not replication is wired at all. That
    is not hypothetical: it is what the first version of these tests did, and the
    `test_unreplicated_loss_recomputes_*` control below is what caught it. Wrap the actual
    mechanism instead.
    """
    from batcher.dist.executors.ray_runtime import reduce as reduce_mod

    calls: list[int] = []
    real = reduce_mod.run_bucket_reduce

    def _spy(*, republish, **kw):
        def _counted(target: int, src: int) -> None:
            calls.append(src)
            return republish(target, src)

        return real(republish=_counted, **kw)

    # The flight drivers do `from ...ray_runtime import run_bucket_reduce` *inside* the
    # reduce function, so patching the defining module is what the call site resolves.
    monkeypatch.setattr(reduce_mod, "run_bucket_reduce", _spy)
    monkeypatch.setattr("batcher.dist.executors.ray_runtime.run_bucket_reduce", _spy, raising=False)
    return calls


def _join_tables():
    rng = np.random.default_rng(13)
    n = 80_000
    left = pa.table(
        {"k": rng.integers(0, 80, n).astype("int64"), "lv": rng.integers(0, 50, n).astype("int64")}
    )
    right = pa.table({"k": np.arange(80, dtype="int64"), "label": [f"g{i}" for i in range(80)]})
    return left, right


def _joined():
    left, right = _join_tables()
    return bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner")


@pytest.mark.parametrize("killed", [{1}, {0, 2}])
def test_replicated_join_survives_worker_loss(killed):
    # A join mapper publishes BOTH sides under one address (left on shuffle stage 0,
    # right on stage 1), so its replica must hold both. A copy carrying only the left
    # side would not raise — an unregistered ticket reads back as an empty bucket — it
    # would silently emit an under-joined result, which is exactly what this compares.
    from batcher.dist.flight_join import execute_join_flight

    ds = _joined()
    expected = ds.collect()
    with _replicated():
        recovered = execute_join_flight([], ds._plan, ds._sources, workers=4, _fault_inject=killed)
    assert _norm(recovered) == _norm(expected)


def test_join_replication_serves_the_loss_without_a_recompute(monkeypatch):
    from batcher.dist.flight_join import execute_join_flight

    calls = _spy_republish(monkeypatch)
    ds = _joined()
    expected = ds.collect()
    with _replicated():
        recovered = execute_join_flight([], ds._plan, ds._sources, workers=4, _fault_inject={1})

    assert _norm(recovered) == _norm(expected)
    assert calls == [], f"expected the replica to serve the loss, but recomputed sources {calls}"


def _sorted_ds():
    return bt.from_arrow(_data()).sort("v")


@pytest.mark.parametrize("killed", [{1}, {0, 2}])
def test_replicated_sort_survives_worker_loss(killed):
    from batcher.dist.flight_sort import execute_sort_flight

    ds = _sorted_ds()
    expected = ds.collect()
    with _replicated():
        recovered = execute_sort_flight([], ds._plan, ds._sources, workers=4, _fault_inject=killed)
    # A sort is the one operator whose ORDER is the answer, so this compares the key
    # column position-by-position rather than as a multiset — `_norm` could not see a
    # range delivered out of order, which is precisely how a replication bug here
    # would present.
    assert recovered.column("v").to_pylist() == expected.column("v").to_pylist()


def test_sort_replication_serves_the_loss_without_a_recompute(monkeypatch):
    from batcher.dist.flight_sort import execute_sort_flight

    calls = _spy_republish(monkeypatch)
    ds = _sorted_ds()
    expected = ds.collect()
    with _replicated():
        recovered = execute_sort_flight([], ds._plan, ds._sources, workers=4, _fault_inject={1})

    assert recovered.column("v").to_pylist() == expected.column("v").to_pylist()
    assert calls == [], f"expected the replica to serve the loss, but recomputed sources {calls}"


def _windowed():
    return bt.from_arrow(_data()).with_columns(r=col("v").sum().over("k"))


@pytest.mark.parametrize("killed", [{1}, {0, 2}])
def test_replicated_window_survives_worker_loss(killed):
    from batcher.dist.flight_window import execute_window_flight

    ds = _windowed()
    expected = ds.collect()
    with _replicated():
        recovered = execute_window_flight(
            [], ds._plan, ds._sources, workers=4, _fault_inject=killed
        )
    assert _norm(recovered) == _norm(expected)


def test_window_replication_serves_the_loss_without_a_recompute(monkeypatch):
    from batcher.dist.flight_window import execute_window_flight

    calls = _spy_republish(monkeypatch)
    ds = _windowed()
    expected = ds.collect()
    with _replicated():
        recovered = execute_window_flight([], ds._plan, ds._sources, workers=4, _fault_inject={1})

    assert _norm(recovered) == _norm(expected)
    assert calls == [], f"expected the replica to serve the loss, but recomputed sources {calls}"


@pytest.mark.parametrize(
    ("shuffle", "stages"),
    [("aggregate", (0,)), ("sort", (0,)), ("window", (0,)), ("join", (0, 1))],
)
def test_every_shuffle_declares_the_stages_it_publishes(shuffle, stages, monkeypatch):
    # The wiring contract, asserted directly rather than inferred from a kill: each driver
    # must ask for a replica of every stage it published. A join that asked for stage 0
    # only would place a half-copy that silently under-joins, and no correctness test
    # above would fail on a cluster where the replica was never needed.
    import batcher.dist.shuffle_replication as repl

    seen: dict[str, tuple] = {}
    real = repl.replicate_shuffle_output

    def _spy(actors, addrs, n_reducers, workers, dead, stages=(0,)):
        seen["stages"] = tuple(stages)
        return real(actors, addrs, n_reducers, workers, dead, stages)

    for mod in ("flight_aggregate", "flight_join", "flight_sort", "flight_window"):
        monkeypatch.setattr(f"batcher.dist.{mod}.replicate_shuffle_output", _spy, raising=False)

    plans = {
        "aggregate": (_agg, "batcher.dist.flight_aggregate", "execute_aggregate_flight"),
        "join": (_joined, "batcher.dist.flight_join", "execute_join_flight"),
        "sort": (_sorted_ds, "batcher.dist.flight_sort", "execute_sort_flight"),
        "window": (_windowed, "batcher.dist.flight_window", "execute_window_flight"),
    }
    build, module, fn_name = plans[shuffle]
    import importlib

    fn = getattr(importlib.import_module(module), fn_name)
    ds = build()
    with _replicated():
        fn([], ds._plan, ds._sources, workers=4)
    assert seen.get("stages") == stages, f"{shuffle} replicated stages {seen.get('stages')}"


@pytest.mark.parametrize(
    ("module", "fn_name", "build"),
    [
        ("batcher.dist.flight_join", "execute_join_flight", "join"),
        ("batcher.dist.flight_sort", "execute_sort_flight", "sort"),
        ("batcher.dist.flight_window", "execute_window_flight", "window"),
    ],
)
def test_unreplicated_loss_recomputes_for_every_shuffle(module, fn_name, build, monkeypatch):
    # The control for the three `..._serves_the_loss_without_a_recompute` tests above.
    # Without it those assertions are vacuous: `calls == []` also holds for a shuffle that
    # never loses anything, or whose kill hook stopped working. With replication off the
    # identical kill must drive at least one recompute, so an empty list there means the
    # replica did the work rather than the fault never happening.
    import importlib

    calls = _spy_republish(monkeypatch)
    ds = {"join": _joined, "sort": _sorted_ds, "window": _windowed}[build]()
    fn = getattr(importlib.import_module(module), fn_name)
    with _replicated(factor=1):
        fn([], ds._plan, ds._sources, workers=4, _fault_inject={1})

    assert calls, f"unreplicated {build} worker loss should have recomputed the lost source"
