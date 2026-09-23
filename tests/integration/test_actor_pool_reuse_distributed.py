"""Back-to-back distributed runs over a session-warm `map_batches` actor pool must not hang.

A class UDF (and `lookup_join`, which runs as one) executes on a *session-warm* actor pool
that outlives the `collect()` that built it. The pool used to be keyed on the pipeline and
the stage's accelerator `opts` only, and a CPU stage's `opts` is empty: its cores come from
the class-level grant `_ensure_ray` wraps `_MapActor` with, which is sized from the worker
count. So `collect(distributed=True, num_workers=4)` left four 12-CPU actors holding all 48
cores, and a following wider run *reused* them and grew the pool with 42 one-CPU actors that
could never place. `_pipeline_actor_pool` then waited in `ray.wait` forever with every core
reserved and nothing running.

Each test here runs two differently-shaped distributed collects in a row, under a hard
deadline so a regression fails instead of hanging the suite, and holds both to
`collect(distributed=False)`. The UDFs derive a column, so a pool that served the wrong
model (the two-class test) or dropped a partition is caught by value, not only by row count.

Every UDF class is defined in this module and shipped by value
(`register_pickle_by_value`): Ray pickles a module-level class by reference, and the tests
directory is not importable on a worker.
"""

from __future__ import annotations

import sys
import threading

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="the distributed path needs Ray")
pytest.importorskip("batcher._native", reason="native engine not built")

import ray  # noqa: E402

from batcher.dist.executors import map as M  # noqa: E402
from batcher.io.lookup.stage import LookupStage  # noqa: E402

ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

#: Seconds one `collect` may take before the test calls it a hang. The healthy runs take
#: 5-15 s on a local cluster; the regression never returns.
_DEADLINE_S = 150.0
_ROWS = 4000


@pytest.fixture(scope="module", autouse=True)
def _ray():
    from _ray_cluster import init_test_ray, shutdown_test_ray

    started = init_test_ray(4)
    yield
    M.release_inference_pools()
    shutdown_test_ray(started)


@pytest.fixture(autouse=True)
def _fresh_pools():
    """Start and end every test with no warm pool, so one test's actors cannot mask another's."""
    M.release_inference_pools()
    yield
    M.release_inference_pools()


def _cluster_cpus() -> int:
    return int(ray.cluster_resources().get("CPU", 0))


def _within_deadline(fn):
    """Run `fn()` on a thread and fail, rather than hang, if it misses `_DEADLINE_S`."""
    box: dict = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # re-raised on the test thread below
            box["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(_DEADLINE_S)
    if worker.is_alive():
        pytest.fail(
            f"distributed collect did not finish within {_DEADLINE_S:.0f}s; "
            f"free resources: {ray.available_resources()}"
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


def _sorted(table: pa.Table) -> pa.Table:
    return table.sort_by("k")


def _base() -> bt.Dataset:
    return bt.from_pydict({"k": list(range(_ROWS))})


class _Double:
    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        k = batch.column("k")
        return pa.RecordBatch.from_arrays([k, pc.multiply(k, 2)], names=["k", "v"])


class _Triple:
    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        k = batch.column("k")
        return pa.RecordBatch.from_arrays([k, pc.multiply(k, 3)], names=["k", "v"])


def _granted(run):
    """Run `run`, returning its result and the actor grants the warm pool was asked for.

    Read from `_actor_grant` rather than from the registry key: the key deliberately leaves
    out the envelope-sized `num_cpus`/`memory` (a resident pool must find itself again on
    the next run), so two widths share a key while still asking for different actors.
    """
    seen: set[tuple] = set()
    real = M._actor_grant

    def spy(opts):
        grant = real(opts)
        seen.add(tuple(sorted(grant.items())))
        return grant

    M._actor_grant = spy
    try:
        return run(), seen
    finally:
        M._actor_grant = real


def _narrow_then_wide(ds_first: bt.Dataset, ds_second: bt.Dataset) -> None:
    """Run `ds_first` narrow and `ds_second` wide, each held to its single-node answer."""
    narrow, wide = 2, max(4, _cluster_cpus())
    expected_first = _sorted(ds_first.collect(distributed=False))
    expected_second = _sorted(ds_second.collect(distributed=False))

    first, grants_first = _granted(
        lambda: _within_deadline(lambda: ds_first.collect(distributed=True, num_workers=narrow))
    )
    assert M._SESSION_POOLS, "the first run did not build a session-warm actor pool"
    second, grants_second = _granted(
        lambda: _within_deadline(lambda: ds_second.collect(distributed=True, num_workers=wide))
    )

    # The route was the warm actor pool, and the two runs asked for differently-sized actors
    # -- the shape that hung. Without this the comparison below could agree for the wrong
    # reason (a stateless-task route, or two runs with one grant).
    assert grants_first and grants_second and grants_first != grants_second
    assert _sorted(first).equals(expected_first)
    assert _sorted(second).equals(expected_second)


def test_the_same_class_udf_at_two_widths():
    ds = _base().map_batches(_Double)
    _narrow_then_wide(ds, ds)


def test_the_audited_shape_bare_distributed_after_num_workers():
    """The exact repro: `num_workers=4` then a bare `distributed=True`."""
    ds = _base().map_batches(_Double)
    expected = _sorted(ds.collect(distributed=False))
    first = _within_deadline(lambda: ds.collect(distributed=True, num_workers=4))
    second = _within_deadline(lambda: ds.collect(distributed=True))
    assert _sorted(first).equals(expected)
    assert _sorted(second).equals(expected)


def test_wide_then_narrow_reuses_nothing_stale():
    ds = _base().map_batches(_Double)
    expected = _sorted(ds.collect(distributed=False))
    wide = _within_deadline(
        lambda: ds.collect(distributed=True, num_workers=max(4, _cluster_cpus()))
    )
    narrow = _within_deadline(lambda: ds.collect(distributed=True, num_workers=2))
    assert _sorted(wide).equals(expected)
    assert _sorted(narrow).equals(expected)


def test_two_different_class_udfs_back_to_back():
    """A second model must not wait on cores an idle first model's pool still holds."""
    _narrow_then_wide(_base().map_batches(_Double), _base().map_batches(_Triple))


def test_the_same_class_udf_at_two_concurrencies():
    small = _base().map_batches(_Double, concurrency=2)
    large = _base().map_batches(_Double, concurrency=6)
    expected = _sorted(small.collect(distributed=False))
    for ds, workers in ((small, 2), (large, max(4, _cluster_cpus())), (small, 2)):
        out = _within_deadline(lambda ds=ds, w=workers: ds.collect(distributed=True, num_workers=w))
        assert _sorted(out).equals(expected)


_DIM = pa.table(
    {"k": [f"c{i}" for i in range(0, _ROWS, 3)], "name": ["x"] * len(range(0, _ROWS, 3))}
)


class _InMemoryLookupStage(LookupStage):
    """`LookupStage` with its store swapped for `InMemoryLookup`, in whichever process runs it.

    `lookup_join` reaches its store through `spec.build_lookup`, and a driver-side
    monkeypatch of that does not follow the stage onto a Ray actor. Patching it from the
    stage's own `__init__` does, and leaves the rest of the real `LookupStage` path intact.
    """

    def __init__(self, **kwargs) -> None:
        from batcher.io.lookup import backends, spec

        spec.build_lookup = lambda uri, schema, options: backends.InMemoryLookup(_DIM, "k")
        super().__init__(**kwargs)


def test_lookup_join_back_to_back(monkeypatch):
    import batcher.io.lookup as lookup

    monkeypatch.setattr(lookup, "LookupStage", _InMemoryLookupStage)
    ds = bt.from_pydict({"k": [f"c{i}" for i in range(_ROWS)]}).lookup_join(
        "memory://", on="k", schema={"name": "string"}
    )
    # The store really answered: hits carry a name, misses are null-filled by the left join.
    single = ds.collect(distributed=False)
    assert single.column("name").null_count == _ROWS - _DIM.num_rows
    _narrow_then_wide(ds, ds)
