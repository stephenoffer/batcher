"""Which distributed executor each operator actually reaches — not merely what it returns.

The cluster half of `tests/unit/test_out_of_core_engagement.py`, split out because answering
it needs a real Ray cluster and that file is deliberately Ray-free.

The question here differs from the one that file asks. There, the failure is a silent fall
back to the in-memory kernel. Here it is a silent fall back to one *node*:
`dist.executor._unsupported` raises on splittable data precisely so a missing path is loud,
but on in-memory sources it runs single-node — which is right (there is no distributed data to
speak of) and therefore indistinguishable from a gap. So every shape below runs against a real
splittable Parquet source, and the assertion is which dispatch branch fired.

It earned itself on the first run. A global window with a **multi-key** `ORDER BY` did not run
slowly, it *raised*: `supports_ordered_bucket_offsets` required exactly one order key, and a
global window is not a `_split_at` pass-through so nothing carried it up. All three drivers
already cut on the leading key alone. A global `lag` raised too, and no longer does: the rows
its bucket does not hold are only the `k` before it, which a bounded boundary exchange carries
across the cut.

No shape in the table expects a raise any more, so the branch that asserted one is gone with
them. The policy it checked has not changed — `_unsupported` still refuses rather than running
a distributable shape on one node, and `test_dist_single_node_fallback` and the dispatcher's
own message tests still hold it to that — but a branch no shape reaches is a branch that would
go on passing after it stopped meaning anything.

The shape table is imported from the unit file, so the two cannot drift: an operator added
there is asked this question too, and `test_every_declared_shape_is_covered` fails if the
distributed table does not name it.
"""

from __future__ import annotations

import importlib

import pytest

import batcher as bt
from _engagement_shapes import _BUILDERS, _R, _T, EXPECTED_DISTRIBUTED
from _ray_cluster import init_test_ray, shutdown_test_ray

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


#: The distributed executors each shape must reach, as `(attribute, modules binding the name)`.
#:
#: **Both** the defining module and each re-exporting package are patched, and that is not
#: belt-and-braces. A monkeypatch follows the *name*: `_dispatch` imports several of these
#: *inside* the function, so it resolves the package attribute — through a PEP 562
#: `__getattr__` for the global-window pair. Watching only the submodule left a shape that ran
#: perfectly reporting "reached nothing", which is the instrument version of the failure this
#: file exists to find.
_DIST_ENTRY_POINTS = {
    "map": ("_distributed_map", ("batcher.dist.executors.map",)),
    "sort": ("_distributed_sort", ("batcher.dist.executors.sort",)),
    "topn": ("_distributed_topn", ("batcher.dist.executors.sort",)),
    "aggregate": ("_distributed_aggregate", ("batcher.dist.executors.aggregate",)),
    "distinct": ("_distributed_distinct", ("batcher.dist.executors.distinct",)),
    "window": ("_distributed_window", ("batcher.dist.executors.window",)),
    "join": ("_distributed_join", ("batcher.dist.executors.join",)),
    # The Flight halves of the four shapes that have one. `_dispatch` picks between these
    # and the `_distributed_*` entries above on `transport`, which is resolved from the
    # cluster — so watching only the disk half is not a narrower check, it is a check that
    # silently inverts on a fleet that has Flight. Measured: on this 5-node cluster a plain
    # `left.sort("k")` runs `flight_sort.execute_sort_flight` and returns all 2,400 rows,
    # and the spy reported "reached nothing" for seven of the twelve shapes — the exact
    # instrument failure the module docstring above describes, arrived at from the other
    # direction. `distinct` needs no entry here: `_distributed_distinct` takes `transport`
    # as an argument rather than being chosen by it.
    "sort_flight": ("execute_sort_flight", ("batcher.dist.flight_sort",)),
    "topn_flight": ("execute_topn_flight", ("batcher.dist.flight_sort",)),
    "aggregate_flight": ("execute_aggregate_flight", ("batcher.dist.flight_aggregate",)),
    "join_flight": ("execute_join_flight", ("batcher.dist.flight_join",)),
    "window_flight": ("execute_window_flight", ("batcher.dist.flight_window",)),
    "global_window": (
        "execute_global_window_disk",
        ("batcher.dist.global_window.disk", "batcher.dist.global_window"),
    ),
    "global_window_flight": (
        "execute_global_window_flight",
        ("batcher.dist.global_window.flight", "batcher.dist.global_window"),
    ),
    # The two ASOF decompositions live in `dist.executor` beside the dispatch that routes to
    # them, and they are genuinely different algebras: `by` keys co-partition on a hash, a
    # keyless ASOF range-partitions on `on` and lends each bucket the one boundary row that
    # can match across the cut. Asserting *which* ran is the point — routing a keyless ASOF
    # through the hash path would put every row in one bucket and still return the right rows.
    "asof_by": ("_distributed_asof", ("batcher.dist.executor",)),
    "asof_keyless": ("_distributed_asof_keyless", ("batcher.dist.executor",)),
}

_WORKERS = 2


@pytest.fixture
def dist_spy(monkeypatch):
    """Record which distributed executors fire, leaving their behaviour unchanged."""
    fired: list[str] = []

    for label, (attr, module_paths) in _DIST_ENTRY_POINTS.items():
        original = getattr(importlib.import_module(module_paths[0]), attr)

        def wrapper(*args, _label=label, _original=original, **kwargs):
            fired.append(_label)
            return _original(*args, **kwargs)

        for path in module_paths:
            module = importlib.import_module(path)
            assert hasattr(module, attr), f"{path} no longer binds {attr!r}"
            monkeypatch.setattr(module, attr, wrapper)
    return fired


@pytest.fixture(scope="module")
def splittable(cluster_scratch):
    """A four-file Parquet directory — a genuinely splittable source.

    In-memory sources are deliberately not used: `_unsupported` runs those single-node by
    design, so a missing distributed path would be indistinguishable from the right answer.
    """
    import pyarrow.parquet as pq

    directory = cluster_scratch("engagement_parquet")
    for part in range(4):
        pq.write_table(_T, directory / f"p{part}.parquet")
    return str(directory)


def test_the_spy_itself_records_a_call(dist_spy, splittable):
    """Guard against a vacuous suite.

    Every assertion below is "executor X fired". If the spy silently stopped patching — an
    entry point renamed, a caller switched to a spelling nothing wraps — those would fail
    loudly, but nothing would distinguish that from the engine having genuinely changed. A
    plain sort has the longest-standing distributed path, so its absence means the harness.

    Which of the two sort drivers runs is the fleet's choice, so the guard reads the same
    `EXPECTED_DISTRIBUTED` entry the parametrized test does rather than restating one of
    them. Naming `"sort"` alone made this guard fail on a Flight fleet — announcing a broken
    harness while the harness was working and the query was running correctly, which is the
    one report a vacuity guard must never produce.
    """
    _BUILDERS["sort_plain"](bt.read.parquet(splittable), bt.from_arrow(_R)).collect(
        distributed=True, num_workers=_WORKERS
    )
    assert set(dist_spy) & set(EXPECTED_DISTRIBUTED["sort_plain"])


@pytest.mark.parametrize("shape", sorted(EXPECTED_DISTRIBUTED))
def test_the_dispatcher_routes_the_shape_to_an_executor(dist_spy, splittable, shape):
    ds = _BUILDERS[shape](bt.read.parquet(splittable), bt.from_arrow(_R))
    expected = EXPECTED_DISTRIBUTED[shape]
    ds.collect(distributed=True, num_workers=_WORKERS)
    reached = set(dist_spy)
    wanted = {expected} if isinstance(expected, str) else set(expected)
    assert reached & wanted, (
        f"{shape} did not reach any of {sorted(wanted)} (reached: "
        f"{sorted(reached) or 'nothing'}) — the dispatcher routed it elsewhere, and on "
        "in-memory sources that would have been a silent single-node run"
    )
