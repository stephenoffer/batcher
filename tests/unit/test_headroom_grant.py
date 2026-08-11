"""The shuffle fleet must never reserve every core in the cluster.

The fleet is a SPREAD placement group, but not every distributed task runs inside it:
`executors.map._map_udf_task` is a plain Ray task taking a skew-adaptive, often sub-core
`num_cpus`, and the hardware probe is another. A grant that tiles each node exactly leaves
those nowhere to run, which is a deadlock rather than a slowdown — the session fleet is
cached across queries and the staged path leases it before the first stage, so the only
actor that could release the cores is the query waiting on them.

Measured on a 16 x 16-core cluster running TPC-H sf100: q1-q15 pass, then q16 hangs
indefinitely with `256.0/256.0 CPU (256.0 reserved in placement groups)` and one
`_map_udf_task` pending on `{'CPU': 0.5}`; q16 run on its own finishes in 3.2s.

The thinning must not cost fan-out — a smaller fleet is the worse trade — so every case
here asserts the worker count is preserved alongside the headroom.
"""

from __future__ import annotations

import pytest

from batcher.dist.executor import _fill_grant, _headroom_grant

pytestmark = pytest.mark.unit


def workers_at(node_cpus: list[float], grant: float) -> int:
    return sum(max(1, int(c // grant)) for c in node_cpus)


def reserved(node_cpus: list[float], grant: float) -> float:
    return sum(int(c // grant) * grant for c in node_cpus)


@pytest.mark.parametrize(
    "node_cpus",
    [
        [16.0] * 16,  # the cluster the deadlock was observed on
        [64.0] * 3,
        [2.0, 64.0, 64.0, 64.0],  # the tiny-utility-node shape `_fill_grant` handles
        [32.0, 64.0],
        [16.0, 32.0, 32.0],
        [4.0, 4.0],
    ],
)
def test_a_free_core_is_left_on_every_node_hosting_a_worker(node_cpus):
    grant = _headroom_grant(_fill_grant(node_cpus), node_cpus)
    for cores in node_cpus:
        if cores >= grant:
            assert cores - int(cores // grant) * grant >= 1.0


@pytest.mark.parametrize(
    "node_cpus",
    [[16.0] * 16, [64.0] * 3, [2.0, 64.0, 64.0, 64.0], [32.0, 64.0], [16.0, 32.0, 32.0]],
)
def test_thinning_never_costs_a_worker(node_cpus):
    """A narrower fleet would be the worse trade; only the per-worker grant may shrink."""
    fill = _fill_grant(node_cpus)
    thinned = _headroom_grant(fill, node_cpus)
    assert thinned <= fill
    assert workers_at(node_cpus, thinned) >= workers_at(node_cpus, fill)


def test_the_deadlock_shape_keeps_sixteen_workers_and_frees_a_core_per_node():
    """The exact cluster: 16 workers either way, 240 of 256 cores reserved instead of 256."""
    node_cpus = [16.0] * 16
    assert _fill_grant(node_cpus) == 16.0
    grant = _headroom_grant(16.0, node_cpus)
    assert grant == 15.0
    assert workers_at(node_cpus, grant) == 16
    assert reserved(node_cpus, grant) == 240.0
    assert sum(node_cpus) - reserved(node_cpus, grant) == 16.0


def test_a_single_core_node_is_left_alone():
    """Nothing to give back: a 1-core grant is already the thinnest possible."""
    assert _headroom_grant(1.0, [1.0, 1.0]) == 1.0


def test_a_grant_that_already_leaves_headroom_is_unchanged():
    """A 15-core grant on 16-core nodes is untouched — no gratuitous thinning."""
    assert _headroom_grant(15.0, [16.0] * 4) == 15.0
