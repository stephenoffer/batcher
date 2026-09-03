"""A shuffle worker's control questions must not queue behind its shuffle.

A fleet actor takes `max_concurrency = FLEET_CONCURRENCY` and its data methods hold a thread
for as long as a shuffle stage runs. With the whole actor sharing one thread pool, a one-line
question queued behind minutes of `map_publish` — and three callers read a slow answer as
something worse than slow:

* `fleet._fleet` probes every actor's `addr` under a 10-second timeout to decide the warm
  fleet is healthy. A saturated worker times out and is discarded as dead.
* The drain loop asks `is_draining` at each stage boundary to migrate a spot worker's output
  before reclamation. An answer after the window closes is the same as no answer.
* The map barrier reads `published_bucket_bytes` to size the reduce against measured skew,
  on the critical path between the two phases.

Ray's concurrency groups give those methods their own threads. These tests pin the partition
by introspecting the actor's Ray metadata, which needs no cluster — and pin it in **both**
directions, because a one-sided assertion is satisfied by putting every method in one group.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ray", reason="the fleet actor is only defined behind the ray extra")

from batcher.dist.flight_worker import _CONTROL_THREADS, _FlightWorker

pytestmark = pytest.mark.unit

#: Every method that must answer while the worker is busy, and the caller that needs it to.
_CONTROL = {
    "addr": "fleet._fleet health probe, under a 10s timeout",
    "node_id": "replica placement and drain, to find a worker's failure domain",
    "partition_count": "the bucket-leak oracle",
    "published_bucket_bytes": "the map barrier's skew read, between map and reduce",
    "is_draining": "the spot-preemption drain check at a stage boundary",
    "drain_metrics": "the per-stage measurement pull the cost model learns from",
    "set_grant": "re-granting a warm fleet before the next query borrows it",
    "set_shm_peers": "`fleet._fleet` telling each worker whether a peer shares its node",
}

#: Methods that run a shuffle stage. These are what the control group exists to not wait for,
#: so each one being in the *default* group is half of the contract.
_DATA = (
    "map_publish",
    "map_publish_raw",
    "map_publish_join",
    "reduce_fetch",
    "reduce_join",
    "sort_reduce",
    "range_publish",
    "combine_finalize_fetch",
    "local_topn",
    "sample_quantiles",
    "replicate_buckets",
)


def _groups() -> dict[str, str]:
    """Method name -> the concurrency group Ray will run it in, for grouped methods only."""
    return dict(_FlightWorker.__ray_metadata__.method_meta.concurrency_group_for_methods)


def test_the_actor_declares_a_control_group():
    assert _FlightWorker.__ray_metadata__.concurrency_groups == {"control": _CONTROL_THREADS}
    # Two, not one: with a single thread a probe still queues behind the one probe already
    # running, which is the failure this exists to remove rather than halve.
    assert _CONTROL_THREADS >= 2


@pytest.mark.parametrize(("method", "why"), sorted(_CONTROL.items()))
def test_every_control_method_runs_in_the_control_group(method, why):
    assert hasattr(_FlightWorker, method), (
        f"{method} was renamed; its caller ({why}) still needs it"
    )
    assert _groups().get(method) == "control", why


@pytest.mark.parametrize("method", _DATA)
def test_a_shuffle_stage_never_runs_in_the_control_group(method):
    """The other half. Without it the whole class could be one group and every assertion
    above would still pass, while a probe queued behind a shuffle exactly as before."""
    assert hasattr(_FlightWorker, method), f"{method} was renamed; update this contract"
    assert method not in _groups()


def test_the_partition_is_exactly_the_declared_one():
    """A new method silently landing in `control` would give a long call a reserved thread and
    reintroduce the queueing one probe at a time. Asserted as an exact set so adding one is a
    deliberate edit here, with its caller named."""
    assert set(_groups()) == set(_CONTROL)


def test_ray_accepts_the_declaration():
    """The positive control. `concurrency_groups` is rejected by `.options()` and only valid on
    the decorator, so a refactor that moved it would be a silent no-op — the actor would build,
    the tests above read the class's own metadata, and nothing would say the grouping was gone.
    This asserts Ray itself resolved the declaration into its runtime metadata."""
    meta = _FlightWorker.__ray_metadata__
    assert meta.concurrency_groups, "Ray did not record any concurrency group for this actor"
    # `fleet_actor_options` sets `max_concurrency` through `.options()`; that must still be
    # accepted alongside the decorator's groups rather than colliding with them.
    _FlightWorker.options(max_concurrency=4, num_cpus=1)
