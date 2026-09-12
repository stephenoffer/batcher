"""The per-worker core grant must be one the cluster can actually place *right now*.

A fleet is gang-scheduled: it needs one free block of `num_cpus` cores per worker, all at
once. The grant is chosen from nameplate capacity, which is the right basis for the cluster's
*shape* and says nothing about whether the bundles fit. On a cluster with a co-tenant holding
a core or two per node — another job, a second pipeline, a placement group the previous query
has not finished releasing — a nameplate-sized grant is a bundle with nowhere to go. The
placement group then pends until the timeout and the query fails with `no distributed worker
became available`, on a cluster that is almost entirely idle. Measured on this four-node
fleet: `4 bundles x 8 CPU` unsatisfiable, the same four workers at 5 cores each placed
immediately.

The thinning **preserves the worker count**, and that is the whole design. Deriving the
cluster's shape from free capacity instead was tried first and is far worse: a node whose
cores are momentarily all held disappears from the topology, a busy four-node cluster reads as
a one-node one, and the fan-out collapses to a single worker with nothing said.

"Preserves the worker count" is a claim about the *composition*, and it was false for as long
as these tests asserted it of `_fill_grant` alone. `_placeable_grant` accepts any grant tiling
free capacity into **at least** the wanted number of slots, so on a busy cluster it happily
returns a much thinner one; `_cluster_fill_workers` then recomputed the count from that thinned
grant against *nameplate* cores, and a shape asking for 8 workers of 47 came out as **192 of
2**. So the count now comes from the nameplate shape and the thinning applies only to the
per-worker CPU ask — and the tests below say so at the level the property actually holds.
"""

from __future__ import annotations

import pytest

from batcher.dist import executor
from batcher.dist.executors.ray_runtime import scaling

pytestmark = pytest.mark.unit


def _cluster(monkeypatch, specs):
    """specs: list of (nameplate_cpus, free_cpus)."""
    rows = [
        {
            "node_id": f"n{i}",
            "cpus": float(c),
            "free_cpus": float(f),
            "gpus": 0.0,
            "memory": 0.0,
            "accelerators": 0.0,
            "accelerator_type": None,
        }
        for i, (c, f) in enumerate(specs)
    ]
    monkeypatch.setattr(scaling, "node_classes", lambda: rows)
    return [r["cpus"] for r in rows]


def _single_numa(monkeypatch):
    """One NUMA domain, so `_numa_sliced` is a no-op and these tests are about placement."""
    monkeypatch.setattr(scaling, "cluster_numa_nodes", lambda: 1)


def test_an_idle_cluster_keeps_the_nameplate_grant(monkeypatch):
    """The common case — a single-tenant run — must be untouched."""
    cpus = _cluster(monkeypatch, [(8, 8)] * 4)
    assert executor._fill_grant(cpus) == 8.0
    assert executor._placeable_grant(8.0, cpus) == 8.0


def test_a_co_tenant_thins_the_grant_until_the_gang_fits(monkeypatch):
    """The regression: one core held per node makes an 8-core bundle unplaceable on every
    node, so four 8-core workers can never be gang-scheduled. Seven still tiles four ways."""
    cpus = _cluster(monkeypatch, [(8, 7)] * 4)
    assert executor._fill_grant(cpus) == 8.0  # the shape is nameplate
    assert executor._placeable_grant(8.0, cpus) == 7.0  # the ask is what fits


def test_thinning_preserves_the_worker_count(monkeypatch):
    """Uneven free capacity: 8 tiles twice, 7 three times, 5 four times. The nameplate wants
    four workers, so 5 is the largest grant that still yields four."""
    cpus = _cluster(monkeypatch, [(8, 5), (8, 8), (8, 8), (8, 7)])
    grant = executor._placeable_grant(executor._fill_grant(cpus), cpus)
    assert grant == 5.0
    free = [5.0, 8.0, 8.0, 7.0]
    assert sum(int(c // grant) for c in free) >= 4


def test_a_busy_cluster_does_not_multiply_the_fan_out(monkeypatch):
    """The thinned *ask* must never become a thinner *shape*.

    A cluster whose cores are nearly all held — which is every stage after the first of a
    staged query, because the fleet it is about to borrow is holding them — thins the grant
    hard. Counting workers from that thinned grant is what turned 4 nodes x 1 worker into
    `sum(8 // 1) = 32` two-core workers on the real cluster (8 of 47 became 192 of 2, and the
    query ran 2.4x slower than the same fan-out asked for explicitly).
    """
    _single_numa(monkeypatch)
    _cluster(monkeypatch, [(8, 1)] * 4)
    workers, num_cpus, _width = executor._cluster_fill_workers()
    assert workers == 4, "the fan-out is the cluster's shape, not what is free this instant"
    assert num_cpus <= 8.0


def test_a_genuinely_full_cluster_keeps_the_grant(monkeypatch):
    """With nothing free, no grant tiles — thinning to one core would buy nothing and would
    cache a one-core fleet for the rest of the session. Wait for capacity instead."""
    cpus = _cluster(monkeypatch, [(8, 0)] * 4)
    assert executor._placeable_grant(executor._fill_grant(cpus), cpus) == 8.0


def test_an_unreadable_free_reading_keeps_the_grant(monkeypatch):
    """`free_cpus` is absent wherever the per-node figures could not be read; absent must
    mean nameplate, which is the behaviour before any of this existed."""
    monkeypatch.setattr(scaling, "node_classes", list)
    assert executor._placeable_grant(8.0, [8.0] * 4) == 8.0


# ---- the explicit `num_workers` path is thinned too --------------------------------------


def test_an_explicit_fan_out_is_thinned_against_free_capacity(monkeypatch):
    """`num_workers=N` composes `_even_cpu_share` with the same thinning the fill path uses.

    The automatic fan-out has consulted free capacity since `_placeable_grant` existed; the
    explicit one never did, and it is the path every benchmark and integration test in this
    repo takes. `_even_cpu_share` divides *nameplate* cores by the worker count, so on this
    four-node cluster `num_workers=4` asks for a full 8-core bundle per worker however much
    of the cluster something else is already holding — a gang with nowhere to go, and a
    placement group that pends until the timeout before the query falls back and runs anyway.

    Seven still tiles four ways with one core per node held, so the ask comes down to seven
    and the worker count is untouched, exactly as on the fill path.
    """
    cpus = _cluster(monkeypatch, [(8, 7)] * 4)
    assert executor._even_cpu_share(4) == 8.0, "nameplate sizing is unchanged"
    assert executor._placeable_grant(executor._even_cpu_share(4), cpus) == 7.0


def test_an_explicit_fan_out_on_an_idle_cluster_is_unchanged(monkeypatch):
    """The single-tenant run — the common one — must keep the grant it had, byte for byte."""
    cpus = _cluster(monkeypatch, [(8, 8)] * 4)
    assert executor._placeable_grant(executor._even_cpu_share(4), cpus) == 8.0
    assert executor._placeable_grant(executor._even_cpu_share(8), cpus) == 4.0  # 32 // 8
