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


def test_an_idle_cluster_keeps_the_nameplate_grant(monkeypatch):
    """The common case — a single-tenant run — must be untouched."""
    cpus = _cluster(monkeypatch, [(8, 8)] * 4)
    assert executor._fill_grant(cpus) == 8.0


def test_a_co_tenant_thins_the_grant_until_the_gang_fits(monkeypatch):
    """The regression: one core held per node makes an 8-core bundle unplaceable on every
    node, so four 8-core workers can never be gang-scheduled. Seven still tiles four ways."""
    cpus = _cluster(monkeypatch, [(8, 7)] * 4)
    assert executor._fill_grant(cpus) == 7.0


def test_thinning_preserves_the_worker_count(monkeypatch):
    """Uneven free capacity: 8 tiles twice, 7 three times, 5 four times. The nameplate wants
    four workers, so 5 is the largest grant that still yields four."""
    cpus = _cluster(monkeypatch, [(8, 5), (8, 8), (8, 8), (8, 7)])
    grant = executor._fill_grant(cpus)
    assert grant == 5.0
    free = [5.0, 8.0, 8.0, 7.0]
    assert sum(int(c // grant) for c in free) >= 4


def test_a_genuinely_full_cluster_keeps_the_grant(monkeypatch):
    """With nothing free, no grant tiles — thinning to one core would buy nothing and would
    cache a one-core fleet for the rest of the session. Wait for capacity instead."""
    cpus = _cluster(monkeypatch, [(8, 0)] * 4)
    assert executor._fill_grant(cpus) == 8.0


def test_an_unreadable_free_reading_keeps_the_grant(monkeypatch):
    """`free_cpus` is absent wherever the per-node figures could not be read; absent must
    mean nameplate, which is the behaviour before any of this existed."""
    monkeypatch.setattr(scaling, "node_classes", lambda: [])
    assert executor._fill_grant([8.0] * 4) == 8.0
