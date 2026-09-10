"""A warm pool for one pipeline must not wedge a different pipeline that needs its cores.

The map/aggregate route keeps a pipeline's actors alive for `distributed.session_fleet_idle_s`
so a back-to-back query reuses their scan cache, and a timer returns the cores when the session
goes quiet. That is right for the pipeline running again and wrong for a **different** one, and
the failure it produced was a self-deadlock rather than a slowdown.

Measured on six 8-core nodes: an ordinary CPU query returns leaving its pool holding **42 of 48
cores**; the next query — a different pipeline — enters the route, takes the in-use lease, and
submits map tasks at **6.144 CPU each against 6.0 free**. None can be placed, and the lease it
holds is exactly what stops the stale pool being reclaimed. `0/6 tasks finished`, indefinitely,
every GPU idle. The controlled repro was one variable: run an ordinary CPU query first, and the
identical GPU query goes from 7.94 s to never finishing.

What is pinned here is the *selection rule*, because that is where this can silently regress in
either direction: releasing a pool the current pipeline would have reused (losing the warm cache
the feature exists for), or keeping a foreign one while a stage starves.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors import map as M

pytestmark = pytest.mark.unit


@pytest.fixture
def pools(monkeypatch):
    """An empty pool registry restored after the test, so nothing leaks between cases."""
    registry: dict = {}
    monkeypatch.setattr(M, "_AGG_POOLS", registry)
    monkeypatch.setattr(M, "_kill_pool_keys", lambda keys, reg: [reg.pop(k, None) for k in keys])
    return registry


def _free_cpu(monkeypatch, free: float):
    """Pin what Ray reports free, since the rule is gated on the cluster actually being short."""
    import sys
    import types

    fake = types.SimpleNamespace(available_resources=lambda: {"CPU": free})
    monkeypatch.setitem(sys.modules, "ray", fake)


def test_a_foreign_pool_is_released_when_this_stage_cannot_place(monkeypatch, pools):
    """The measured shape: someone else's pool holds the cores and this stage needs them."""
    monkeypatch.setattr(M, "_pipeline_signature", lambda plan: "mine")
    pools[("theirs", "cpu")] = ["actor"]
    _free_cpu(monkeypatch, free=6.0)

    assert M.release_foreign_agg_pools(object(), needed_cpus=16.0) is True
    assert pools == {}


def test_this_pipelines_own_pool_is_never_released(monkeypatch, pools):
    """The positive control, and the whole point of keying on the signature.

    Without it, an assertion that a pool is released would pass just as well against an
    implementation that released every pool — which would delete the warm-cache optimization
    while looking like a fix.
    """
    monkeypatch.setattr(M, "_pipeline_signature", lambda plan: "mine")
    pools[("mine", "cpu")] = ["actor"]
    _free_cpu(monkeypatch, free=0.0)

    assert M.release_foreign_agg_pools(object(), needed_cpus=16.0) is False
    assert ("mine", "cpu") in pools


def test_a_foreign_pool_is_kept_when_the_stage_fits_anyway(monkeypatch, pools):
    """Gated on the cluster being short: a query that fits leaves the cache warm."""
    monkeypatch.setattr(M, "_pipeline_signature", lambda plan: "mine")
    pools[("theirs", "cpu")] = ["actor"]
    _free_cpu(monkeypatch, free=48.0)

    assert M.release_foreign_agg_pools(object(), needed_cpus=16.0) is False
    assert ("theirs", "cpu") in pools


def test_only_the_foreign_entries_go(monkeypatch, pools):
    """A session holding both must keep its own and give up the other."""
    monkeypatch.setattr(M, "_pipeline_signature", lambda plan: "mine")
    pools[("mine", "cpu")] = ["a"]
    pools[("theirs", "cpu")] = ["b"]
    pools[("other", "cpu")] = ["c"]
    _free_cpu(monkeypatch, free=1.0)

    assert M.release_foreign_agg_pools(object(), needed_cpus=16.0) is True
    assert list(pools) == [("mine", "cpu")]


def test_no_pools_is_not_an_error(monkeypatch, pools):
    """The common case — nothing warm — must be a cheap False and touch nothing."""
    monkeypatch.setattr(M, "_pipeline_signature", lambda plan: "mine")
    assert M.release_foreign_agg_pools(object(), needed_cpus=16.0) is False


def test_an_unreadable_cluster_keeps_the_pool(monkeypatch, pools):
    """A scheduling courtesy must never fail a query, and must not act on a reading it
    could not take: if free capacity cannot be read, the pool stays."""
    monkeypatch.setattr(M, "_pipeline_signature", lambda plan: "mine")
    pools[("theirs", "cpu")] = ["actor"]

    import sys
    import types

    def _boom():
        raise RuntimeError("no gcs")

    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(available_resources=_boom))

    assert M.release_foreign_agg_pools(object(), needed_cpus=16.0) is False
    assert ("theirs", "cpu") in pools
