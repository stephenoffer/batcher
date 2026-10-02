"""`distributed="auto"` answers single-node when the distributed route refuses the plan.

`auto` promises the same answer on either route, so a plan shape with no distributed
decomposition is not the caller's problem. TPC-H q22 at SF1 routed to the cluster by size
and raised `PlanError` there, where `distributed=False` returned its seven rows. An explicit
`distributed=True` is a request for the cluster, so it still raises.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.terminal import core

pytestmark = pytest.mark.unit


@pytest.fixture
def refusing_cluster(monkeypatch):
    """`auto` picks the cluster, and the cluster route refuses every plan."""
    calls: list[object] = []
    original = core._collect

    def collect(*args, distributed="auto", **kwargs):
        calls.append(distributed)
        if distributed is True:
            raise PlanError("no distributed decomposition")
        return original(*args, distributed=distributed, **kwargs)

    monkeypatch.setattr(core, "_collect", collect)
    monkeypatch.setattr(
        core, "_resolve_distributed", lambda d, plan=None, sources=None: d == "auto" or d is True
    )
    return calls


def test_auto_answers_single_node_when_the_cluster_refuses(refusing_cluster):
    ds = (
        bt.from_pydict({"k": [1, 2, 2], "v": [1.0, 2.0, 3.0]})
        .group_by("k")
        .agg(s=bt.col("v").sum())
    )
    got = core._collect(ds._plan, ds._sources, ds.columns)
    assert sorted(got.to_pydict()["s"]) == [1.0, 5.0]
    # The cluster was tried first: this is the fallback, not a route that never left home.
    assert True in refusing_cluster


def test_an_explicit_cluster_request_still_raises(refusing_cluster):
    ds = bt.from_pydict({"k": [1]})
    with pytest.raises(PlanError):
        core._collect(ds._plan, ds._sources, ds.columns, distributed=True)


def test_auto_routes_on_the_input_size_not_a_learned_output_size(monkeypatch):
    """A query reading 60M rows and returning 4 still distributes on its second run.

    The size learned from past runs is the query's output. Deciding on it kept TPC-H q1 at
    SF10 on the driver from its second run on (10.2 s against 0.47 s distributed).
    """
    import sys
    import types

    from batcher import dist
    from batcher.api.terminal import routing

    fake_ray = types.SimpleNamespace(is_initialized=lambda: True)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(routing, "_ray_already_live", lambda: True)
    monkeypatch.setattr(dist, "cluster_topology", lambda: {"nodes": 8, "gpus": 0.0})
    monkeypatch.setattr(routing, "_learned_size", lambda plan: 4.0)
    monkeypatch.setattr(routing, "total_source_rows", lambda sources: 60_000_000)
    ds = bt.from_pydict({"k": [1]})
    source = type("FileSource", (), {"resident": False})()
    assert routing._resolve_distributed("auto", ds._plan, [source]) is True
    # Positive control: with no row count to read, the learned size still decides.
    monkeypatch.setattr(routing, "total_source_rows", lambda sources: None)
    assert routing._resolve_distributed("auto", ds._plan, [source]) is False
