"""`distributed="auto"` routes by DATA SIZE, not just cluster topology.

On a multi-node cluster the Ray fan-out is a ~2 s fixed cost, so a small query must stay
single-node (the sub-second-small-query mandate). An unknown/large input distributes; an
explicit bool always wins. Result is identical either way — this only chooses where to run.

A GPU stage distributes regardless of size, because the work has to reach the accelerators
— but only if the cluster *has* any. That qualifier was missing, and on a GPU-less cluster
the forced route asked Ray for a `num_gpus=1` resource no node could ever offer, so the
task never scheduled: `TaskUnschedulableError`, or a hang, from a query that runs fine
in-process. The same plan already runs locally whenever Ray is not up, so falling through
to the size decision is the consistent answer, not a special case.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.api.terminal.routing import resolve_distributed

pytestmark = pytest.mark.unit


class _Src:
    def __init__(self, rows: int | None, *, node_local: bool = False):
        self._rows = rows
        self.node_local = node_local

    def row_count(self) -> int | None:
        return self._rows


def _fake_cluster(monkeypatch, *, nodes: int, gpus: float) -> None:
    """Pretend we're on an initialized Ray cluster of a given shape."""

    class _Ray:
        @staticmethod
        def is_initialized():
            return True

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
    monkeypatch.setattr(
        "batcher.dist.cluster_topology",
        lambda: {"nodes": nodes, "cpus": 32.0, "gpus": gpus},
        raising=False,
    )


@pytest.fixture
def multinode(monkeypatch):
    """A 4-node cluster with GPUs — the shape most of these tests assume."""
    _fake_cluster(monkeypatch, nodes=4, gpus=4.0)


@pytest.fixture
def multinode_cpu_only(monkeypatch):
    """A 4-node cluster with no accelerators at all."""
    _fake_cluster(monkeypatch, nodes=4, gpus=0.0)


def test_explicit_bool_always_wins(multinode):
    assert resolve_distributed(True) is True
    assert resolve_distributed(False) is False


def test_small_input_stays_single_node(multinode):
    # 80k rows, well below the 1M default threshold -> single-node (avoid the fan-out tax).
    assert resolve_distributed("auto", None, [_Src(80_000)]) is False


def test_large_input_distributes(multinode):
    assert resolve_distributed("auto", None, [_Src(50_000_000)]) is True


def test_unknown_size_distributes(multinode):
    # A source that can't cheaply report a row count -> distribute (safe for large data).
    assert resolve_distributed("auto", None, [_Src(None)]) is True
    assert resolve_distributed("auto", None, None) is True


def test_unknown_size_over_a_node_local_path_stays_single_node(multinode):
    """ "Distribute when the size is unknown" is a throughput bet, and it needs reachable data.

    A bare filesystem path may be this node's own disk, and shipping the scan to workers on
    other machines fails outright — `path ... does not exist` from the worker — rather than
    running slowly. The two conditions coincide exactly: the formats with no cheap row count
    (CSV, JSON, text, the bioinformatics readers) are the ones that reach this branch, while
    Parquet answers from its footer and never does. So on any Ray-connected process
    `bt.read.csv("/tmp/x.csv").collect()` died on a three-row file, with no `distributed=`
    argument anywhere in the call.
    """
    assert resolve_distributed("auto", None, [_Src(None, node_local=True)]) is False
    # One unreachable source is enough to ground the whole plan.
    assert resolve_distributed("auto", None, [_Src(None), _Src(None, node_local=True)]) is False
    # A remote URI of unknown size still distributes — that is the case the rule is for.
    assert resolve_distributed("auto", None, [_Src(None)]) is True


def test_a_node_local_path_of_known_size_is_unaffected(multinode):
    """The locality check guards the *unknown-size* branch only.

    A shared mount is a bare path too, and the big ones are Parquet, which reports an exact
    footer row count. Grounding those would have turned every recorded distributed benchmark
    single-node, so the size decision still runs first and still wins.
    """
    assert resolve_distributed("auto", None, [_Src(10_000_000, node_local=True)]) is True
    assert resolve_distributed("auto", None, [_Src(1, node_local=True)]) is False
    # And an explicit request always beats the inference: the caller may know it is shared.
    assert resolve_distributed(True, None, [_Src(None, node_local=True)]) is True


def _gpu_stage():
    """A tiny input carrying a GPU map stage. The `fn` is CPU work; `num_gpus` is the tag."""
    return bt.from_pydict({"x": [1, 2, 3]}).map_batches(lambda b: b, num_gpus=1.0)


def test_gpu_stage_distributes_when_the_cluster_has_gpus(multinode):
    # A tiny input, but the work has to reach the cluster's accelerators.
    ds = _gpu_stage()
    assert resolve_distributed("auto", ds._plan, [_Src(10)]) is True


def test_gpu_stage_stays_local_on_a_gpuless_cluster(multinode_cpu_only):
    # Routing it to a cluster with no GPUs asks Ray for a resource no node can offer, and
    # the task never schedules. Running it here is both correct and the only thing that
    # finishes.
    ds = _gpu_stage()
    assert resolve_distributed("auto", ds._plan, [_Src(10)]) is False


def test_explicit_true_beats_the_gpuless_gate(multinode_cpu_only):
    # The gate is for "auto". An explicit choice is the caller's to make, and to own.
    ds = _gpu_stage()
    assert resolve_distributed(True, ds._plan, [_Src(10)]) is True


def test_threshold_respects_config(multinode, monkeypatch):
    import dataclasses

    from batcher.config import active_config

    base = active_config()
    dist_cfg = dataclasses.replace(base.distributed, distribute_min_rows=100)
    lowered = base.replace(distributed=dist_cfg)
    monkeypatch.setattr("batcher.config.active_config", lambda: lowered)
    # 80k rows now exceeds a tiny threshold -> distribute.
    assert resolve_distributed("auto", None, [_Src(80_000)]) is True


def test_an_accelerator_stage_that_stays_local_says_so(monkeypatch, recwarn):
    """The silent outcome this replaces, on the most ordinary cluster shape there is.

    A CPU head node with GPU workers is the normal Ray/Anyscale layout. `auto` will not start
    Ray to discover the cluster (a deliberate 444 ms saving on every local query), so a stage
    carrying `num_gpus` that forgets `distributed=True` resolves to single-node and runs the
    model on the driver's CPU — the right answer, arbitrarily slower, with nothing said. The
    routing decision is unchanged; it just stops being silent.
    """
    import batcher as bt
    from batcher._internal.errors import PerformanceWarning
    from batcher.api.terminal import routing

    monkeypatch.setattr(routing, "_local_accelerator_present", lambda: False)
    gpu = bt.from_pydict({"x": [1]}).ml.map_batches(lambda b: b, num_gpus=1)

    assert routing.resolve_distributed(False, gpu._plan, None) is False
    messages = [str(w.message) for w in recwarn if w.category is PerformanceWarning]
    assert any("requested an accelerator" in m for m in messages), messages
    assert any("distributed=True" in m for m in messages), messages


def test_no_accelerator_warning_without_an_accelerator_stage(monkeypatch, recwarn):
    """A plain CPU query must stay quiet — this warning fires on a real GPU pipeline or not
    at all, or it becomes noise everyone filters."""
    import batcher as bt
    from batcher._internal.errors import PerformanceWarning
    from batcher.api.terminal import routing

    monkeypatch.setattr(routing, "_local_accelerator_present", lambda: False)
    cpu = bt.from_pydict({"x": [1]}).ml.map_batches(lambda b: b)

    routing.resolve_distributed(False, cpu._plan, None)
    assert [w for w in recwarn if w.category is PerformanceWarning] == []


def test_no_accelerator_warning_when_the_device_is_present_or_unknown(monkeypatch, recwarn):
    """Only a *positively established* absence warns.

    A host whose devices cannot be read is not a host without devices, and warning on every
    GPU query run on an actual GPU box would be worse than the silence this replaces.
    """
    import batcher as bt
    from batcher._internal.errors import PerformanceWarning
    from batcher.api.terminal import routing

    gpu = bt.from_pydict({"x": [1]}).ml.map_batches(lambda b: b, num_gpus=1)
    for verdict in (True, None):
        monkeypatch.setattr(routing, "_local_accelerator_present", lambda v=verdict: v)
        routing.resolve_distributed(False, gpu._plan, None)
    # Matched on the message: building the plan above emits its own PerformanceWarning (a
    # plain function on a GPU stage reloads the model per batch), which is a different
    # complaint and not the one under test.
    assert [
        str(w.message)
        for w in recwarn
        if w.category is PerformanceWarning and "requested an accelerator" in str(w.message)
    ] == []


def test_resident_cpu_sources_never_read_the_cluster(monkeypatch):
    """A resident CPU-only plan answers "single-node" without a `cluster_topology()` RPC.

    The answer was never in doubt — resident sources return `False` at every cluster size —
    but it used to cost a GCS round-trip to confirm, on *every* terminal op in any process
    where something had initialized Ray (an Anyscale workspace, a Daft/Ray Data comparison,
    any Ray-using library). Pinned by making the read fail: a call would raise, and the
    `except` arm would return the same `False` for the wrong reason, so the counter is what
    makes this test able to fail.
    """
    calls = []

    class _Ray:
        @staticmethod
        def is_initialized():
            return True

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)

    def _counted():
        calls.append(1)
        return {"nodes": 4, "cpus": 32.0, "gpus": 4.0}

    monkeypatch.setattr("batcher.dist.cluster_topology", _counted, raising=False)

    resident = _Src(50_000_000)
    resident.resident = True
    assert resolve_distributed("auto", None, [resident]) is False
    assert calls == []

    # The GPU arm still has to read it: an accelerator stage distributes on capability, not
    # size, and whether the cluster *has* accelerators is only knowable from the topology.
    gpu = bt.from_pydict({"x": [1, 2, 3]}).map_batches(lambda b: b, num_gpus=1.0)
    assert resolve_distributed("auto", gpu._plan, [resident]) is True
    assert calls == [1]
