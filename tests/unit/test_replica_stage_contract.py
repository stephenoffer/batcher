"""A shuffle replica copies the ticket stages its mapper actually published under.

Every Flight shuffle publishes its map buckets under its own `next_stage_base` block (never
below 100), while `replicate_shuffle_output` used to default to stage 0. So each replica
copied a ticket nobody had published — and an unregistered ticket reads back as an EMPTY
bucket rather than an error. The empty copy acked, was advertised, and a reducer whose
primary was gone fell over to it and silently dropped that mapper's rows: 90,000 of 120,000
on a four-worker aggregate with two workers killed, every group present and every sum short.

Pinned here without Ray: the stage a caller names is the stage that is copied, a caller can
no longer omit it, and no call site in the package passes a literal.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

from batcher.config import Config, config_context
from batcher.dist import shuffle_replication as repl

_PKG = Path(repl.__file__).resolve().parents[1]


class _Ref:
    def __init__(self, value=None, error: Exception | None = None) -> None:
        self.value, self.error = value, error


class _Ray:
    """The `ray` calls the module makes, resolving `_Ref`s immediately."""

    def get(self, ref, timeout=None):
        if isinstance(ref, list):
            return [self.get(r) for r in ref]
        if ref.error is not None:
            raise ref.error
        return ref.value

    def wait(self, refs, num_returns=1, **_):
        return list(refs), []


class _Actor:
    """A worker on its own node that records every `replicate_buckets` call it receives."""

    def __init__(self, node: str, addr: str, calls: list) -> None:
        self._node, self._addr, self._calls = node, addr, calls

    def _remote(self, fn):
        return type("_M", (), {"remote": staticmethod(fn)})()

    @property
    def node_id(self):
        return self._remote(lambda: _Ref(self._node))

    @property
    def addr(self):
        return self._remote(lambda: _Ref(self._addr))

    @property
    def replicate_buckets(self):
        def _call(primary, src, n_buckets, stage, epoch, plan_id):
            self._calls.append((self._addr, src, stage, epoch))
            return _Ref(self._addr)

        return self._remote(_call)


@pytest.fixture
def replicating(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", _Ray())
    base = Config()
    cfg = base.replace(distributed=dataclasses.replace(base.distributed, shuffle_replication=2))
    with config_context(cfg):
        yield


@pytest.mark.parametrize("stages", [(137,), (212, 213)])
def test_the_replica_copies_exactly_the_stages_the_caller_published(replicating, stages):
    calls: list = []
    addrs = ["a0:1", "a1:1"]
    actors = [_Actor("nodeA", addrs[0], calls), _Actor("nodeB", addrs[1], calls)]

    out = repl.replicate_shuffle_output(actors, addrs, 3, 2, set(), stages=stages)

    assert out == [["a1:1"], ["a0:1"]], "each source's copy lives on the other node"
    # Every (source, stage) pair is copied once, at the published stage and epoch 0 — and no
    # other stage is touched, stage 0 above all.
    assert sorted((src, stage) for _a, src, stage, _e in calls) == sorted(
        (src, stage) for src in range(2) for stage in stages
    )
    assert {epoch for *_x, epoch in calls} == {0}


def test_the_stage_cannot_be_left_to_a_default(replicating):
    # The default was the bug: omitting the stage must be a TypeError at the call site, not
    # a replica of a ticket that was never published.
    with pytest.raises(TypeError):
        repl.replicate_shuffle_output([], [], 1, 2, set())  # type: ignore[call-arg]


def _call_sites() -> list[tuple[Path, ast.Call]]:
    sites = []
    for path in sorted(_PKG.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if name == "replicate_shuffle_output":
                    sites.append((path, node))
    return sites


def test_the_package_has_the_call_sites_this_guard_reads():
    # A positive control: without it the guard below would pass on a moved package or a
    # renamed function by reading zero call sites. Aggregate, join, sort and both windows.
    assert len(_call_sites()) >= 5


@pytest.mark.parametrize(
    "site", _call_sites(), ids=lambda s: f"{s[0].relative_to(_PKG)}:{s[1].lineno}"
)
def test_no_call_site_names_a_literal_stage(site):
    path, call = site
    kw = {k.arg: k.value for k in call.keywords}
    assert "stages" in kw, f"{path}:{call.lineno} omits stages="
    literals = [
        n for n in ast.walk(kw["stages"]) if isinstance(n, ast.Constant) and n.value != 1
    ]  # `stage_base + 1` is a join's right side, not a literal stage
    assert not literals, (
        f"{path}:{call.lineno} replicates a literal stage; pass the block the mappers "
        "published under (`next_stage_base`)"
    )


def test_the_fault_hook_returns_only_once_the_workers_address_is_closed(monkeypatch):
    # `ray.kill` is asynchronous: the killed worker kept serving its Flight port for ~30 ms,
    # long enough for the reduce to read every bucket from it, so the hooks injected no loss
    # and the recompute/replica assertions were decided by that race. A call on the killed
    # handle raising is not the signal either — it fires while the server is still up.
    import socket

    from batcher.dist.executors.ray_runtime import kill_workers

    calls: list = []
    actors = [_Actor(f"n{i}", f"10.0.0.{i}:70{i}", calls) for i in range(3)]
    killed: list = []
    fake = _Ray()
    fake.kill = killed.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", fake)

    # Worker 1's server keeps accepting for three probes after the kill, worker 2's for none.
    still_up = {("10.0.0.1", 701): 3, ("10.0.0.2", 702): 0}
    probes: list = []

    class _Conn:
        def close(self) -> None:
            pass

    def _connect(address, timeout=None):
        probes.append(address)
        if still_up.get(address, 0) > 0:
            still_up[address] -= 1
            return _Conn()
        raise ConnectionRefusedError(address)

    monkeypatch.setattr(socket, "create_connection", _connect)

    kill_workers(actors, [1, 2])

    assert killed == [actors[1], actors[2]]
    assert probes.count(("10.0.0.1", 701)) == 4, "kept waiting while the port still accepted"
    assert probes.count(("10.0.0.2", 702)) == 1
    assert ("10.0.0.0", 700) not in probes, "a worker nobody killed is not probed"
