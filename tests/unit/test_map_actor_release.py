"""A distributed `map_batches` actor closes its models before it is killed.

The local paths always call a class UDF's `close()`; the cluster's `_MapActor` never did,
because `ray.kill` ends the process without running anything in it. These tests drive the
actor class and the shutdown helper directly, with a stand-in for `ray`, so they run with no
cluster. They prove the wiring, not the cluster: the recorded cluster run in
`benchmarks/BENCHMARK_RESULTS.md` is what shows it on real actors.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import batcher as bt
from batcher.dist.executors.map import _MapActor, _release_and_kill

pytestmark = pytest.mark.unit


class _Model:
    """A load-once class UDF that counts its `close()` calls."""

    closed = 0

    def __call__(self, batch: Any) -> Any:
        return batch

    def close(self) -> None:
        type(self).closed += 1


def test_release_closes_every_built_model_once() -> None:
    _Model.closed = 0
    plan = bt.from_pydict({"x": [1, 2]}).map_batches(_Model).map_batches(_Model)._plan
    actor = _MapActor(plan)
    assert _Model.closed == 0  # building the actor loads the models and closes nothing
    actor.release()
    assert _Model.closed == 2
    actor.release()  # a scale-down racing a pool shutdown asks twice
    assert _Model.closed == 2


class _Remote:
    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def remote(self) -> Any:
        return self._fn()


class _FakeActor:
    def __init__(self, log: list[str], name: str, *, release_fails: bool = False) -> None:
        def release() -> str:
            log.append(f"release {name}")
            if release_fails:
                raise RuntimeError("actor already dead")
            return name

        self.release = _Remote(release)
        self.name = name


@pytest.fixture
def fake_ray(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    module = types.ModuleType("ray")
    module.wait = lambda refs, num_returns, timeout: log.append(f"wait {timeout}") or (refs, [])
    module.kill = lambda actor: log.append(f"kill {actor.name}")
    monkeypatch.setitem(sys.modules, "ray", module)
    return log


def test_shutdown_asks_for_release_before_it_kills(fake_ray: list[str]) -> None:
    _release_and_kill([_FakeActor(fake_ray, "a"), _FakeActor(fake_ray, "b")])
    assert fake_ray[:2] == ["release a", "release b"]
    assert fake_ray[2].startswith("wait ")
    assert float(fake_ray[2].split()[1]) > 0  # the wait is bounded
    assert fake_ray[3:] == ["kill a", "kill b"]


def test_a_failing_release_still_kills(fake_ray: list[str]) -> None:
    _release_and_kill([_FakeActor(fake_ray, "a", release_fails=True), _FakeActor(fake_ray, "b")])
    assert fake_ray[-2:] == ["kill a", "kill b"]


def test_an_empty_pool_touches_nothing(fake_ray: list[str]) -> None:
    _release_and_kill([])
    assert fake_ray == []
