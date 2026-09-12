"""Each worker's rayon width must follow the cores it was granted, not the driver's cores.

The per-node fleet plan cuts unequal slots for an unequal cluster -- on the 27-node cluster
these numbers come from, 3 cores on each 4-core node and 24 on the 96-core one. That plan only
reaches the data plane through `EngineConfig.parallelism`, and nothing else checks it: a
regression here changes no result, breaks no test, and silently runs every worker at the wrong
width. Both directions cost real throughput, and they cost it differently -- too many threads
thrash a small node (the `engine_config_json` docstring records 256 threads on 16 cores), too
few leave a big one idle.

The second half of the file pins the sharp edge found while measuring this: a non-zero
`execution.parallelism` is taken as an instruction and shipped to every worker unchanged, so
setting it on a mixed fleet replaces the per-node widths with one number. That is defensible
behaviour -- an explicit setting is an instruction -- but it is a footgun worth a failing test
if anyone changes which side wins by accident, in either direction.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from batcher.config import active_config, set_config
from batcher.dist.executors.ray_runtime.lifecycle import engine_config_json


@pytest.fixture
def parallelism(monkeypatch):
    """Set `execution.parallelism`, restoring the previous config afterwards."""
    original = active_config()

    def _set(value: int):
        base = active_config()
        set_config(
            dataclasses.replace(
                base, execution=dataclasses.replace(base.execution, parallelism=value)
            )
        )

    yield _set
    set_config(original)


def _shipped(grant: float | None) -> int:
    return json.loads(engine_config_json(num_cpus=grant))["parallelism"]


@pytest.mark.parametrize("grant", [1.0, 3.0, 15.0, 24.0, 96.0])
def test_the_width_follows_the_grant(parallelism, grant):
    """With the default 0, a worker's thread count is the cores it was admitted for."""
    parallelism(0)
    assert _shipped(grant) == int(grant)


def test_unequal_grants_get_unequal_widths(parallelism):
    """The point of the per-node plan: the big node's worker must not be sized like the small.

    Asserted as a relationship rather than two constants, because the failure that matters is
    the two collapsing to one number -- which is what happens whenever the grant stops being
    read, whatever value it collapses to.
    """
    parallelism(0)
    small, big = _shipped(3.0), _shipped(24.0)
    assert small < big, f"a 3-core and a 24-core worker were sized alike ({small} vs {big})"
    assert (small, big) == (3, 24)


def test_a_fractional_grant_still_opens_a_thread(parallelism):
    """A sub-core grant must floor to 1, never to 0 -- a 0-width pool computes nothing."""
    parallelism(0)
    assert _shipped(0.5) == 1


def test_an_explicit_setting_overrides_every_grant(parallelism):
    """A non-zero value is an instruction and is shipped verbatim to every worker.

    This is the footgun, pinned so a change of mind is deliberate: on a mixed fleet it gives a
    4-core node a 16-thread pool and caps a 24-core worker at the same 16.
    """
    parallelism(16)
    assert _shipped(3.0) == 16, "the explicit setting did not reach a small worker"
    assert _shipped(24.0) == 16, "the explicit setting did not cap a large worker"


def test_the_default_is_zero():
    """The pin only engages while the default is 0, so the default is part of the contract."""
    from batcher.config import ExecutionConfig

    assert ExecutionConfig().parallelism == 0
