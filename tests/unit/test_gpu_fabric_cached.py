"""The node's wiring is read once, not on the critical path of every query.

`node_collective_env` measures **hardware**: which NIC each device is rail-aligned with, which
device pairs can reach each other peer-to-peer, and how far a GPUDirect path would have to
cross. None of that changes while a process runs.

It was read live on every call, and the caller is `gpu_task_options` — which a fan-out reaches
on every query, twice more when the shards are packed, and again for the admission probe. The
read walks `/sys`, enumerates the RDMA devices and prices every device-to-NIC pair; on a driver
with no device that is a sequence of misses ending in a subprocess that is not installed, which
is the slowest available way to learn nothing. Measured on this head node: 10 ms per call.
"""

from __future__ import annotations

import importlib

import pytest

# `import batcher.dist.gpu.fabric.collective_env as mod` binds the *function* of that name,
# which the package re-exports and which therefore shadows the submodule attribute.
mod = importlib.import_module("batcher.dist.gpu.fabric.collective_env")

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    mod.reset_node_collective_env()
    yield
    mod.reset_node_collective_env()


def test_the_machine_is_read_once(monkeypatch):
    reads = []
    monkeypatch.setattr(
        mod, "_measure_node_collective_env", lambda: reads.append(1) or {"NCCL_X": "1"}
    )
    for _ in range(25):
        assert mod.node_collective_env() == {"NCCL_X": "1"}
    assert len(reads) == 1


def test_the_caller_cannot_mutate_the_cached_reading(monkeypatch):
    """`gpu_task_runtime_env` merges into what it is handed, so it must get its own copy."""
    monkeypatch.setattr(mod, "_measure_node_collective_env", lambda: {"NCCL_X": "1"})
    first = mod.node_collective_env()
    first["NCCL_X"] = "tampered"
    first["INVENTED"] = "1"
    assert mod.node_collective_env() == {"NCCL_X": "1"}


def test_an_empty_reading_is_still_cached(monkeypatch):
    """A node whose fabric cannot be read is the *most* expensive case to re-measure — it is
    every probe missing — so it must not be the one case that repeats."""
    reads = []
    monkeypatch.setattr(mod, "_measure_node_collective_env", lambda: reads.append(1) or {})
    for _ in range(10):
        assert mod.node_collective_env() == {}
    assert len(reads) == 1


def test_resetting_reads_the_machine_again(monkeypatch):
    reads = []
    monkeypatch.setattr(mod, "_measure_node_collective_env", lambda: reads.append(1) or {})
    mod.node_collective_env()
    mod.reset_node_collective_env()
    mod.node_collective_env()
    assert len(reads) == 2
