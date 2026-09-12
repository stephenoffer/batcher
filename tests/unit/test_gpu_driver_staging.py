"""The device tier's last-resort path must never stage a stored relation on the driver.

`_translated` has three rungs: fan out across devices, run on one worker that reads for itself,
and — last — ship the table from the driver. The third is correct for an *in-memory* source,
whose rows are in this process by construction, and catastrophic for anything else: it calls
`list(source.read())`, so a 15M-row Parquet relation is materialized in the smallest node in the
cluster, which is precisely what the whole descriptor mechanism exists to prevent.

It was reached by inference rather than by asking. Rung two returns `None` for two unrelated
reasons — the source is in-memory and there is nothing to describe, or the dispatch to a worker
**failed** — and the caller read both as the first. So a device that ran out of memory answered
by staging the relation on the driver. Measured on a 30 GB head node, TPC-H q4 and q14 at sf10
were SIGKILLed by the kernel: no traceback, no fallback, the query simply gone.

These tests pin the predicate and the three call sites that gate on it.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.dist.gpu.dispatch import rows_already_on_driver
from batcher.io import InMemorySource

pytestmark = pytest.mark.unit


def _parquet_source(tmp_path):
    import batcher as bt

    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"x": list(range(64)), "y": list(range(64))}), path)
    return bt.read.parquet(str(path))._sources[0]


class _CountingSource:
    """A source that delegates everything and counts `read()` — the call under test.

    A wrapper rather than a monkeypatched attribute: the real sources use `__slots__`, so
    `source.read = ...` raises. Delegation by `__getattr__` also means this stays correct if the
    `Source` protocol grows a method.
    """

    def __init__(self, inner):
        self._inner = inner
        self.reads = 0

    def read(self, *args, **kwargs):
        self.reads += 1
        return self._inner.read(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# --- the predicate -----------------------------------------------------------


def test_an_in_memory_source_is_already_on_the_driver():
    src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
    assert rows_already_on_driver(src) is True


def test_a_file_source_is_not(tmp_path):
    """The case that mattered: a relation in storage must never be read into the driver here."""
    assert rows_already_on_driver(_parquet_source(tmp_path)) is False


def test_a_source_that_cannot_be_split_answers_conservatively():
    """Unknown must read as "not on the driver" — guessing wrong the other way is an OOM."""

    class _Opaque:
        def schema(self):
            raise RuntimeError("no")

    assert rows_already_on_driver(_Opaque()) is False


# --- the call sites ----------------------------------------------------------


def test_a_failed_worker_dispatch_does_not_stage_a_file_source(monkeypatch, tmp_path):
    """Rung two failing must go to the CPU engine, not to `list(source.read())` on the driver."""
    import batcher as bt
    from batcher.api.terminal.gpu_backend import translate

    ds = bt.read.parquet(str(_write(tmp_path))).group_by("k").agg(s=bt.col("v").sum())
    plan = ds._plan
    from batcher.core.gpu_plan import gpu_plan_ops

    matched = gpu_plan_ops(plan)
    assert matched is not None, "the fixture must reach the chain matcher for this to test it"

    counted = _CountingSource(ds._sources[matched[0].source_id])
    sources = [counted if i == matched[0].source_id else s for i, s in enumerate(ds._sources)]

    monkeypatch.setattr(translate, "_try_sharded_aggregate", lambda *a, **k: None)
    monkeypatch.setattr("batcher.dist.gpu.gpu_chain_on_worker", lambda *a, **k: None, raising=False)

    out = translate._translated(plan, sources, gpu_count=1, decision=_decision())
    assert out is None
    assert counted.reads == 0, "a stored relation was staged on the driver"


def _decision():
    from batcher.kyber.gpu.policy import GpuDecision

    return GpuDecision(True, False, "test", 64, 1, False)


def _write(tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"k": [1, 1, 2, 2] * 16, "v": [1.0, 2.0, 3.0, 4.0] * 16}), path)
    return path
